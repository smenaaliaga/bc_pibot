"""LangGraph agent graph wiring PIBot orchestration."""

from __future__ import annotations

import datetime
import logging
import os
import re
import uuid
from typing import Annotated, Any, Dict, Iterable, List, Optional, TypedDict, Tuple

from langgraph.channels.topic import Topic
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import get_runtime
from langgraph.types import StreamWriter


from orchestrator.classifier.classifier_agent import (
    build_intent_info,
    classify_question_with_history,
    ClassificationResult,
    predict_with_interpreter,
    predict_with_router,
    predict_with_router_and_interpreter,
)
from orchestrator.classifier.intent_store import IntentStoreBase, create_intent_store
from orchestrator.llm.llm_adapter import LLMAdapter, build_llm
from orchestrator.memory.memory_adapter import MemoryAdapter
from orchestrator.rag.rag_factory import create_retriever
from orchestrator.data import flow_data
from orchestrator.routes import intent_router as intent_router

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    history: List[Dict[str, str]]
    context: Dict[str, Any]
    session_id: str
    user_turn_id: int
    facts: Dict[str, str]
    classification: ClassificationResult
    history_text: str
    intent_info: Dict[str, Any]
    route_decision: str
    output: str
    stream_chunks: Annotated[List[str], Topic(str, accumulate=True)]
    # Entidades normalizadas desde clasificación (transversal)
    period_context: Optional[Dict[str, Any]]
    indicator_context: Optional[str]
    component_context: Optional[str]
    seasonality_context: Optional[str]


def _emit_stream_chunk(chunk_text: str, writer: Optional[StreamWriter]) -> None:
    if not chunk_text:
        return

    resolved_writer: Optional[StreamWriter] = writer
    if resolved_writer is None:
        try:
            runtime = get_runtime()
            resolved_writer = getattr(runtime, "stream_writer", None)
        except Exception:  # pragma: no cover - runtime may be unavailable
            resolved_writer = None

    if resolved_writer:
        try:
            resolved_writer({"stream_chunks": chunk_text})
        except Exception:  # pragma: no cover - defensive logging only
            logger.debug("[GRAPH] stream writer emit failed", exc_info=True)


class _StreamChunkFilter:
    """Drop consecutive duplicate chunks from streaming routes."""

    def __init__(self) -> None:
        self._last_norm: Optional[str] = None
        self._last_was_whitespace = False

    def _normalize(self, chunk: str) -> str:
        return re.sub(r"\s+", " ", chunk).strip()

    def allow(self, chunk: Optional[str]) -> bool:
        if not chunk:
            return False
        if chunk.strip():
            normalized = self._normalize(chunk)
            if not normalized:
                return False
            if normalized == self._last_norm:
                return False
            self._last_norm = normalized
            self._last_was_whitespace = False
            return True
        # chunk contains only whitespace (newline/padding)
        if self._last_norm is None:
            return False
        if self._last_was_whitespace:
            return False
        self._last_was_whitespace = True
        return True


def _safe_memory_adapter() -> Any:
    require_pg = os.getenv("REQUIRE_PG_MEMORY", "0").lower() in {"1", "true", "yes", "on"}
    force_local = os.getenv("PG_FORCE_LOCALHOST", "1").lower() in {"1", "true", "yes", "on"}
    pg_dsn = None
    if force_local:
        pg_dsn = os.getenv("PG_LOCALHOST_DSN") or "postgresql://postgres:postgres@localhost:5432/pibot"
    else:
        pg_dsn = os.getenv("PG_DSN") or os.getenv("DATABASE_URL")
    try:
        if pg_dsn:
            return MemoryAdapter(pg_dsn=pg_dsn)
        return MemoryAdapter()
    except Exception as exc:  # pragma: no cover - best-effort init
        if require_pg:
            logger.error(
                "[GRAPH] MemoryAdapter require_pg enabled but failed: %s",
                exc,
            )
            raise
        logger.warning("[GRAPH] MemoryAdapter unavailable, using in-process fallback: %s", exc)
        return _InProcessMemoryAdapter()


class _InProcessMemoryAdapter:
    """Minimal drop-in replacement when persistent memory is unavailable."""

    def __init__(self, max_turns: int = 8):
        self.saver = None
        self._facts: Dict[str, Dict[str, str]] = {}
        self._history: Dict[str, List[Dict[str, str]]] = {}
        self._max_turns = max_turns

    def set_facts(self, session_id: str, facts: Dict[str, str]) -> None:
        if not session_id or not facts:
            return
        bucket = self._facts.setdefault(session_id, {})
        bucket.update({k: str(v) for k, v in facts.items()})

    def get_facts(self, session_id: str) -> Dict[str, str]:
        if not session_id:
            return {}
        return dict(self._facts.get(session_id, {}))

    def on_user_turn(self, session_id: str, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._append_history(session_id, "user", message, metadata=metadata)

    def on_assistant_turn(self, session_id: str, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._append_history(session_id, "assistant", message, metadata=metadata)

    def _append_history(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not session_id or not content:
            return
        history = self._history.setdefault(session_id, [])
        entry = {"role": role, "content": str(content)}
        if metadata:
            entry["metadata"] = dict(metadata)
        history.append(entry)
        if len(history) > 200:
            self._history[session_id] = history[-200:]

    def get_history_for_llm(self, session_id: str) -> List[Dict[str, str]]:
        if not session_id:
            return []
        history = self._history.get(session_id, [])
        return history[-self._max_turns :]

    def get_backend_status(self) -> Dict[str, Any]:
        return {
            "using_pg": False,
            "require_pg": False,
            "backend": "in-process",
            "facts_sessions": len(self._facts),
        }

    def clear_session(self, session_id: str) -> bool:
        if not session_id:
            return False
        try:
            self._facts.pop(session_id, None)
            self._history.pop(session_id, None)
            return True
        except Exception:
            return False


def _safe_retriever():
    try:
        return create_retriever()
    except Exception:  # pragma: no cover - retriever optional
        logger.exception("[GRAPH] Retriever initialization failed")
        return None


def _safe_intent_store() -> Optional[IntentStoreBase]:
    try:
        return create_intent_store()
    except Exception:
        logger.exception("[GRAPH] IntentStore initialization failed")
        return None


def _safe_llm(*, retriever=None, mode: str = "rag") -> Optional[LLMAdapter]:
    try:
        return build_llm(streaming=True, retriever=retriever, mode=mode)
    except Exception:  # pragma: no cover - avoid crashing import
        logger.exception("[GRAPH] LLMAdapter initialization failed")
        return None


_MEMORY: Optional[Any] = None
_RETRIEVER = None
_RAG_LLM = None
_FALLBACK_LLM = None
_INTENT_STORE: Optional[IntentStoreBase] = None

_CHART_BLOCK_PATTERN = re.compile(r"##CHART_START(?P<body>.*?)##CHART_END", re.DOTALL)


def _generate_suggested_questions(
    state: AgentState,
) -> List[str]:
    """Generate contextual follow-up questions based on normalized entities in memory."""
    suggestions: List[str] = []
    
    # Entradas desde el estado y memoria previa
    session_id = state.get("session_id")
    facts = state.get("facts") or {}
    indicator = state.get("indicator_context") or facts.get("indicator")
    component = state.get("component_context") or facts.get("component")
    seasonality = state.get("seasonality_context") or facts.get("seasonality")
    period = state.get("period_context") or facts.get("period")
    classification = state.get("classification")
    intent = getattr(classification, "intent", "") if classification else ""

    # Fallback: si no hay indicador en estado/facts, mirar historial del intent store
    if not indicator:
        last_ctx = _get_last_indicator_context(session_id)
        if last_ctx:
            indicator = last_ctx.get("indicator") or indicator
            component = component or last_ctx.get("component") or last_ctx.get("sector")

    # Si no hay indicador, ofrecer sugerencias genéricas
    if not indicator:
        suggestions.extend(
            [
                "¿Quieres que busque los datos más recientes?",
                "¿Te muestro un gráfico con la última variación?",
                "¿Prefieres consultar IMACEC o PIB?",
            ]
        )
    else:
        indicator_lower = indicator.lower()

        # Sugerencia sobre estacionalidad
        if seasonality:
            if "desestacionalizado" in seasonality.lower():
                suggestions.append(f"¿Cuál es el {indicator} sin desestacionalizar?")
            else:
                suggestions.append(f"¿Cuál es el {indicator} desestacionalizado?")
        else:
            suggestions.append(f"¿Cuál es el {indicator} desestacionalizado?")

        # Sugerencias específicas por indicador
        if "imacec" in indicator_lower:
            if not component or str(component).lower() == "total":
                suggestions.append("¿Cómo estuvo el IMACEC minero?")
            elif "minero" in str(component).lower():
                suggestions.append("¿Cómo estuvo el IMACEC no minero?")
            else:
                suggestions.append("¿Cómo estuvo el IMACEC total?")

        if "pib" in indicator_lower:
            if not component:
                suggestions.append("¿Cuál es la variación del PIB por sectores?")

        # Sugerencia sobre metodología
        suggestions.append(f"¿Qué mide el {indicator}?")

        # Sugerencia sobre periodos
        if period:
            suggestions.append(f"¿Cómo ha evolucionado el {indicator} en los últimos años?")

        # Si la intención es metodológica/definición, priorizar datos puntuales
        if intent and intent.lower() in ("methodology", "definition"):
            suggestions.insert(0, f"¿Cuál es el último valor del {indicator}?")

    # Deduplicar y limitar a 3
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        normalized_key = re.sub(r"[^a-z0-9]+", "", s.lower())
        if normalized_key not in seen:
            seen.add(normalized_key)
            unique_suggestions.append(s)

    return unique_suggestions[:3]


def _extract_chart_metadata_from_output(output: str) -> Optional[Dict[str, str]]:
    if not output:
        return None
    match = _CHART_BLOCK_PATTERN.search(output)
    if not match:
        return None
    body = match.group("body") or ""
    domain_match = re.search(r"domain\s*=\s*([A-Za-z0-9_\- ]+)", body, flags=re.IGNORECASE)
    if not domain_match:
        return None
    domain = domain_match.group(1).strip().upper()
    if not domain:
        return None
    return {
        "chart_domain": domain,
        "chart_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _ensure_backends() -> None:
    global _MEMORY, _RETRIEVER, _RAG_LLM, _FALLBACK_LLM, _INTENT_STORE
    if _MEMORY is None:
        _MEMORY = _safe_memory_adapter()
    if _RETRIEVER is None:
        _RETRIEVER = _safe_retriever()
    if _RAG_LLM is None:
        _RAG_LLM = _safe_llm(retriever=_RETRIEVER, mode="rag")
    if _FALLBACK_LLM is None:
        _FALLBACK_LLM = _safe_llm(retriever=None, mode="fallback")
    if _INTENT_STORE is None:
        _INTENT_STORE = _safe_intent_store()


def _ensure_list(history: Optional[Iterable[Dict[str, str]]]) -> List[Dict[str, str]]:
    if not history:
        return []
    try:
        return [dict(item) for item in history]
    except Exception:
        return list(history)  # type: ignore[arg-type]


def _ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_choice_content(choice: Any) -> Iterable[str]:
    if choice is None:
        return []
    snippets: List[str] = []
    delta = getattr(choice, "delta", None)
    if delta is not None:
        content = getattr(delta, "content", None)
        if isinstance(content, list):
            for item in content:
                snippets.append(_ensure_text(getattr(item, "text", getattr(item, "content", item))))
        elif content:
            snippets.append(_ensure_text(content))
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            snippets.append(_ensure_text(content))
    text = getattr(choice, "text", None)
    if text:
        snippets.append(_ensure_text(text))
    return [s for s in snippets if s]


def _yield_openai_stream_chunks(stream: Any) -> Iterable[str]:
    if stream is None:
        return []
    if isinstance(stream, (str, bytes)):
        return [str(stream)]

    try:
        iterator = iter(stream)
    except TypeError:
        iterator = None

    chunks: List[str] = []
    if iterator is not None and not isinstance(stream, dict):
        for chunk in iterator:
            choices = getattr(chunk, "choices", None)
            if choices:
                for choice in choices:
                    chunks.extend(_extract_choice_content(choice))
            else:
                chunks.append(_ensure_text(getattr(chunk, "content", getattr(chunk, "text", chunk))))
        return [c for c in chunks if c]

    choices = getattr(stream, "choices", None)
    if choices:
        for choice in choices:
            chunks.extend(_extract_choice_content(choice))
    else:
        content = getattr(stream, "content", getattr(stream, "text", None))
        if content:
            chunks.append(_ensure_text(content))
    return [c for c in chunks if c]


def ingest_node(state: AgentState) -> AgentState:
    question = _ensure_text(state.get("question", "")).strip()
    history = _ensure_list(state.get("history"))
    context = dict(state.get("context") or {})
    session_id = context.get("session_id") or f"graph-{uuid.uuid4().hex}"
    context["session_id"] = session_id
    facts: Dict[str, str] = {}
    memory_history: List[Dict[str, str]] = []
    user_turn_id: Optional[int] = None
    if _MEMORY and session_id:
        try:
            if question:
                user_turn_id = _MEMORY.on_user_turn(session_id, question)
            facts = _MEMORY.get_facts(session_id)
            memory_history = _MEMORY.get_window_for_llm(session_id)
        except Exception:
            logger.debug("[GRAPH] Unable to read memory facts", exc_info=True)
    if memory_history:
        history = memory_history
    next_state: AgentState = {
        "question": question,
        "history": history,
        "context": context,
        "session_id": session_id,
        "facts": facts,
    }
    if user_turn_id is not None:
        next_state["user_turn_id"] = user_turn_id
    return next_state


def intent_node(state: AgentState) -> AgentState:
    """Nodo simple: usa IntentRouter y decide la ruta siguiente.

    - value  -> data
    - methodology -> rag
    - otro  -> fallback
    """
    question = state.get("question", "")
    session_id = state.get("session_id", "")

    results = predict_with_router(question)
    intent_label = getattr(getattr(results, "intent", None), "label", "") or ""

    if intent_label == "value":
        decision = "data"
    elif intent_label == "methodology":
        decision = "rag"
    else:
        decision = "fallback"

    logger.info(
        "[INTENT_NODE] PIBOT_INTENT_ROUTE | intent=%s | decision=%s",
        intent_label,
        decision,
    )

    # Persistir intención mínima en memoria (opcional)
    if session_id and _MEMORY:
        try:
            _MEMORY.set_facts(session_id, {"intent": intent_label})
        except Exception:
            logger.debug("[INTENT_NODE] Unable to persist intent fact", exc_info=True)

    return {"route_decision": decision, "intent": intent_label}


def _persist_intent_event(
    state: AgentState,
    classification: Optional[ClassificationResult],
    intent_info: Optional[Dict[str, Any]],
) -> None:
    if not _INTENT_STORE:
        return
    session_id = state.get("session_id")
    turn_id = state.get("user_turn_id")
    if not session_id or not turn_id:
        return
    payload = intent_info or {}
    raw_intent = payload.get("intent")
    if not raw_intent and classification:
        raw_intent = getattr(classification, "intent", None)
    intent = _ensure_text(raw_intent)
    if not intent:
        return
    score_raw = payload.get("score")
    try:
        score = float(score_raw) if score_raw is not None else 1.0
    except (TypeError, ValueError):
        score = 1.0
    spans = payload.get("spans") or []
    base_entities = payload.get("entities") or {}
    entities = dict(base_entities) if isinstance(base_entities, dict) else {}
    extra_jointbert: Dict[str, Any] = {}
    if classification:
        if getattr(classification, "intent", None):
            extra_jointbert["raw_intent"] = classification.intent
        if getattr(classification, "confidence", None) is not None:
            extra_jointbert["confidence"] = classification.confidence
        if getattr(classification, "entities", None):
            extra_jointbert["entities"] = classification.entities
        if getattr(classification, "normalized", None):
            extra_jointbert["normalized"] = classification.normalized
    if extra_jointbert:
        entities = dict(entities)
        entities["jointbert"] = extra_jointbert
    try:
        _INTENT_STORE.record(
            session_id,
            intent,
            score,
            spans=spans,
            entities=entities,
            turn_id=int(turn_id),
        )
        logger.info(
            "[GRAPH] intent stored | session=%s turn=%s intent=%s score=%.3f",
            session_id,
            turn_id,
            intent,
            score,
        )
    except Exception:
        logger.debug("[GRAPH] Unable to persist intent event", exc_info=True)


def _coerce_indicator_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("standard_name", "normalized", "original", "text_normalized", "label"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _extract_indicator_context_from_entities(entities: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(entities, dict):
        return None
    jointbert = entities.get("jointbert")
    if not isinstance(jointbert, dict):
        return None
    normalized = jointbert.get("normalized")
    if not isinstance(normalized, dict):
        return None
    indicator = _coerce_indicator_value(normalized.get("indicator"))
    sector = _coerce_indicator_value(normalized.get("sector"))
    component = _coerce_indicator_value(normalized.get("component"))
    if not sector:
        sector = component
    if indicator or sector:
        context: Dict[str, str] = {}
        if indicator:
            context["indicator"] = indicator
        if sector:
            context["sector"] = sector
            context.setdefault("component", sector)
        elif component:
            context["component"] = component
        return context
    raw_entities = jointbert.get("entities")
    if isinstance(raw_entities, dict):
        indicator = _coerce_indicator_value(raw_entities.get("indicator"))
        if indicator:
            return {"indicator": indicator}
    return None


def _get_last_indicator_context(session_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not session_id or not _INTENT_STORE or not hasattr(_INTENT_STORE, "history"):
        return None
    try:
        records = _INTENT_STORE.history(session_id, k=25)
    except Exception:
        logger.debug("[GRAPH] Unable to read intent history for indicator context", exc_info=True)
        return None
    for record in reversed(records or []):
        context = _extract_indicator_context_from_entities(getattr(record, "entities", None))
        if context:
            try:
                logger.debug("[GRAPH] indicator_context_resolved session_id=%s context=%s", session_id, context)
            except Exception:
                logger.exception("[GRAPH] Failed to log indicator_context")
            return context
    return None


def intent_shortcuts_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    classification = state.get("classification")
    question = state.get("question", "")
    history_text = state.get("history_text", "")
    session_id = state.get("session_id")
    if not classification:
        return
    
    direct_iter = None
    try:
        direct_iter = intent_router.route_intents(
            classification,
            question,
            history_text,
            memory=_MEMORY,
            session_id=session_id,
            facts=state.get("facts"),
        )
    except Exception:
        logger.exception("[GRAPH] intent_router failed")
    
    if direct_iter is None:
        return
    collected: List[str] = []
    memory_metadata = getattr(direct_iter, "metadata", None)
    chunk_filter = _StreamChunkFilter()
    # Consumir stream: valida chunks, filtra duplicados, emite en tiempo real,
    # y acumula para retornar output final
    for chunk in direct_iter:
        chunk_text = _ensure_text(chunk)
        if chunk_text and chunk_filter.allow(chunk_text):
            collected.append(chunk_text)
            _emit_stream_chunk(chunk_text, writer)
    if collected:
        yield {
            "output": "".join(collected),
            "route_decision": "direct",
            "memory_metadata": memory_metadata,
        }


def data_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    """Nodo de datos: obtiene series económicas reales.
    
    Estado disponible:
    - question: pregunta del usuario
    - session_id: identificador de sesión
    - facts: hechos acumulados en memoria
    """
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    
    # Usar predict_with_interpreter para obtener clasificación desde question
    result = predict_with_interpreter(question)
    
    logger.info(
        "[=====DATA_NODE=====] PIBOT_SERIES_INTERPRETER | %s",
        vars(result) if hasattr(result, '__dict__') else result
    )
    
    if not result:
        text = "[GRAPH] No se recibió clasificación para el nodo DATA."
        logger.warning("[=====DATA_NODE=====] %s", text)
        _emit_stream_chunk(text, writer)
        return {"output": text}
    collected: List[str] = []
    chunk_filter = _StreamChunkFilter()
    
    # OBTENER FECHA
    from orchestrator.data.date_parser import parse_point_date, parse_range
    
    parsed_point: Optional[str] = None
    parsed_range: Optional[Tuple[str, str]] = None
    if result.req_form.label == "point":
        parsed_point = parse_point_date(question, result.frequency.label)
        if not parsed_point:
            logger.warning("[=====DATA_NODE=====] parse_point_date falló | question=%s | frequency=%s", question, result.frequency.label)
        logger.info("[=====DATA_NODE=====] parse_point_date exitoso | parsed_point=%s | frequency=%s", parsed_point, result.frequency.label)
    elif result.req_form.label == "range":
        parsed_range = parse_range(question, result.frequency.label)
        if not parsed_range:
            logger.warning("[=====DATA_NODE=====] parse_range falló | question=%s | frequency=%s", question, result.frequency.label)
        
        logger.info("[=====DATA_NODE=====] parse_range exitoso | parsed_range=%s | frequency=%s", parsed_range, result.frequency.label)
        
    # OBTENER SERIES
    from registry import get_catalog_service, get_bde_client
    
    catalog = get_catalog_service(catalog_path="catalog/series_catalog.json")
    bde = get_bde_client()
    
    # Locate matching series in the catalog.
    match = catalog.find_series(
        indicator=result.indicator.label,
        metric_type=result.metric_type.label,
        seasonality=result.seasonality.label,
        activity=result.activity.label,
        frequency=result.frequency.label,
    )
    
    logger.info(
        "[=====DATA_NODE=====] catalog.find_series | match=%s",
        match if match else "(no encontrado)"
    )
    
    
    # OBTENER DATOS
    from orchestrator.data.common import normalize_series_obs
    
    # Fetch and normalize observations from BDE/local source.
    obs = bde.fetch_series(match["id"])  # list of {date, value}
    normalized = normalize_series_obs(obs, result.frequency.label)  # list of (datetime, float)
    
    # MOSTRAR DATOS
    from orchestrator.data.variations import yoy as compute_yoy, prev_period as compute_prev, compute_variations_for_range
    
    if result.req_form.label == "latest":
        d, v = normalized[-1]
        output_dict = {"date": d.strftime("%d-%m-%Y"), "value": v}
        if result.calc_mode.label == "yoy":
            var = compute_yoy(normalized, d, v)
            if var is not None:
                output_dict["yoy"] = var
        elif result.calc_mode.label == "prev_period":
            var = compute_prev(normalized, d, v)
            if var is not None:
                output_dict["prev_period"] = var
        payload = {
            "intent": "value",
            "classification": {
                "indicator": result.indicator.label,
                "metric_type": result.metric_type.label,
                "seasonality": result.seasonality.label,
                "activity": result.activity.label,
                "frequency": result.frequency.label,
                "calc_mode": result.calc_mode.label,
                "req_form": result.req_form.label,
            },
            "series": match["id"],
            "parsed_point": None,
            "parsed_range": None,
            "result": output_dict,
        }
        logger.info(f"[=====DATA_NODE=====] Result (latest): {payload}")
        # return payload

    elif result.req_form.label == "point":
        # parsed_point already computed above
        target_dt = datetime.datetime.strptime(parsed_point, "%d-%m-%Y")  # type: ignore[arg-type]
        d_target = None
        for d, v in normalized:
            if d == target_dt:
                d_target = d
                v_target = v
                break
        if d_target is None:
            return {"error": f"No existe observación para la fecha {parsed_point}."}
        output_dict = {"date": d_target.strftime("%d-%m-%Y"), "value": v_target}
        if result.calc_mode.label == "yoy":
            var = compute_yoy(normalized, d_target, v_target)
            if var is not None:
                output_dict["yoy"] = var
        elif result.calc_mode.label == "prev_period":
            var = compute_prev(normalized, d_target, v_target)
            if var is not None:
                output_dict["prev_period"] = var
        payload = {
            "intent": "value",
            "classification": {
                "indicator": result.indicator.label,
                "metric_type": result.metric_type.label,
                "seasonality": result.seasonality.label,
                "activity": result.activity.label,
                "frequency": result.frequency.label,
                "calc_mode": result.calc_mode.label,
                "req_form": result.req_form.label,
            },
            "series": match["id"],
            "parsed_point": parsed_point,
            "parsed_range": None,
            "result": output_dict,
        }
        logger.info(f"[=====DATA_NODE=====] Result (point): {payload}")
        # return payload

    elif result.req_form.label == "range":
        # parsed_range already computed above
        start_str, end_str = parsed_range  # type: ignore[misc]
        from_dt = None
        to_dt = None
        
        logger.info(f"[=====DATA_NODE=====] Attempting to parse range | start_str={start_str} end_str={end_str}")
        
        try:
            from_dt = datetime.datetime.strptime(start_str, "%d-%m-%Y")
            to_dt = datetime.datetime.strptime(end_str, "%d-%m-%Y")
            logger.info(f"[=====DATA_NODE=====] Successfully parsed range dates | from_dt={from_dt} to_dt={to_dt}")
        except Exception as e:
            logger.error(f"[=====DATA_NODE=====] Failed to parse range dates | start_str={start_str} end_str={end_str} error={type(e).__name__}: {e}")
            from_dt = normalized[0][0]
            to_dt = normalized[-1][0]
            logger.info(f"[=====DATA_NODE=====] Using fallback range (full history) | from_dt={from_dt} to_dt={to_dt}")

        logger.info(f"[=====DATA_NODE=====] Calling compute_variations_for_range | normalized_size={len(normalized)} from_dt={from_dt} to_dt={to_dt}")
        values = compute_variations_for_range(normalized, result.calc_mode.label, from_dt, to_dt)
        logger.info(f"[=====DATA_NODE=====] Computed range values | count={len(values)} first_date={values[0]['date'] if values else 'N/A'} last_date={values[-1]['date'] if values else 'N/A'}")
        
        payload = {
            "intent": "value",
            "classification": {
                "indicator": result.indicator.label,
                "metric_type": result.metric_type.label,
                "seasonality": result.seasonality.label,
                "activity": result.activity.label,
                "frequency": result.frequency.label,
                "calc_mode": result.calc_mode.label,
                "req_form": result.req_form.label,
            },
            "series": match["id"],
            "parsed_point": None,
            "parsed_range": parsed_range,
            "result": values,
        }
        logger.info(f"[=====DATA_NODE=====] Result (range): payload with {len(values)} observations")


    try: 
        # Invocar flujo de datos directamente: usa payload con serie, clasificación y resultados ya procesados
        stream = flow_data.stream_data_flow(
            payload,
            session_id=session_id,
        )
        # Consumir stream: valida chunks, filtra duplicados, emite en tiempo real,
        # y acumula para retornar output final
        for chunk in stream:
            chunk_text = _ensure_text(chunk)
            if chunk_text and chunk_filter.allow(chunk_text):
                collected.append(chunk_text)
                _emit_stream_chunk(chunk_text, writer)
    except Exception:
        logger.exception("[DATA_NODE] Flujo fallido")
        if not collected:
            fallback = "Ocurrió un problema al obtener los datos solicitados."
            collected.append(fallback)
            _emit_stream_chunk(fallback, writer)
        return {"output": "".join(collected)}

    # Guardar clasificación en memoria para reutilización en siguientes turnos
    if _MEMORY and session_id:
        try:
            classification_facts = {
                "indicator": result.indicator.label,
                "metric_type": result.metric_type.label,
                "seasonality": result.seasonality.label,
                "activity": result.activity.label,
                "frequency": result.frequency.label,
                "calc_mode": result.calc_mode.label,
                "req_form": result.req_form.label,
            }
            _MEMORY.set_facts(session_id, classification_facts)
            logger.info("[DATA_NODE] Clasificación guardada en memoria | session=%s | facts=%s", session_id, list(classification_facts.keys()))
        except Exception:
            logger.debug("[DATA_NODE] Unable to persist classification facts", exc_info=True)

    logger.info("[DATA_NODE] Completado | chunks=%d", len(collected))
    return {"output": "".join(collected)}


def _run_llm(
    state: AgentState,
    adapter: Optional[LLMAdapter],
    *,
    writer: Optional[StreamWriter] = None,
):
    question = state.get("question", "")
    history = state.get("history") or []
    intent_info = state.get("intent_info")
    if not question:
        text = "No recibí una pregunta para responder."
        _emit_stream_chunk(text, writer)
        return {"output": text}
    if adapter is None:
        text = "No pude inicializar el modelo de lenguaje para esta ruta."
        _emit_stream_chunk(text, writer)
        return {"output": text}
    collected: List[str] = []
    chunk_filter = _StreamChunkFilter()

    try:
        for chunk in adapter.stream(question, history=history, intent_info=intent_info):
            chunk_text = _ensure_text(chunk)
            if not chunk_text:
                continue
            try:
                if os.getenv("STREAM_CHUNK_LOGS", "0").lower() in {"1", "true", "yes", "on"}:
                    logger.debug("[GRAPH_LLM_CHUNK] %s", chunk_text[:200])
            except Exception:
                pass
            if not chunk_filter.allow(chunk_text):
                continue
            collected.append(chunk_text)
            _emit_stream_chunk(chunk_text, writer)
    except Exception:
        logger.exception("[GRAPH] LLM streaming failed")
        if not collected:
            fallback = "Tuve un problema generando la respuesta."
            collected.append(fallback)
            _emit_stream_chunk(fallback, writer)
        return {"output": "".join(collected)}
    return {"output": "".join(collected)}


def rag_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    """Nodo RAG: búsqueda y generación con contexto.
    
    Estado disponible:
    - question: pregunta del usuario
    - intent: intención clasificada (ej: 'methodology', 'definition')
    - history: historial de conversación
    - session_id: identificador de sesión
    - facts: hechos acumulados en memoria
    """
    question = state.get("question", "")
    intent = state.get("intent", "")
    history = state.get("history", [])
    session_id = state.get("session_id", "")
    
    logger.info(
        "[RAG_NODE] Iniciando | intent=%s | question=%s | history_len=%d | session=%s",
        intent,
        question[:100] if question else "(vacío)",
        len(history),
        session_id[:12] if session_id else "(vacío)",
    )
    
    result = _run_llm(state, _RAG_LLM, writer=writer)
    
    logger.info(
        "[RAG_NODE] Completado | output_len=%d",
        len(result.get("output", "")),
    )
    return result


def fallback_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    return _run_llm(state, _FALLBACK_LLM, writer=writer)


def direct_node(state: AgentState) -> AgentState:
    return {"output": state.get("output", "")}


def memory_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id")
    output = state.get("output", "")
    metadata = dict(state.get("memory_metadata") or {})
    chart_meta = _extract_chart_metadata_from_output(output)
    if chart_meta:
        metadata.update(chart_meta)
    has_chart_metadata = bool(metadata.get("chart_domain"))
    if _MEMORY and session_id:
        try:
            if output:
                if metadata:
                    _MEMORY.on_assistant_turn(session_id, output, metadata=metadata)
                else:
                    _MEMORY.on_assistant_turn(session_id, output)
            if has_chart_metadata:
                _persist_chart_state(session_id, metadata)
            else:
                _maybe_clear_chart_state(session_id, state.get("facts"))
            
            # Persistir entidades normalizadas desde state en facts (transversal)
            entities_to_persist: Dict[str, Any] = {}
            if state.get("indicator_context"):
                entities_to_persist["indicator"] = state["indicator_context"]
            if state.get("component_context"):
                entities_to_persist["component"] = state["component_context"]
            if state.get("seasonality_context"):
                entities_to_persist["seasonality"] = state["seasonality_context"]
            if state.get("period_context"):
                entities_to_persist["period"] = state["period_context"]
            if entities_to_persist:
                try:
                    _MEMORY.set_facts(session_id, entities_to_persist)  # type: ignore
                    logger.debug("[GRAPH] Entidades normalizadas guardadas en facts: %s", list(entities_to_persist.keys()))
                except Exception:
                    logger.debug("[GRAPH] Unable to persist normalized entities to facts", exc_info=True)
            
            checkpoint_payload = {
                "question": state.get("question"),
                "output": output,
                "route_decision": state.get("route_decision"),
            }
            classification = state.get("classification")
            intent = getattr(classification, "intent", None) if classification else None
            if intent:
                checkpoint_payload["intent"] = intent
            facts = state.get("facts")
            if facts:
                checkpoint_payload["facts"] = facts
            _MEMORY.save_checkpoint(
                session_id,
                checkpoint_payload,
                metadata={"source": "memory_node"},
            )
        except Exception:
            logger.debug("[GRAPH] Unable to persist assistant turn", exc_info=True)
    
    # Generar preguntas sugeridas basadas en entidades
    suggested_questions = _generate_suggested_questions(state)
    if not suggested_questions:
        # Fallback absoluto para garantizar botones
        suggested_questions = [
            "¿Quieres que busque los datos más recientes?",
            "¿Te muestro un gráfico con la última variación?",
            "¿Prefieres consultar IMACEC o PIB?",
        ]
    followup_block = "\n\n##FOLLOWUP_START\n"
    for i, question in enumerate(suggested_questions[:3], start=1):
        followup_block += f"suggestion_{i}={question}\n"
    followup_block += "##FOLLOWUP_END"
    output = output + followup_block
    
    return {"output": output}


def _persist_chart_state(session_id: str, metadata: Dict[str, Any]) -> None:
    if not _MEMORY or not hasattr(_MEMORY, "set_facts"):
        return
    domain = str(metadata.get("chart_domain") or "").upper()
    if not domain:
        return
    ts_value = metadata.get("chart_ts") or datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        _MEMORY.set_facts(
            session_id,
            {
                "chart_last_domain": domain,
                "chart_last_ts": str(ts_value),
            },
        )  # type: ignore
    except Exception:
        logger.debug("[GRAPH] Unable to persist chart facts", exc_info=True)


def _maybe_clear_chart_state(session_id: str, prior_facts: Optional[Dict[str, str]]) -> None:
    if not _MEMORY or not hasattr(_MEMORY, "set_facts"):
        return
    facts = prior_facts or {}
    if not facts.get("chart_last_domain"):
        return
    try:
        _MEMORY.set_facts(session_id, {"chart_last_domain": "", "chart_last_ts": ""})  # type: ignore
    except Exception:
        logger.debug("[GRAPH] Unable to clear chart facts", exc_info=True)


def build_graph():
    _ensure_backends()
    builder = StateGraph(AgentState)
    builder.add_node("ingest", ingest_node)
    builder.add_node("intent", intent_node)
    builder.add_node("direct", direct_node)
    builder.add_node("data", data_node)
    builder.add_node("rag", rag_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("memory", memory_node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "intent")
    # intent_node ya decide la ruta (data, rag o fallback) y la pone en route_decision
    builder.add_conditional_edges(
        "intent",
        lambda state: state.get("route_decision", "fallback"),
        {"data": "data", "rag": "rag", "fallback": "fallback"},
    )
    builder.add_edge("data", "memory")
    builder.add_edge("rag", "memory")
    builder.add_edge("fallback", "memory")
    builder.add_edge("memory", END)

    checkpointer = getattr(_MEMORY, "saver", None)
    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "AgentState",
    "build_graph",
    "_yield_openai_stream_chunks",
]
