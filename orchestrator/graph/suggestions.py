"""Follow-up suggestion helpers for PIBot graph."""

from __future__ import annotations

import datetime
import logging
import os
import re
from typing import Any, Dict, List, Optional

from orchestrator.classifier.intent_store import IntentStoreBase

try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore[assignment]

try:
    from langchain.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:
    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
    except Exception:
        SystemMessage = None  # type: ignore[assignment,misc]
        HumanMessage = None  # type: ignore[assignment,misc]

from .state import AgentState

logger = logging.getLogger(__name__)

_CHART_BLOCK_PATTERN = re.compile(r"##CHART_START(?P<body>.*?)##CHART_END", re.DOTALL)
_FOLLOWUP_MARKER_BLOCK_PATTERN = re.compile(r"##[A-Z_]+_START.*?##[A-Z_]+_END", re.DOTALL)

_FOLLOWUP_SYSTEM_PROMPT = """\
Eres un asistente económico. Genera preguntas de seguimiento para una conversación sobre datos (PIB/IMACEC).

Reglas obligatorias:
- Devuelve entre 2 y 3 preguntas.
- Deben estar directamente relacionadas con la pregunta y respuesta del turno actual.
- Deben ser útiles, accionables y no repetidas.
- Mantén cada pregunta en una sola línea y breve (máximo 16 palabras).
- Evita preguntas genéricas como "¿Algo más?".

Formato de salida obligatorio:
1. <pregunta>
2. <pregunta>
3. <pregunta opcional>

Devuelve solo las líneas de preguntas, sin explicación adicional.
"""


def _coerce_indicator_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            resolved = _coerce_indicator_value(item)
            if resolved:
                return resolved
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("standard_name", "normalized", "original", "text_normalized", "label"):
            candidate = value.get(key)
            resolved = _coerce_indicator_value(candidate)
            if resolved:
                return resolved
    return None


def _coerce_intent_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text.lower() or None
    if isinstance(value, dict):
        label = value.get("label")
        if isinstance(label, str):
            text = label.strip()
            return text.lower() or None
    return None


def _strip_marker_blocks(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    cleaned = _FOLLOWUP_MARKER_BLOCK_PATTERN.sub("", raw)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _normalize_followup_question(text: str) -> str:
    question = str(text or "").strip()
    question = re.sub(r"^[-*\s]+", "", question)
    question = re.sub(r"^\d+[\.)\-:]\s*", "", question)
    question = re.sub(r"\s+", " ", question).strip()
    if not question:
        return ""
    if not question.endswith("?"):
        question = question.rstrip(". !") + "?"
    return question


def _dedupe_questions(candidates: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for candidate in candidates:
        normalized = _normalize_followup_question(candidate)
        if not normalized:
            continue
        key = re.sub(r"[^a-z0-9]+", "", normalized.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _extract_followups_from_llm_text(text: str) -> List[str]:
    lines = str(text or "").splitlines()
    candidates: List[str] = []
    for line in lines:
        cleaned = _normalize_followup_question(line)
        if not cleaned:
            continue
        if len(cleaned) < 12:
            continue
        candidates.append(cleaned)
    return _dedupe_questions(candidates)


def _resolve_followup_limits() -> tuple[int, int]:
    raw_min = int(os.getenv("FOLLOWUP_SUGGESTIONS_MIN", "2"))
    raw_max = int(os.getenv("FOLLOWUP_SUGGESTIONS_MAX", "3"))
    max_count = max(2, min(3, raw_max))
    min_count = max(2, min(max_count, raw_min))
    return min_count, max_count


def _finalize_followups(*, primary: List[str], fallback: List[str]) -> List[str]:
    min_count, max_count = _resolve_followup_limits()
    merged = _dedupe_questions(primary + fallback)
    if len(merged) < min_count:
        defaults = _dedupe_questions(
            [
                "Quieres que busque los datos mas recientes?",
                "Te muestro la variacion del periodo anterior?",
                "Quieres comparar con otro periodo?",
            ]
        )
        merged = _dedupe_questions(merged + defaults)
    return merged[:max_count]


def _should_use_llm_followups() -> bool:
    flag = str(os.getenv("FOLLOWUP_SUGGESTIONS_USE_LLM", "1") or "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if ChatOpenAI is None or SystemMessage is None or HumanMessage is None:
        return False
    # Evita intentos de red cuando no hay credenciales.
    if not str(os.getenv("OPENAI_API_KEY", "") or "").strip():
        return False
    return True


def _build_llm_followup_messages(
    *,
    question: str,
    answer: str,
    indicator: Optional[str],
    component: Optional[str],
    intent: Optional[str],
) -> list:
    if SystemMessage is None or HumanMessage is None:
        return []

    answer_excerpt = _strip_marker_blocks(answer)[:900]
    user_content = (
        f"Pregunta del usuario: {str(question or '').strip()}\n"
        f"Respuesta entregada: {answer_excerpt}\n"
        f"Indicador detectado: {indicator or 'none'}\n"
        f"Componente detectado: {component or 'none'}\n"
        f"Intent detectado: {intent or 'none'}\n"
    )
    return [
        SystemMessage(content=_FOLLOWUP_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


def _generate_llm_followups(
    *,
    question: str,
    answer: str,
    indicator: Optional[str],
    component: Optional[str],
    intent: Optional[str],
) -> List[str]:
    if not _should_use_llm_followups():
        return []

    messages = _build_llm_followup_messages(
        question=question,
        answer=answer,
        indicator=indicator,
        component=component,
        intent=intent,
    )
    if not messages:
        return []

    model_name = str(os.getenv("FOLLOWUP_SUGGESTIONS_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")) or "gpt-4o-mini")
    temperature = float(os.getenv("FOLLOWUP_SUGGESTIONS_TEMPERATURE", "0.9"))
    max_tokens = int(os.getenv("FOLLOWUP_SUGGESTIONS_MAX_TOKENS", "160"))

    try:
        chat = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = chat.invoke(messages)
    except Exception:
        logger.debug("[GRAPH] LLM followup generation failed", exc_info=True)
        return []

    content = str(getattr(response, "content", "") or "")
    return _extract_followups_from_llm_text(content)


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
    return None


def _get_last_indicator_context(intent_store: Optional[IntentStoreBase], session_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not session_id or not intent_store or not hasattr(intent_store, "history"):
        return None
    try:
        records = intent_store.history(session_id, k=25)
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


def extract_chart_metadata_from_output(output: str) -> Optional[Dict[str, str]]:
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


def generate_suggested_questions(state: AgentState, intent_store: Optional[IntentStoreBase]) -> List[str]:
    suggestions: List[str] = []
    session_id = state.get("session_id")
    question_text = str(state.get("question") or "")
    answer_text = str(state.get("output") or "")
    entities = state.get("entities") or []
    primary_entity = next((ent for ent in entities if isinstance(ent, dict)), None)
    indicator = None
    component = None
    seasonality = None
    period: Optional[str] = None
    if primary_entity:
        indicator = primary_entity.get("indicator") or primary_entity.get("indicador")
        component = primary_entity.get("activity") or primary_entity.get("component")
        seasonality = primary_entity.get("seasonality")
    indicator = _coerce_indicator_value(indicator)
    component = _coerce_indicator_value(component)
    seasonality = _coerce_indicator_value(seasonality)
    classification = state.get("classification")
    intent = _coerce_intent_label(getattr(classification, "intent", None) if classification else None)

    if not indicator:
        last_ctx = _get_last_indicator_context(intent_store, session_id)
        if last_ctx:
            indicator = _coerce_indicator_value(last_ctx.get("indicator")) or indicator
            component = component or _coerce_indicator_value(last_ctx.get("component") or last_ctx.get("sector"))

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
        if seasonality:
            if "desestacionalizado" in seasonality.lower():
                suggestions.append(f"¿Cuál es el {indicator} sin desestacionalizar?")
            else:
                suggestions.append(f"¿Cuál es el {indicator} desestacionalizado?")
        else:
            suggestions.append(f"¿Cuál es el {indicator} desestacionalizado?")

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

        suggestions.append(f"¿Qué mide el {indicator}?")

        if period:
            suggestions.append(f"¿Cómo ha evolucionado el {indicator} en los últimos años?")

        if intent in ("methodology", "definition"):
            suggestions.insert(0, f"¿Cuál es el último valor del {indicator}?")

    heuristic_suggestions = _dedupe_questions(suggestions)
    llm_suggestions = _generate_llm_followups(
        question=question_text,
        answer=answer_text,
        indicator=indicator,
        component=component,
        intent=intent,
    )

    return _finalize_followups(primary=llm_suggestions, fallback=heuristic_suggestions)


__all__ = [
    "extract_chart_metadata_from_output",
    "generate_suggested_questions",
]
