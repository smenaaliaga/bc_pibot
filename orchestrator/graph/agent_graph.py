"""LangGraph agent graph wiring PIBot orchestration."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Annotated, Any, Dict, Iterable, List, Optional, TypedDict

from langgraph.channels.topic import Topic
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import get_runtime
from langgraph.types import StreamWriter

from orchestrator.intents.classifier_agent import (
    build_intent_info,
    classify_question_with_history,
)
from orchestrator.llm.llm_adapter import LLMAdapter, build_llm
from orchestrator.memory.memory_adapter import MemoryAdapter
from orchestrator.rag.rag_factory import create_retriever
from orchestrator.routes import data_router, intent_router
from prompt import ClassificationResult

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    history: List[Dict[str, str]]
    context: Dict[str, Any]
    session_id: str
    facts: Dict[str, str]
    classification: ClassificationResult
    history_text: str
    intent_info: Dict[str, Any]
    route_decision: str
    output: str
    stream_chunks: Annotated[List[str], Topic(str, accumulate=True)]


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

    def on_user_turn(self, session_id: str, message: str) -> None:
        self._append_history(session_id, "user", message)

    def on_assistant_turn(self, session_id: str, message: str) -> None:
        self._append_history(session_id, "assistant", message)

    def _append_history(self, session_id: str, role: str, content: str) -> None:
        if not session_id or not content:
            return
        history = self._history.setdefault(session_id, [])
        history.append({"role": role, "content": str(content)})
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


def _safe_retriever():
    try:
        return create_retriever()
    except Exception:  # pragma: no cover - retriever optional
        logger.exception("[GRAPH] Retriever initialization failed")
        return None


def _safe_llm(*, retriever=None) -> Optional[LLMAdapter]:
    try:
        return build_llm(streaming=True, retriever=retriever)
    except Exception:  # pragma: no cover - avoid crashing import
        logger.exception("[GRAPH] LLMAdapter initialization failed")
        return None


_MEMORY: Optional[Any] = None
_RETRIEVER = None
_RAG_LLM = None
_FALLBACK_LLM = None


def _ensure_backends() -> None:
    global _MEMORY, _RETRIEVER, _RAG_LLM, _FALLBACK_LLM
    if _MEMORY is None:
        _MEMORY = _safe_memory_adapter()
    if _RETRIEVER is None:
        _RETRIEVER = _safe_retriever()
    if _RAG_LLM is None:
        _RAG_LLM = _safe_llm(retriever=_RETRIEVER)
    if _FALLBACK_LLM is None:
        _FALLBACK_LLM = _safe_llm(retriever=None)


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
    if _MEMORY and session_id:
        try:
            if question:
                _MEMORY.on_user_turn(session_id, question)
            facts = _MEMORY.get_facts(session_id)
        except Exception:
            logger.debug("[GRAPH] Unable to read memory facts", exc_info=True)
    return {
        "question": question,
        "history": history,
        "context": context,
        "session_id": session_id,
        "facts": facts,
    }


def classify_node(state: AgentState) -> AgentState:
    question = state.get("question", "")
    history = state.get("history") or []
    try:
        classification, history_text = classify_question_with_history(question, history)
    except Exception:
        logger.exception("[GRAPH] classify_question_with_history failed")
        classification = ClassificationResult(query_type="METHODOLOGICAL")
        history_text = ""
    intent_info = build_intent_info(classification)
    facts = state.get("facts") or {}
    if facts:
        intent_info = intent_info or {}
        intent_info.setdefault("facts", facts)
    return {
        "classification": classification,
        "history_text": history_text,
        "intent_info": intent_info,
    }


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
        )
    except Exception:
        logger.exception("[GRAPH] intent_router.route_intents failed")
    if direct_iter is None:
        return
    collected: List[str] = []
    for chunk in direct_iter:
        chunk_text = _ensure_text(chunk)
        if not chunk_text:
            continue
        collected.append(chunk_text)
        _emit_stream_chunk(chunk_text, writer)
        yield {"stream_chunks": chunk_text}
    if collected:
        yield {
            "output": "".join(collected),
            "route_decision": "direct",
        }


def data_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    classification = state.get("classification")
    question = state.get("question", "")
    history_text = state.get("history_text", "")
    if not classification:
        text = "No pude clasificar la consulta para obtener datos."
        _emit_stream_chunk(text, writer)
        yield {"stream_chunks": text}
        return {"output": text}
    collected: List[str] = []
    try:
        stream = data_router.stream_data_flow(classification, question, history_text)
        for chunk in stream:
            chunk_text = _ensure_text(chunk)
            if not chunk_text:
                continue
            collected.append(chunk_text)
            _emit_stream_chunk(chunk_text, writer)
            yield {"stream_chunks": chunk_text}
    except Exception:
        logger.exception("[GRAPH] data route failed")
        if not collected:
            fallback = "Ocurrió un problema al obtener los datos solicitados."
            collected.append(fallback)
            _emit_stream_chunk(fallback, writer)
            yield {"stream_chunks": fallback}
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
        yield {"stream_chunks": text}
        return {"output": text}
    if adapter is None:
        text = "No pude inicializar el modelo de lenguaje para esta ruta."
        _emit_stream_chunk(text, writer)
        yield {"stream_chunks": text}
        return {"output": text}
    collected: List[str] = []
    try:
        for chunk in adapter.stream(question, history=history, intent_info=intent_info):
            chunk_text = _ensure_text(chunk)
            if not chunk_text:
                continue
            try:
                logger.debug("[GRAPH_LLM_CHUNK] %s", chunk_text[:200])
            except Exception:
                pass
            collected.append(chunk_text)
            _emit_stream_chunk(chunk_text, writer)
            yield {"stream_chunks": chunk_text}
    except Exception:
        logger.exception("[GRAPH] LLM streaming failed")
        if not collected:
            fallback = "Tuve un problema generando la respuesta."
            collected.append(fallback)
            _emit_stream_chunk(fallback, writer)
            yield {"stream_chunks": fallback}
    return {"output": "".join(collected)}


def rag_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    yield from _run_llm(state, _RAG_LLM, writer=writer)


def fallback_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    yield from _run_llm(state, _FALLBACK_LLM, writer=writer)


def direct_node(state: AgentState) -> AgentState:
    return {"output": state.get("output", "")}


def memory_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id")
    output = state.get("output", "")
    if _MEMORY and session_id and output:
        try:
            _MEMORY.on_assistant_turn(session_id, output)
        except Exception:
            logger.debug("[GRAPH] Unable to persist assistant turn", exc_info=True)
    return {"output": output}


def route_decider(state: AgentState) -> str:
    decision = _ensure_text(state.get("route_decision", "")).strip()
    if decision:
        return decision
    classification = state.get("classification")
    query_type = _ensure_text(getattr(classification, "query_type", "")).upper()
    if query_type == "DATA":
        return "data"
    if query_type == "METHODOLOGICAL":
        return "rag"
    return "fallback"


def build_graph():
    _ensure_backends()
    builder = StateGraph(AgentState)
    builder.add_node("ingest", ingest_node)
    builder.add_node("classify", classify_node)
    builder.add_node("intent_shortcuts", intent_shortcuts_node)
    builder.add_node("direct", direct_node)
    builder.add_node("data", data_node)
    builder.add_node("rag", rag_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("memory", memory_node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "classify")
    builder.add_edge("classify", "intent_shortcuts")
    builder.add_conditional_edges(
        "intent_shortcuts",
        route_decider,
        {
            "direct": "direct",
            "data": "data",
            "rag": "rag",
            "fallback": "fallback",
        },
    )
    builder.add_edge("direct", "memory")
    builder.add_edge("data", "memory")
    builder.add_edge("rag", "memory")
    builder.add_edge("fallback", "memory")
    builder.add_edge("memory", END)

    checkpointer = getattr(_MEMORY, "saver", None)
    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "AgentState",
    "build_graph",
    "route_decider",
    "_yield_openai_stream_chunks",
]
