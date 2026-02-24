"""LangGraph agent graph wiring for PIBot orchestration."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from orchestrator.classifier.classifier_agent import (
    build_intent_info,
    classify_question_with_history,
)
from orchestrator.classifier.intent_store import IntentStoreBase

from .backends import _safe_intent_store, _safe_llm, _safe_memory_adapter, _safe_retriever
from .nodes import (
    make_classify_node,
    make_data_node,
    make_fallback_node,
    make_ingest_node,
    make_intent_node,
    make_memory_node,
    make_rag_node,
    make_router_node,
)
from .state import (
    AgentState,
    _emit_stream_chunk,
    _yield_openai_stream_chunks,
)

logger = logging.getLogger(__name__)

_MEMORY: Optional[Any] = None
_RETRIEVER = None
_RAG_LLM = None
_FALLBACK_LLM = None
_INTENT_STORE: Optional[IntentStoreBase] = None


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


def ingest_node(state: AgentState) -> AgentState:
    node_fn = make_ingest_node(_MEMORY)
    return node_fn(state)


def classify_node(state: AgentState) -> AgentState:
    node_fn = make_classify_node(
        _INTENT_STORE,
        classify_question_with_history,
        build_intent_info,
    )
    return node_fn(state)


def intent_node(state: AgentState) -> AgentState:
    node_fn = make_intent_node(_MEMORY)
    return node_fn(state)


def router_node(state: AgentState) -> AgentState:
    node_fn = make_router_node()
    return node_fn(state)


def data_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    node_fn = make_data_node(_MEMORY)
    return node_fn(state, writer=writer)


def rag_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    node_fn = make_rag_node(_RAG_LLM)
    return node_fn(state, writer=writer)


def fallback_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
    node_fn = make_fallback_node(_FALLBACK_LLM)
    return node_fn(state, writer=writer)


def memory_node(state: AgentState) -> AgentState:
    node_fn = make_memory_node(_MEMORY, _INTENT_STORE)
    return node_fn(state)


def _route_from_router(state: AgentState) -> str:
    decision = str(state.get("route_decision") or "fallback").strip().lower()
    if decision not in {"data", "rag", "fallback"}:
        return "fallback"
    return decision


def build_graph():
    _ensure_backends()
    builder = StateGraph(AgentState)

    ingest = make_ingest_node(_MEMORY)
    classify = make_classify_node(
        _INTENT_STORE,
        classify_question_with_history,
        build_intent_info,
    )
    intent = make_intent_node(_MEMORY)
    router = make_router_node()
    data = make_data_node(_MEMORY)
    rag = make_rag_node(_RAG_LLM)
    fallback = make_fallback_node(_FALLBACK_LLM)
    memory = make_memory_node(_MEMORY, _INTENT_STORE)

    builder.add_node("ingest", ingest)
    builder.add_node("classify", classify)
    builder.add_node("intent", intent)
    builder.add_node("router", router)
    builder.add_node("data", data)
    builder.add_node("rag", rag)
    builder.add_node("fallback", fallback)
    builder.add_node("memory", memory)

    # --- DEBUG: forzar ruta directa a DATA ---
    # (Descomentar el bloque original para restaurar el enrutamiento correcto)
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "classify")
    # --- DEBUG: forzar ruta directa a DATA desde classify ---
    # (Descomentar el bloque original para restaurar el enrutamiento correcto)
    builder.add_edge("classify", "data")
    # builder.add_edge("classify", "intent")
    # builder.add_edge("intent", "router")
    # builder.add_conditional_edges(
    #     "router",
    #     _route_from_router,
    #     {"data": "data", "rag": "rag", "fallback": "fallback"},
    # )
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
    "_emit_stream_chunk",
]
