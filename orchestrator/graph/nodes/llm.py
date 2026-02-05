"""LLM-based nodes (RAG/fallback) for the PIBot graph."""

from __future__ import annotations

import os
import logging
from typing import List, Optional

from langgraph.types import StreamWriter

from ..state import (
    AgentState,
    _emit_stream_chunk,
    _ensure_text,
)

logger = logging.getLogger(__name__)


def _run_llm(
    state: AgentState,
    adapter,
    *,
    writer: Optional[StreamWriter] = None,
):
    question = state.get("question", "")
    history = state.get("conversation_history") or []
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


def make_rag_node(llm_adapter):
    def rag_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        intent_payload = state.get("intent")
        if isinstance(intent_payload, dict):
            intent = intent_payload.get("intent", "")
        else:
            intent = _ensure_text(intent_payload)
        question = state.get("question", "")
        history = state.get("conversation_history", [])
        session_id = state.get("session_id", "")

        logger.info(
            "[RAG_NODE] Iniciando | intent=%s | question=%s | history_len=%d | session=%s",
            intent,
            question[:100] if question else "(vacío)",
            len(history),
            session_id[:12] if session_id else "(vacío)",
        )

        result = _run_llm(state, llm_adapter, writer=writer)

        logger.info(
            "[RAG_NODE] Completado | output_len=%d",
            len(result.get("output", "")),
        )
        return result

    return rag_node


def make_fallback_node(llm_adapter):
    def fallback_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        return _run_llm(state, llm_adapter, writer=writer)

    return fallback_node


__all__ = ["make_rag_node", "make_fallback_node"]
