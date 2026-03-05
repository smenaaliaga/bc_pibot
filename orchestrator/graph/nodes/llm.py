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


def _is_generation_error_output(text: str) -> bool:
    normalized = _ensure_text(text).strip().lower()
    return normalized.startswith("(error generando)") or normalized.startswith("tuve un problema generando")


def _has_existing_methodology_footer(output: str, footer: str) -> bool:
    current = _ensure_text(output)
    built_footer = _ensure_text(footer)
    if not current or not built_footer:
        return False

    normalized_current = current.lower()
    if "para mayor información, puedes consultar los documentos disponibles en la web oficial del banco central de chile" in normalized_current:
        return True

    for line in built_footer.splitlines():
        candidate = _ensure_text(line).strip()
        if candidate.startswith("- [") and candidate in current:
            return True
    return False


def _build_methodology_footer(adapter, max_sources: int = 2) -> str:
    if adapter is None or not hasattr(adapter, "get_last_rag_sources"):
        return ""
    try:
        sources = adapter.get_last_rag_sources()  # type: ignore[attr-defined]
    except Exception:
        return ""
    if not isinstance(sources, list):
        return ""

    lines: List[str] = []
    seen: set[tuple[str, str]] = set()
    for item in sources:
        if not isinstance(item, dict):
            continue
        docname = _ensure_text(item.get("docname")).strip()
        link = _ensure_text(item.get("link")).strip()
        if not link:
            continue
        if not docname:
            docname = "documento"
        key = (docname.lower(), link.lower())
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{docname}]({link})")
        if len(lines) >= max_sources:
            break

    if not lines:
        return ""

    return (
        "\n\nPara mayor información, puedes consultar los documentos disponibles en la web oficial del Banco Central de Chile:\n"
        + "\n".join(lines)
    )


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
        footer = _build_methodology_footer(llm_adapter, max_sources=2)
        current_output = _ensure_text(result.get("output"))
        if footer and not _is_generation_error_output(current_output) and not _has_existing_methodology_footer(current_output, footer):
            result["output"] = f"{current_output}{footer}"
            _emit_stream_chunk(footer, writer)

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
