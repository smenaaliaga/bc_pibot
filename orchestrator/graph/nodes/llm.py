"""LLM-based nodes (RAG/fallback) for the PIBot graph."""

from __future__ import annotations

import os
import re
import logging
from typing import List, Optional

from langgraph.types import StreamWriter

from ..state import (
    AgentState,
    _emit_stream_chunk,
    _ensure_text,
)
from ...llm.llm_adapter import (
    CALENDAR_URL,
    CALENDAR_LINK_MD,
    CALENDAR_LINK_LABEL,
    _CALENDAR_KEYWORDS,
)

logger = logging.getLogger(__name__)

_OLD_CALENDAR_URL = "https://www.bcentral.cl/web/banco-central/areas/estadisticas/calendario-de-publicaciones"

# Regex to detect the raw CALENDAR_URL NOT already inside a markdown link
_RAW_URL_RE = re.compile(
    r"(?<!\]\()(?<!\()"           # not preceded by '](' or '('
    + re.escape(CALENDAR_URL)
    + r"(?!\))"                    # not followed by ')'
)


def _fix_calendar_url(text: str) -> str:
    """Replace hallucinated / raw calendar URLs with a clean markdown link."""
    # 1. Replace old hallucinated URL (inside markdown links or raw)
    if _OLD_CALENDAR_URL in text:
        text = text.replace(_OLD_CALENDAR_URL, CALENDAR_URL)

    # 2. Replace any standalone raw CALENDAR_URL with markdown link
    text = _RAW_URL_RE.sub(CALENDAR_LINK_MD, text)
    return text


# Regex for a markdown link whose text contains "calendario"
_CALENDAR_LINK_RE = re.compile(
    r"-?\s*\[([^\]]*calendario[^\]]*)\]\([^\)]+\)\s*\n?",
    re.IGNORECASE,
)

# Matches a sentence or list bullet in the body that references the calendar link.
_BODY_CALENDAR_SENTENCE_RE = re.compile(
    r"(?:(?<=[.!?\n])|^)[^.!?\n]*\[[^\]]*calendario[^\]]*\]\([^)]+\)[^.!?\n]*[.!?]?\s*",
    re.IGNORECASE,
)

_FOOTER_MARKER = "para mayor información"


def _strip_calendar_from_body(text: str) -> str:
    """Remove any sentence/bullet in the body that references the calendar link.

    The calendar reference is always surfaced via the methodology footer instead,
    so we avoid showing it twice.
    """
    marker_pos = text.lower().find(_FOOTER_MARKER)
    if marker_pos < 0:
        body, footer = text, ""
    else:
        body, footer = text[:marker_pos], text[marker_pos:]
    cleaned_body = _BODY_CALENDAR_SENTENCE_RE.sub("", body)
    cleaned_body = re.sub(r"\n{3,}", "\n\n", cleaned_body)
    return cleaned_body + footer


def _dedup_calendar_refs(text: str) -> str:
    """If the calendar link appears both in the body and the footer, remove it from the footer."""
    marker_pos = text.lower().find(_FOOTER_MARKER)
    if marker_pos < 0:
        return text
    body = text[:marker_pos]
    footer = text[marker_pos:]

    # Only deduplicate if body already contains a calendar link
    if not _CALENDAR_LINK_RE.search(body):
        return text

    # Remove calendar-link lines from footer only
    cleaned_footer = _CALENDAR_LINK_RE.sub("", footer)
    return body + cleaned_footer


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


def _build_methodology_footer(adapter, question: str = "", max_sources: int = 2) -> str:
    if adapter is None or not hasattr(adapter, "get_last_rag_sources"):
        sources: List = []
    else:
        try:
            sources = adapter.get_last_rag_sources()  # type: ignore[attr-defined]
        except Exception:
            sources = []
    if not isinstance(sources, list):
        sources = []

    # If the user asked about the calendar, the canonical calendar reference
    # must ALWAYS be the first entry in the footer. We strip any retrieved
    # duplicate (regardless of position) and prepend the canonical one, so it
    # is never truncated by `max_sources`.
    if question and _CALENDAR_KEYWORDS.search(question):
        sources = [
            s for s in sources
            if not (
                isinstance(s, dict)
                and CALENDAR_URL in _ensure_text(s.get("link"))
            )
        ]
        sources = [{"docname": CALENDAR_LINK_LABEL, "link": CALENDAR_URL}] + list(sources)

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
        current_output = _ensure_text(result.get("output"))
        current_output = _fix_calendar_url(current_output)
        current_output = _strip_calendar_from_body(current_output)
        current_output = _dedup_calendar_refs(current_output)
        result["output"] = current_output
        footer = _build_methodology_footer(llm_adapter, question=question, max_sources=2)
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
        result = _run_llm(state, llm_adapter, writer=writer)
        current_output = _ensure_text(result.get("output"))
        current_output = _strip_calendar_from_body(current_output)
        current_output = _fix_calendar_url(current_output)
        result["output"] = _dedup_calendar_refs(current_output)
        return result

    return fallback_node


__all__ = ["make_rag_node", "make_fallback_node"]
