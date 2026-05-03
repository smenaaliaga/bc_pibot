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


# ---------------------------------------------------------------------------
# Scope-block node: bloquea consultas fuera de PIB/IMACEC con guardrail fijo
# + sugerencia generada por LLM (variación semántica controlada).
# ---------------------------------------------------------------------------

OUT_OF_SCOPE_GUARDRAIL = "Esta IA responde solamente consultas del PIB e IMACEC."

OUT_OF_SCOPE_FALLBACK_SUGGESTION = (
    "¿Te gustaría consultar algún valor del PIB o el IMACEC? Por ejemplo, "
    "puedes preguntar por la variación mensual del IMACEC o la variación "
    "trimestral del PIB en un período específico."
)

_OUT_OF_SCOPE_SUGGESTION_SYSTEM_PROMPT = (
    "Eres el asistente PIBot del Banco Central de Chile.\n"
    "La pregunta del usuario está FUERA DE ALCANCE (no es PIB ni IMACEC).\n\n"
    "REGLAS ABSOLUTAS:\n"
    "1. NO respondas la pregunta del usuario bajo ninguna circunstancia.\n"
    "2. NO menciones tipo de cambio, dólar, paridades, autoridades del Banco "
    "Central, IPC, TPM, ni ningún tema distinto a PIB / IMACEC.\n"
    "3. Genera UN SOLO PÁRRAFO de máximo 2 oraciones, en español, que invite "
    "al usuario a consultar valores del PIB o el IMACEC.\n"
    "4. Puedes ejemplificar UNO o DOS de estos temas (sin inventar otros): "
    "variación mensual del IMACEC, variación trimestral del PIB, PIB "
    "desestacionalizado, contribución de actividades al PIB, IMACEC por "
    "sector (minería, servicios, comercio), PIB regional, PIB anual.\n"
    "5. NO uses listas, viñetas ni markdown. Solo texto plano.\n"
    "6. NO repitas la frase guardrail ni la cites textualmente.\n\n"
    "Devuelve solo el párrafo, sin saludo ni cierre."
)

# Tokens prohibidos en la sugerencia (defensa contra fugas del LLM).
_OUT_OF_SCOPE_BLOCKLIST_RE = re.compile(
    r"\b("
    r"d[oó]lar(?:es)?|"
    r"euro|"
    r"yuan|"
    r"yen|"
    r"tipo\s+de\s+cambio|"
    r"paridad(?:es)?|"
    r"tpm|"
    r"tasa\s+de\s+pol[ií]tica|"
    r"ipc|"
    r"inflaci[oó]n|"
    r"uf|"
    r"utm|"
    r"bitcoin|cripto|"
    r"presidenta?|"
    r"consejer[oa]s?|"
    r"divisi[oó]n(?:es)?|"
    r"hern[aá]n\s+fern[aá]ndez"
    r")\b",
    re.IGNORECASE,
)

_GUARDRAIL_ECHO_RE = re.compile(
    r"esta\s+ia\s+responde\s+solamente\s+consultas\s+del\s+pib\s+e\s+imacec\.?",
    re.IGNORECASE,
)


def _sanitize_scope_suggestion(text: str) -> str:
    """Sanea la sugerencia generada por el LLM. Retorna '' si debe descartarse."""
    if not text:
        return ""
    cleaned = _ensure_text(text).strip()
    if not cleaned:
        return ""
    # Elimina eco del guardrail.
    cleaned = _GUARDRAIL_ECHO_RE.sub("", cleaned).strip()
    # Colapsa saltos de línea.
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if not cleaned:
        return ""
    # Rechaza si menciona temas prohibidos.
    if _OUT_OF_SCOPE_BLOCKLIST_RE.search(cleaned):
        return ""
    # Debe mencionar PIB o IMACEC.
    if not re.search(r"\b(pib|imacec)\b", cleaned, re.IGNORECASE):
        return ""
    # Sin markdown / viñetas.
    if cleaned.startswith(("-", "*", "•", "#", "[")):
        return ""
    # Truncado defensivo.
    if len(cleaned) > 320:
        cut = cleaned[:320]
        idx = cut.rfind(".")
        cleaned = (cut[: idx + 1] if idx > 0 else cut).strip()
    return cleaned


# Importes diferidos para evitar ciclo y para entornos sin langchain.
try:
    from langchain.messages import SystemMessage as _ScopeSystemMessage, HumanMessage as _ScopeHumanMessage  # type: ignore
except Exception:  # pragma: no cover
    _ScopeSystemMessage = None  # type: ignore
    _ScopeHumanMessage = None  # type: ignore


def _generate_scope_suggestion(llm_adapter) -> str:
    """Genera la variante semántica usando el LLM. '' si falla o no disponible."""
    if llm_adapter is None:
        return ""
    chat = getattr(llm_adapter, "_chat", None)
    if chat is None or _ScopeSystemMessage is None or _ScopeHumanMessage is None:
        return ""
    try:
        msgs = [
            _ScopeSystemMessage(content=_OUT_OF_SCOPE_SUGGESTION_SYSTEM_PROMPT),
            _ScopeHumanMessage(content="Genera la sugerencia ahora."),
        ]
        # Una sola invocación (sin streaming) para latencia y simplicidad.
        out = chat.invoke(msgs)
        text = getattr(out, "content", None) or str(out)
        return _ensure_text(text)
    except Exception:
        logger.exception("[SCOPE_BLOCK] Variant generation failed; using fallback")
        return ""


def make_scope_block_node(llm_adapter=None):
    """Nodo determinístico para preguntas fuera de alcance (no PIB/IMACEC).

    Estructura de salida:
      <GUARDRAIL fijo>\\n\\n<sugerencia variable>
    Si el LLM no produce una sugerencia válida → texto fallback determinístico.
    """

    def scope_block_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        # 1. Emitir guardrail fijo.
        first_chunk = OUT_OF_SCOPE_GUARDRAIL + "\n\n"
        _emit_stream_chunk(first_chunk, writer)

        # 2. Generar variante o fallback.
        raw_suggestion = _generate_scope_suggestion(llm_adapter)
        suggestion = _sanitize_scope_suggestion(raw_suggestion)
        if not suggestion:
            suggestion = OUT_OF_SCOPE_FALLBACK_SUGGESTION

        _emit_stream_chunk(suggestion, writer)

        output = f"{OUT_OF_SCOPE_GUARDRAIL}\n\n{suggestion}"
        logger.info(
            "[SCOPE_BLOCK] Out-of-scope question handled | output_len=%d | variant_used=%s",
            len(output),
            bool(raw_suggestion and suggestion != OUT_OF_SCOPE_FALLBACK_SUGGESTION),
        )
        return {"output": output, "route_decision": "out_of_scope"}

    return scope_block_node


# ---------------------------------------------------------------------------
# Greeting node: responde a saludos puros con un mensaje amigable + invitación
# a consultar PIB/IMACEC. Determinístico (sin LLM).
# ---------------------------------------------------------------------------

GREETING_RESPONSE = (
    "¡Hola! Soy PIBot, el asistente del Banco Central de Chile para consultas "
    "sobre PIB e IMACEC. ¿En qué te puedo ayudar? Por ejemplo, puedes "
    "preguntarme por el valor del último IMACEC o la variación trimestral del PIB."
)


def make_greeting_node():
    """Nodo determinístico para saludos (\"hola\", \"buenos días\", etc.)."""

    def greeting_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        _emit_stream_chunk(GREETING_RESPONSE, writer)
        logger.info("[GREETING] Greeting response emitted | output_len=%d", len(GREETING_RESPONSE))
        return {"output": GREETING_RESPONSE, "route_decision": "greeting"}

    return greeting_node


__all__ = [
    "make_rag_node",
    "make_fallback_node",
    "make_scope_block_node",
    "make_greeting_node",
    "OUT_OF_SCOPE_GUARDRAIL",
    "OUT_OF_SCOPE_FALLBACK_SUGGESTION",
    "GREETING_RESPONSE",
]
