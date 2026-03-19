"""Response state executor and payload-to-LLM bridge.

This module centralizes the `response` graph state logic:
- Consume `response_payload` produced by upstream nodes.
- Execute data/rag/fallback generation.
- Apply response contract + acceptance checks.

Important: deterministic post-processing does not replace LLM generation.
The LLM still provides the base answer; post-processing primarily reduces the
probability of erroneous output and favors wording improvements over undesired
interpretation or unsolicited text generation.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.data.response import stream_data_response

from .state import AgentState, _emit_stream_chunk
from .nodes.llm import (
    _build_methodology_footer,
    _has_existing_methodology_footer,
    _is_generation_error_output,
    run_llm_stream,
)


_GENERIC_INTRO_TOKENS = (
    "se reporta el resultado solicitado",
    "se reporta reporta el resultado solicitado",
)

CONTRACT_VERSION = "response-contract-v1"
DEFAULT_RESPONSE_SECTIONS: List[str] = [
    "introduccion",
    "respuesta",
    "sugerencias",
    "csv",
]

_VARIATION_KEYWORDS = (
    "cuanto crecio",
    "cuanto creció",
    "cuanto cayo",
    "cuanto cayó",
    "cuanto vario",
    "cuanto varió",
    "variacion",
    "variación",
    "variacion interanual",
    "variación interanual",
    "interanual",
)

_ECONOMY_KEYWORDS = (
    "economia",
    "economía",
    "imacec",
    "actividad economica",
    "actividad económica",
)

_CSV_BLOCK_RE = re.compile(
    r"##CSV_DOWNLOAD_START\n.*?##CSV_DOWNLOAD_END",
    re.DOTALL,
)

_FOLLOWUP_BLOCK_RE = re.compile(
    r"##FOLLOWUP_START\n.*?##FOLLOWUP_END",
    re.DOTALL,
)

_BASE_INTRO_TEXT = "Con base en los datos disponibles."

_FORBIDDEN_INTRO_TAIL_TOKENS = (
    "se reporta el resultado solicitado",
    "se reporta reporta el resultado solicitado",
)

_UNPUBLISHED_TOKENS = (
    "aun no han sido publicados",
    "aun no ha sido publicado",
    "aun no se han publicado",
    "aun no esta publicado",
    "aun no esta disponible",
)

_PREV_PERIOD_TOKENS = (
    "periodo anterior",
    "período anterior",
    "mes anterior",
    "trimestre anterior",
    "en el margen",
    "respecto al periodo anterior",
    "respecto al período anterior",
)

_RANGE_QUERY_TOKENS = (
    "entre",
    "desde",
    "hasta",
    "rango",
    "serie",
    "valores",
)

_ORIGINAL_VALUE_TOKENS = (
    # --- tokens genéricos de valor ---
    "dolar",
    "dólar",
    "dolares",
    "dólares",
    "per capita",
    "pesos",
    "precios corrientes",
    "nominal",
    "cifra original",
    "valor original",
    # --- PIB ---
    "cuanto es el pib",
    "a cuanto asciende el pib",
    "valor del pib",
    "pib de chile",
    "pib en pesos",
    "pib en dolares",
    "pib per capita",
)

_PIB_ORIGINAL_VALUE_HINTS = (
    "precios corrientes",
    "per capita",
    "per cápita",
    "dolares",
    "dólares",
    "usd",
    "en pesos",
)

_VALUE_QUERY_HINTS = (
    "cuanto",
    "valor",
    "asciende",
    "monto",
    "nivel",
)

_CONTRIBUTION_TOKENS = (
    "impulso",
    "contribucion",
    "contribución",
    "aporte",
)

_INTERANUAL_TOKENS = (
    "variacion anual",
    "variación anual",
    "mismo periodo del ano anterior",
    "mismo periodo del año anterior",
    "mismo período del año anterior",
    "mismo mes del ano anterior",
    "mismo mes del año anterior",
    "mismo trimestre del ano anterior",
    "mismo trimestre del año anterior",
)

_ECONOMY_TOKENS = (
    "economia",
    "economía",
    "actividad economica",
    "actividad económica",
    "pib",
)

_VALUE_UNIT_TOKENS = (
    "pesos",
    "dolares",
    "dólares",
    "usd",
    "per capita",
    "per cápita",
    "precios corrientes",
    "miles de millones",
    "promedio 2018",
    "base 2018",
    "base 2018=100",
    "indice",
    "índice",
)

_SUBJECTIVE_TOKENS = (
    "desafio",
    "desafío",
    "desafios",
    "desafíos",
    "fortaleza",
    "debilidad",
    "preocupante",
    "alarmante",
    "optimista",
    "pesimista",
)

_MONTH_TOKENS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "setiembre",
    "octubre",
    "noviembre",
    "diciembre",
)

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(r"\b(?:19|20)\d{2}-Q[1-4]\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"-?\d+(?:[\.,]\d+)?%")
_PERCENT_VALUE_RE = re.compile(r"(-?\d+(?:[\.,]\d+)?)%")
_PERCENT_TOKEN_RE = re.compile(r"\*\*-?\d+(?:[\.,]\d+)?%\*\*|-?\d+(?:[\.,]\d+)?%")
_NUMBER_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
_CAYO_RE = re.compile(r"\bcay[oó]\b", re.IGNORECASE)
_DOUBLE_REPORTA_RE = re.compile(r"\breporta\s+reporta\b", re.IGNORECASE)
_INTERANUAL_FUE_RE = re.compile(r"variaci[oó]n\s+interanual\s+fue", re.IGNORECASE)
_INTERANUAL_VARIATION_RE = re.compile(r"variaci[oó]n\s+interanual", re.IGNORECASE)
_INTERANUAL_WORD_RE = re.compile(r"\binteranual(?:es)?\b", re.IGNORECASE)
_NEGATIVE_GROWTH_PHRASE_RE = re.compile(
    r"\b(?:creci[oó]|aument[oó]|subi[oó])\s+un\s+(\*\*)?-(\d+(?:[\.,]\d+)?%)(?:\*\*)?",
    re.IGNORECASE,
)
_NEGATIVE_DECLINE_PHRASE_RE = re.compile(
    r"\b(?:disminuy[oó]|baj[oó]|retrocedi[oó]|cay[oó])\s+un\s+(\*\*)?-(\d+(?:[\.,]\d+)?%)(?:\*\*)?",
    re.IGNORECASE,
)
_CONTRIBUTION_VALUE_RE = re.compile(
    r"(contribuci[oó]n[^\n]{0,80}?)(\*\*)?(-?\d+(?:[\.,]\d+)?)(\*\*)?(\s*(?:puntos?\s+porcentuales|pp))",
    re.IGNORECASE,
)
_GENERIC_PP_VALUE_RE = re.compile(
    r"(\*\*)?(-?\d+(?:[\.,]\d+)?)(\*\*)?(\s*(?:pp|puntos?\s+porcentuales))",
    re.IGNORECASE,
)
_CONTRIBUTION_BARE_BOLD_VALUE_RE = re.compile(
    r"((?:contribuy[oó]|aport[oó]|contribuci[oó]n)[^.\n]{0,80}?(?:con|de)\s+\*\*)"
    r"(-?\d+(?:[\.,]\d{2,}))(\*\*)(?!\s*%)",
    re.IGNORECASE,
)
_SUGGESTION_LINE_RE = re.compile(r"^\s*suggestion_\d+\s*=\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_RECOMMENDATION_TOKENS = (
    "como recomendacion final",
    "como recomendación final",
    "te recomendamos",
    "te sugerimos",
    "puedes consultar",
    "puedes profundizar",
    "puedes revisar",
    "como siguiente paso",
)
_INTRO_SECTION_RE = re.compile(
    r"Introducci[oó]n:\s*(.*?)\n\s*\nRespuesta:",
    re.IGNORECASE | re.DOTALL,
)
_RESPONSE_SECTION_RE = re.compile(
    r"Respuesta:\s*(.*?)(?:\n\s*\nSugerencias:|\n\s*\nCSV:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?", re.DOTALL)
_NUMERIC_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
# Valores económicos visibles: negrita (**111,57** o **-0,6%**) o porcentaje inline
_ECONOMIC_VALUE_RE = re.compile(
    r"\*\*-?\d+(?:[\.,]\d+)?%?\*\*"   # bold value o bold percent
    r"|-?\d+(?:[\.,]\d+)?%",            # inline percent
)
# Para normalizar decimales de porcentajes a 1 decimal
_BOLD_PERCENT_NORMALIZE_RE = re.compile(
    r"(\*\*)?(-?\d+(?:[\.,]\d+)?)(%)(?:\*\*)?",
)

_MONTH_TO_NUM = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

_NUM_TO_MONTH = {
    1: "enero",
    2: "febrero",
    3: "marzo",
    4: "abril",
    5: "mayo",
    6: "junio",
    7: "julio",
    8: "agosto",
    9: "septiembre",
    10: "octubre",
    11: "noviembre",
    12: "diciembre",
}

_QUARTER_TO_LABEL = {
    1: "1er trimestre",
    2: "2do trimestre",
    3: "3er trimestre",
    4: "4to trimestre",
}

_DEFAULT_TRACE_FILE = "run_detail.log"


def _primary_entity(state: AgentState) -> Dict[str, Any]:
    entities = state.get("entities")
    if isinstance(entities, list) and entities:
        first = entities[0]
        if isinstance(first, dict):
            return dict(first)
    return {}


def _norm_text(value: Any) -> str:
    raw = str(value or "").strip().lower()
    normalized = unicodedata.normalize("NFKD", raw)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_any(text: str, tokens: Tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _split_sentences(text: str) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    sentinel = "__NUM_DOT__"
    protected = _NUMERIC_DOT_RE.sub(sentinel, raw)
    sentences: List[str] = []
    for sentence in _SENTENCE_RE.findall(protected):
        cleaned = str(sentence or "").strip()
        if not cleaned:
            continue
        sentences.append(cleaned.replace(sentinel, "."))
    return sentences


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def resolve_response_temperature(
    value: Any,
    *,
    fallback: float = 0.2,
) -> float:
    """Resolve and clamp response temperature to [0.0, 1.0]."""
    resolved = _safe_float(value, fallback)
    if resolved < 0.0:
        return 0.0
    if resolved > 1.0:
        return 1.0
    return resolved


def _is_variation_question(normalized_question: str) -> bool:
    return any(token in normalized_question for token in _VARIATION_KEYWORDS)


def _is_economy_question(normalized_question: str) -> bool:
    return any(token in normalized_question for token in _ECONOMY_KEYWORDS)


def _extract_first_percentage(text: str) -> Optional[str]:
    match = _PERCENT_RE.search(str(text or ""))
    if not match:
        return None
    return match.group(0)


def _split_marker_block(text: str, pattern: re.Pattern[str]) -> Tuple[str, str]:
    raw_text = str(text or "")
    match = pattern.search(raw_text)
    if not match:
        return raw_text.strip(), ""
    before = (raw_text[: match.start()] + raw_text[match.end() :]).strip()
    block = match.group(0).strip()
    return before, block


def _extract_unavailable_intro(text: str) -> Optional[str]:
    """Extract first sentence when it states period is not published yet."""
    body = str(text or "").strip()
    if not body:
        return None

    first_sentence = body.split(".", 1)[0].strip()
    if not first_sentence:
        return None

    normalized = _norm_text(first_sentence)
    unavailable_markers = (
        "aun no han sido publicados",
        "aun no ha sido publicado",
        "aun no se han publicado",
        "aun no esta publicado",
        "aun no esta disponible",
    )
    if any(marker in normalized for marker in unavailable_markers):
        if first_sentence.endswith("."):
            return first_sentence
        return f"{first_sentence}."
    return None


def build_response_contract(
    question: str,
    *,
    route_decision: str,
    entities: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a transparent response contract payload from request context."""
    normalized_question = _norm_text(question)
    is_variation = _is_variation_question(normalized_question)
    is_economy = _is_economy_question(normalized_question)

    preferred_opening = "Con base en los datos disponibles, se reporta el resultado solicitado."
    if is_variation and is_economy:
        preferred_opening = "La economía varió un {variation} interanual."
    elif is_variation:
        preferred_opening = "La variable consultada varió un {variation} interanual."

    fallback_temperature = _safe_float(os.getenv("DATA_RESPONSE_TEMPERATURE", "0.2"), 0.2)
    resolved_temperature = resolve_response_temperature(
        temperature if temperature is not None else fallback_temperature,
        fallback=fallback_temperature,
    )

    return {
        "version": CONTRACT_VERSION,
        "route_decision": str(route_decision or "fallback").strip().lower(),
        "sections": list(DEFAULT_RESPONSE_SECTIONS),
        "semantic": {
            "is_variation_question": is_variation,
            "is_economy_question": is_economy,
            "require_single_metric": is_variation,
            "require_economy_variation_phrase": is_variation and is_economy,
        },
        "style": {
            "preferred_opening": preferred_opening,
            "allow_lexical_variation": True,
            "decimal_places_percent": 1,
        },
        "entities": dict(entities or {}),
        "temperature": resolved_temperature,
    }


def _build_response_sections(
    output: str,
    contract: Dict[str, Any],
) -> Dict[str, str]:
    without_csv, csv_block = _split_marker_block(output, _CSV_BLOCK_RE)
    without_followup, followup_block = _split_marker_block(without_csv, _FOLLOWUP_BLOCK_RE)

    style = contract.get("style") or {}
    semantic = contract.get("semantic") or {}
    preferred_opening = str(style.get("preferred_opening") or "").strip()

    response_body = without_followup.strip()
    variation = _extract_first_percentage(response_body) or "0,0%"
    unavailable_intro = _extract_unavailable_intro(response_body)
    if unavailable_intro:
        intro = unavailable_intro
    else:
        intro = preferred_opening.replace("{variation}", variation)

    if not intro:
        intro = "Con base en los datos disponibles, se reporta el resultado solicitado."

    if bool(semantic.get("require_economy_variation_phrase")):
        normalized_intro = _norm_text(intro)
        if "la economia vario un" not in normalized_intro:
            intro = f"La economía varió un {variation} interanual."

    body_clean = response_body.strip()
    if body_clean and _norm_text(body_clean).startswith(_norm_text(intro)):
        tail = body_clean[len(intro) :].lstrip("\n \t.,:;-")
        response_body = tail

    return {
        "introduccion": intro.strip(),
        "respuesta": response_body,
        "sugerencias": followup_block.strip(),
        "csv": csv_block.strip(),
    }


def compose_modular_response(
    output: str,
    contract: Dict[str, Any],
    *,
    question: str,
) -> Tuple[str, Dict[str, str]]:
    """Compose a natural response while preserving marker blocks."""
    _ = question  # signature compatibility
    sections = _build_response_sections(output, contract)
    return _compose_from_sections(sections), sections


def evaluate_response_contract(output: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Validate response against a lightweight contract and return diagnostics."""
    errors: List[str] = []
    normalized_output = _norm_text(output)
    semantic = contract.get("semantic") or {}
    intro_text = _extract_intro_text(output)
    response_text = _extract_response_text(output)

    if not str(intro_text or "").strip():
        errors.append("missing_introduccion_section")
    if not str(response_text or "").strip():
        errors.append("missing_respuesta_section")

    if bool(semantic.get("require_economy_variation_phrase")):
        has_economy_variation = (
            "la economia vario un" in normalized_output
            or "la variacion anual fue" in normalized_output
            or "la variacion fue de" in normalized_output
        )
        if not has_economy_variation:
            errors.append("missing_economy_variation_phrase")

    percentages = _PERCENT_RE.findall(str(output or ""))
    if bool(semantic.get("require_single_metric")) and len(percentages) != 1:
        errors.append("unexpected_percentage_count")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "metrics": {
            "percentage_count": len(percentages),
        },
    }


def _has_period_anchor(sentence: str) -> bool:
    s_norm = _norm_text(sentence)
    if _YEAR_RE.search(sentence) or _QUARTER_RE.search(sentence):
        return True
    if _contains_any(s_norm, _MONTH_TOKENS):
        return True
    return (
        "trimestre" in s_norm
        or "mes" in s_norm
        or "ano" in s_norm
        or "año" in str(sentence or "").lower()
    )


def _is_generic_intro(intro: str) -> bool:
    intro_norm = _norm_text(intro)
    if not intro_norm:
        return True
    return _contains_any(intro_norm, _GENERIC_INTRO_TOKENS)


def _enforce_intro_phrase(intro: str, *, mode: str) -> str:
    text = str(intro or "").strip()
    if not text:
        return _BASE_INTRO_TEXT if mode == "data" else text

    intro_norm = _norm_text(text)
    if _contains_any(intro_norm, _FORBIDDEN_INTRO_TAIL_TOKENS):
        return _BASE_INTRO_TEXT

    if intro_norm == "con base en los datos disponibles":
        return _BASE_INTRO_TEXT
    return text


def _sanitize_wording(text: str) -> str:
    sanitized = str(text or "")
    sanitized = _normalize_signed_variation_wording(sanitized)
    sanitized = _CAYO_RE.sub("varió", sanitized)
    sanitized = _DOUBLE_REPORTA_RE.sub("reporta", sanitized)
    sanitized = _INTERANUAL_FUE_RE.sub("variación anual fue", sanitized)
    sanitized = _INTERANUAL_VARIATION_RE.sub("variación anual", sanitized)
    sanitized = _INTERANUAL_WORD_RE.sub("anual", sanitized)
    return sanitized


def _normalize_signed_variation_wording(text: str) -> str:
    raw = str(text or "")

    def _to_neutral(match: re.Match[str]) -> str:
        open_bold = str(match.group(1) or "")
        pct = str(match.group(2) or "")
        close_bold = "**" if open_bold else ""
        return f"la variación fue de {open_bold}-{pct}{close_bold}"

    raw = _NEGATIVE_GROWTH_PHRASE_RE.sub(_to_neutral, raw)
    raw = _NEGATIVE_DECLINE_PHRASE_RE.sub(_to_neutral, raw)
    return raw


def _extract_source_url(generation_logic: Dict[str, Any]) -> str:
    if not isinstance(generation_logic, dict):
        return ""
    inp = generation_logic.get("input")
    if not isinstance(inp, dict):
        return ""
    nested = inp.get("input")
    if not isinstance(nested, dict):
        return ""
    observations = nested.get("observations")
    if not isinstance(observations, dict):
        return ""
    return str(observations.get("source_url") or "").strip()


def _extract_generation_input(generation_logic: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(generation_logic, dict):
        return {}
    inp = generation_logic.get("input")
    if not isinstance(inp, dict):
        return {}
    nested = inp.get("input")
    if not isinstance(nested, dict):
        return {}
    return nested


def _extract_generation_entities(generation_logic: Dict[str, Any]) -> Dict[str, Any]:
    nested = _extract_generation_input(generation_logic)
    entities = nested.get("entities")
    if not isinstance(entities, dict):
        return {}
    return entities


def _extract_generation_observations(generation_logic: Dict[str, Any]) -> Dict[str, Any]:
    nested = _extract_generation_input(generation_logic)
    observations = nested.get("observations")
    if not isinstance(observations, dict):
        return {}
    return observations


def _extract_generation_question(generation_logic: Dict[str, Any]) -> str:
    nested = _extract_generation_input(generation_logic)
    return str(nested.get("question") or "").strip()


def _has_bde_reference(text: str) -> bool:
    out_norm = _norm_text(text)
    return (
        "fuente" in out_norm
        and (
            "bde" in out_norm
            or "base de datos estad" in out_norm
            or "si3.bcentral.cl" in out_norm
        )
    )


def _ensure_bde_source(text: str, generation_logic: Dict[str, Any]) -> str:
    if _has_bde_reference(text):
        return text

    source_url = _extract_source_url(generation_logic)
    if not source_url:
        return text

    suffix = (
        "\n\n**Fuente:** 🔗 [Base de Datos Estadísticos (BDE)]"
        f"({source_url}) del Banco Central de Chile."
    )
    return f"{str(text or '').strip()}{suffix}"


def _derive_contextual_intro(
    response_text: str,
    *,
    prefer_percent_only: bool = False,
    prefer_value_only: bool = False,
) -> str:
    sentences = _split_sentences(response_text)
    if not sentences:
        return ""

    for sentence in sentences:
        if _contains_any(_norm_text(sentence), _UNPUBLISHED_TOKENS):
            return sentence

    for sentence in sentences:
        has_period = _has_period_anchor(sentence)
        has_percent = bool(_PERCENT_RE.search(sentence))
        has_number = bool(_NUMBER_RE.search(sentence))
        if prefer_value_only:
            if has_period and has_number and not has_percent:
                return sentence
            continue
        if prefer_percent_only:
            if has_period and has_percent:
                return sentence
        else:
            if has_period and (has_percent or has_number):
                return sentence

    # Fallback: primera oración con contexto temporal
    for sentence in sentences:
        if _has_period_anchor(sentence):
            return sentence

    return sentences[0]


def _extract_unpublished_sentence(response_text: str) -> str:
    for sentence in _split_sentences(response_text):
        if _contains_any(_norm_text(sentence), _UNPUBLISHED_TOKENS):
            return sentence
    return ""


def _split_narrative_and_source(response_text: str) -> Tuple[str, str]:
    paragraphs = [p.strip() for p in str(response_text or "").split("\n\n") if p.strip()]
    narrative_parts: List[str] = []
    source_parts: List[str] = []
    for paragraph in paragraphs:
        p_norm = _norm_text(paragraph)
        if "fuente" in p_norm and (
            "bde" in p_norm
            or "base de datos estad" in p_norm
            or "si3.bcentral.cl" in p_norm
        ):
            source_parts.append(paragraph)
            continue
        narrative_parts.append(paragraph)
    return "\n\n".join(narrative_parts).strip(), "\n\n".join(source_parts).strip()


def _extract_followup_suggestions(suggestions_block: str) -> List[str]:
    suggestions: List[str] = []
    for match in _SUGGESTION_LINE_RE.finditer(str(suggestions_block or "")):
        candidate = str(match.group(1) or "").strip()
        if candidate:
            suggestions.append(candidate)
    return suggestions


def _is_recommendation_paragraph(paragraph: str) -> bool:
    p_norm = _norm_text(paragraph)
    if not p_norm:
        return False
    return _contains_any(p_norm, _RECOMMENDATION_TOKENS)


def _suggestion_to_recommendation(suggestion: str) -> str:
    clean = str(suggestion or "").strip()
    if not clean:
        return ""

    clean = clean.strip("¿?").strip()
    if not clean:
        return ""

    lower = _norm_text(clean)
    if lower.startswith(("cual", "cuál", "que", "qué", "como", "cómo")):
        return f"Puedes profundizar revisando: {clean}."
    if clean.endswith((".", "!")):
        return clean
    return f"{clean}."


def _build_recommendation_paragraph(
    suggestions_block: str,
    *,
    question: str = "",
    observations: Optional[Dict[str, Any]] = None,
) -> str:
    suggestions = _extract_followup_suggestions(suggestions_block)
    if suggestions:
        candidate = _suggestion_to_recommendation(suggestions[0])
        if candidate:
            return candidate

    q_norm = _norm_text(question)
    obs = observations if isinstance(observations, dict) else {}
    classification = obs.get("classification") if isinstance(obs.get("classification"), dict) else {}
    indicator = _norm_text(classification.get("indicator") or "")

    if _contains_any(q_norm, _CONTRIBUTION_TOKENS):
        return "Puedes profundizar revisando el detalle de contribuciones por actividad en el mismo período."
    if "region" in q_norm or "región" in str(question or "").lower():
        return "Puedes profundizar comparando la región consultada con el promedio nacional del mismo período."
    if indicator == "imacec" or "imacec" in q_norm:
        return "Puedes profundizar revisando el desglose sectorial del IMACEC y su trayectoria reciente."
    if indicator == "pib" or "pib" in q_norm or "economia" in q_norm or "economía" in str(question or "").lower():
        return "Puedes profundizar comparando la trayectoria reciente de la actividad económica y sus componentes."

    return "Puedes profundizar con una comparación del mismo indicador en períodos anteriores."


def _ensure_recommendation_paragraph(
    response_text: str,
    suggestions_block: str,
    *,
    question: str = "",
    observations: Optional[Dict[str, Any]] = None,
) -> str:
    narrative, source = _split_narrative_and_source(response_text)
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]

    if paragraphs and _is_recommendation_paragraph(paragraphs[-1]):
        narrative_out = "\n\n".join(paragraphs).strip()
    else:
        recommendation = _build_recommendation_paragraph(
            suggestions_block,
            question=question,
            observations=observations,
        )
        if recommendation:
            paragraphs.append(recommendation)
        narrative_out = "\n\n".join(paragraphs).strip()

    parts: List[str] = []
    if narrative_out:
        parts.append(narrative_out)
    if source:
        parts.append(source)
    return "\n\n".join(parts).strip()


def _to_float(value: str) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _format_decimal(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _count_economic_values(text: str) -> int:
    """Cuenta valores económicos visibles (bold o porcentajes) en el texto."""
    return len(_ECONOMIC_VALUE_RE.findall(str(text or "")))


def _normalize_percentage_decimals(text: str, decimals: int = 1) -> str:
    """Redondea todos los porcentajes del texto a *decimals* posiciones."""
    def _replace(m: re.Match[str]) -> str:
        open_bold = m.group(1) or ""
        raw = m.group(2)
        pct = m.group(3)
        value = _to_float(raw)
        if value is None:
            return m.group(0)
        rounded = _format_decimal(value, decimals)
        close_bold = "**" if open_bold else ""
        return f"{open_bold}{rounded}{pct}{close_bold}"
    return _BOLD_PERCENT_NORMALIZE_RE.sub(_replace, str(text or ""))


def _normalize_contribution_decimals(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        head = str(match.group(1) or "")
        open_bold = str(match.group(2) or "")
        number = str(match.group(3) or "")
        close_bold = str(match.group(4) or "")
        tail = str(match.group(5) or "")

        value = _to_float(number)
        if value is None:
            return match.group(0)

        rounded = _format_decimal(value, decimals=1)
        if bool(open_bold) != bool(close_bold):
            close_bold = open_bold
        return f"{head}{open_bold}{rounded}{close_bold}{tail}"

    normalized = _CONTRIBUTION_VALUE_RE.sub(_replace, str(text or ""))

    def _replace_bare_contribution(match: re.Match[str]) -> str:
        head = str(match.group(1) or "")
        number = str(match.group(2) or "")
        tail = str(match.group(3) or "")

        value = _to_float(number)
        if value is None:
            return match.group(0)

        rounded = _format_decimal(value, decimals=1)
        return f"{head}{rounded}{tail}"

    normalized = _CONTRIBUTION_BARE_BOLD_VALUE_RE.sub(_replace_bare_contribution, normalized)

    def _replace_pp(match: re.Match[str]) -> str:
        open_bold = str(match.group(1) or "")
        number = str(match.group(2) or "")
        close_bold = str(match.group(3) or "")
        tail = str(match.group(4) or "")

        value = _to_float(number)
        if value is None:
            return match.group(0)

        rounded = _format_decimal(value, decimals=1)
        if bool(open_bold) != bool(close_bold):
            close_bold = open_bold
        return f"{open_bold}{rounded}{close_bold}{tail}"

    return _GENERIC_PP_VALUE_RE.sub(_replace_pp, normalized)


def _is_pib_context(
    question_norm: str,
    entities: Dict[str, Any],
    observations: Dict[str, Any],
) -> bool:
    ent_indicator = _norm_text(
        entities.get("indicator_ent")
        or entities.get("indicator")
        or ""
    )
    classification = (
        observations.get("classification")
        if isinstance(observations.get("classification"), dict)
        else {}
    )
    obs_indicator = _norm_text(classification.get("indicator") or "")

    if ent_indicator == "pib" or obs_indicator == "pib":
        return True

    return "pib" in question_norm or "producto interno bruto" in question_norm


def _requires_original_value_response(
    question: str,
    entities: Dict[str, Any],
    generation_logic: Dict[str, Any],
) -> bool:
    q_norm = _norm_text(question)
    observations = _extract_generation_observations(generation_logic)
    if not _is_pib_context(q_norm, entities, observations):
        return False

    if _contains_any(q_norm, _ORIGINAL_VALUE_TOKENS):
        return True

    if _contains_any(q_norm, _PIB_ORIGINAL_VALUE_HINTS):
        return True

    calc_mode = str(entities.get("calc_mode_cls") or entities.get("calc_mode") or "").strip().lower()
    if calc_mode in {"original_value", "nominal"}:
        return True

    classification = observations.get("classification") if isinstance(observations.get("classification"), dict) else {}
    price = _norm_text(classification.get("price") or "")

    if price == "co" and calc_mode == "value" and _contains_any(q_norm, _VALUE_QUERY_HINTS):
        return True
    return False


def _enforce_original_value_narrative(response_text: str) -> str:
    narrative, source = _split_narrative_and_source(response_text)
    sentences = _split_sentences(narrative)
    if not sentences:
        return response_text

    def _strip_variation_clause(sentence: str) -> str:
        cleaned = str(sentence or "")
        cleaned = re.sub(
            r",?\s*(?:con|y\s+con)?\s*una?\s*variaci[oó]n[^.?!\n]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r",?\s*la\s*variaci[oó]n[^.?!\n]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"-?\d+(?:[\.,]\d+)?%", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:-")
        if cleaned and cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        return cleaned

    kept: List[str] = []
    for sentence in sentences:
        s_norm = _norm_text(sentence)
        candidate = sentence
        if "%" in sentence or "variacion anual" in s_norm or "variacion interanual" in s_norm:
            candidate = _strip_variation_clause(sentence)

        candidate_norm = _norm_text(candidate)
        if not candidate_norm:
            continue
        if not re.search(r"\d", candidate):
            continue
        if "variacion anual" in candidate_norm or "variacion interanual" in candidate_norm:
            continue
        if "%" in candidate:
            continue
        kept.append(candidate)

    rebuilt = " ".join(item.strip() for item in kept if item.strip()).strip()
    parts: List[str] = []
    if rebuilt:
        parts.append(rebuilt)
    if source:
        parts.append(source)
    return "\n\n".join(parts).strip()


def _is_default_yoy_by_question(question: str, entities: Dict[str, Any], mode: str) -> bool:
    if str(mode or "").strip().lower() != "data":
        return False

    q_norm = _norm_text(question)
    if _is_range_or_multi_value_question(question, q_norm):
        return False

    calc_mode = str(entities.get("calc_mode_cls") or entities.get("calc_mode") or "").strip().lower()

    if calc_mode == "contribution" or _contains_any(q_norm, _CONTRIBUTION_TOKENS):
        return False
    if _contains_any(q_norm, _PREV_PERIOD_TOKENS):
        return False
    if _contains_any(q_norm, _ORIGINAL_VALUE_TOKENS):
        return False
    return True


def _is_range_or_multi_value_question(question: str, normalized_question: str) -> bool:
    q_norm = str(normalized_question or "").strip()
    if _contains_any(q_norm, _RANGE_QUERY_TOKENS):
        return True

    years = _YEAR_RE.findall(str(question or ""))
    if len(set(years)) >= 2:
        return True

    return False


def _enforce_single_yoy_narrative(intro: str, response_text: str) -> str:
    narrative, source = _split_narrative_and_source(response_text)
    intro_has_pct = bool(_PERCENT_RE.search(intro))
    intro_is_unpublished = _contains_any(_norm_text(intro), _UNPUBLISHED_TOKENS)

    kept: List[str] = []
    already_has_metric = intro_has_pct
    for sentence in _split_sentences(narrative):
        s_norm = _norm_text(sentence)
        if not s_norm:
            continue

        if "ultimo dato disponible" in s_norm or "último dato disponible" in str(sentence or "").lower():
            kept.append(sentence)
            continue

        if _contains_any(s_norm, _UNPUBLISHED_TOKENS):
            if not intro_is_unpublished:
                kept.append(sentence)
            continue

        has_metric = bool(
            _PERCENT_RE.search(sentence)
            or "interanual" in s_norm
            or "variacion anual" in s_norm
        )
        if has_metric:
            pct_values = _PERCENT_RE.findall(sentence)
            has_prev_period_wording = _contains_any(s_norm, _PREV_PERIOD_TOKENS)
            if has_prev_period_wording or len(pct_values) > 1:
                metric_token = pct_values[0] if pct_values else (_extract_first_percentage(sentence) or "")
                if metric_token:
                    sentence = (
                        f"La variación anual fue {metric_token} respecto al mismo período del año anterior."
                    )
                    s_norm = _norm_text(sentence)

            if already_has_metric:
                continue
            kept.append(sentence)
            already_has_metric = True
            continue

        # Evita interpretación adicional cuando ya se entregó la métrica principal.
        if (
            "tendencia" in s_norm
            or "aceler" in s_norm
            or "desaceler" in s_norm
            or "aunque" in s_norm
            or "sin embargo" in s_norm
        ):
            continue

    narrative_clean = " ".join(item.strip() for item in kept if item.strip()).strip()
    parts: List[str] = []
    if narrative_clean:
        parts.append(narrative_clean)
    if source:
        parts.append(source)
    return "\n\n".join(parts).strip()


def _ensure_annual_variation_phrase(response_text: str) -> str:
    narrative, source = _split_narrative_and_source(response_text)
    sentences = _split_sentences(narrative)
    if not sentences:
        return response_text

    if any("variacion anual" in _norm_text(sentence) for sentence in sentences):
        return response_text

    updated = False
    for idx, sentence in enumerate(sentences):
        match = _PERCENT_TOKEN_RE.search(sentence)
        if not match:
            continue
        pct_token = str(match.group(0) or "").strip()
        if not pct_token:
            continue
        sentences[idx] = (
            f"La variación anual fue {pct_token} respecto al mismo período del año anterior."
        )
        updated = True
        break

    if not updated:
        return response_text

    rebuilt = " ".join(item.strip() for item in sentences if item.strip()).strip()
    parts: List[str] = []
    if rebuilt:
        parts.append(rebuilt)
    if source:
        parts.append(source)
    return "\n\n".join(parts).strip()


def _compose_from_sections(sections: Dict[str, str]) -> str:
    intro = str(sections.get("introduccion") or "").strip()
    response = str(sections.get("respuesta") or "").strip()
    sugerencias = str(sections.get("sugerencias") or "").strip()
    csv_block = str(sections.get("csv") or "").strip()

    chunks: List[str] = []
    narrative_parts: List[str] = []
    if intro:
        narrative_parts.append(intro)
    if response:
        narrative_parts.append(response)
    if narrative_parts:
        chunks.append("\n\n".join(narrative_parts).strip())
    if sugerencias:
        chunks.append(sugerencias)
    if csv_block:
        chunks.append(csv_block)
    return "\n\n".join(chunk for chunk in chunks if chunk.strip())


def _postprocess_response_sections(
    sections: Dict[str, str],
    generation_logic: Dict[str, Any],
) -> Tuple[str, Dict[str, str]]:
    intro = _sanitize_wording(str(sections.get("introduccion") or "").strip())
    response_text = _sanitize_wording(str(sections.get("respuesta") or "").strip())
    suggestions = str(sections.get("sugerencias") or "").strip()
    csv_block = str(sections.get("csv") or "").strip()
    mode = str(generation_logic.get("mode") or "").strip().lower()
    question = _extract_generation_question(generation_logic)
    entities = _extract_generation_entities(generation_logic)
    observations = _extract_generation_observations(generation_logic)

    unpublished_override = None
    if mode == "data":
        unpublished_override = _build_unpublished_period_override(
            question=question,
            entities=entities,
            observations=observations,
        )
    if isinstance(unpublished_override, dict):
        intro = str(unpublished_override.get("intro") or intro).strip()
        response_text = str(unpublished_override.get("response") or response_text).strip()

    unpublished_sentence = _extract_unpublished_sentence(response_text)
    if mode == "data" and unpublished_sentence and not _contains_any(_norm_text(intro), _UNPUBLISHED_TOKENS):
        intro = unpublished_sentence

    # Determinar modo yoy ANTES de derivar intro para elegir estrategia correcta
    is_yoy_default = _is_default_yoy_by_question(question, entities, mode)
    requires_original_value = _requires_original_value_response(question, entities, generation_logic)
    if isinstance(unpublished_override, dict):
        is_yoy_default = True
        requires_original_value = False

    if _is_generic_intro(intro):
        intro = _derive_contextual_intro(
            response_text,
            prefer_percent_only=is_yoy_default,
            prefer_value_only=requires_original_value,
        )

    if _is_generic_intro(intro):
        if mode == "data":
            intro = _BASE_INTRO_TEXT
        else:
            intro = "No se encontró una serie compatible para entregar una cifra verificable."

    intro = _enforce_intro_phrase(intro, mode=mode)

    intro_clean = str(intro or "").strip()
    response_clean = str(response_text or "").strip()
    if intro_clean and response_clean.startswith(intro_clean):
        response_clean = response_clean[len(intro_clean) :].lstrip(" \n\t.,:;-")

    if mode == "data" and requires_original_value and not isinstance(unpublished_override, dict):
        response_clean = _enforce_original_value_narrative(response_clean)
        response_clean = _normalize_percentage_decimals(response_clean)

    if is_yoy_default:
        response_clean = _enforce_single_yoy_narrative(intro_clean, response_clean)
        response_clean = _ensure_annual_variation_phrase(response_clean)
        response_clean = _normalize_percentage_decimals(response_clean)
        intro_clean = _normalize_percentage_decimals(intro_clean)

        intro_norm = _norm_text(intro_clean)
        response_norm = _norm_text(response_clean)
        intro_pcts = _PERCENT_RE.findall(intro_clean)
        response_pcts = _PERCENT_RE.findall(response_clean)
        has_prev_period_trace = _contains_any(intro_norm, _PREV_PERIOD_TOKENS) or _contains_any(
            response_norm,
            _PREV_PERIOD_TOKENS,
        )
        has_multi_metric_trace = len(intro_pcts) > 1 or len(response_pcts) > 1
        if has_prev_period_trace or has_multi_metric_trace:
            metric_token = ""
            if intro_pcts:
                metric_token = intro_pcts[0]
            elif response_pcts:
                metric_token = response_pcts[0]
            if metric_token:
                response_clean = (
                    f"La variación anual fue {metric_token} respecto al mismo período del año anterior."
                )
                if not _contains_any(_norm_text(intro_clean), _UNPUBLISHED_TOKENS):
                    intro_clean = _BASE_INTRO_TEXT

        combined_metric_text = f"{intro_clean} {response_clean}".strip()
        if not _contains_any(_norm_text(combined_metric_text), _INTERANUAL_TOKENS):
            pct_token = _extract_first_percentage(combined_metric_text)
            if pct_token:
                response_clean = (
                    f"La variación anual fue {pct_token} respecto al mismo período del año anterior."
                )
                if _PERCENT_RE.search(intro_clean) and not _contains_any(_norm_text(intro_clean), _UNPUBLISHED_TOKENS):
                    intro_clean = _BASE_INTRO_TEXT

    if mode == "data" and is_yoy_default and not isinstance(unpublished_override, dict):
        response_clean = _ensure_yoy_metric_sentence_in_response(intro_clean, response_clean)

    if mode == "data" and not isinstance(unpublished_override, dict):
        intro_clean = _ensure_data_intro_context(
            intro_clean,
            response_clean,
            question=question,
            entities=entities,
            observations=observations,
        )

    if intro_clean and response_clean.startswith(intro_clean):
        response_clean = response_clean[len(intro_clean):].lstrip(" \n\t.,:;-")

    if mode == "data":
        intro_clean = _normalize_percentage_decimals(intro_clean)
        response_clean = _normalize_percentage_decimals(response_clean)
        response_clean = _normalize_contribution_decimals(response_clean)
        response_clean = _ensure_recommendation_paragraph(
            response_clean,
            suggestions,
            question=question,
            observations=observations,
        )
        response_clean = _ensure_bde_source(response_clean, generation_logic)

    updated_sections = {
        "introduccion": intro_clean,
        "respuesta": response_clean,
        "sugerencias": suggestions,
        "csv": csv_block,
    }
    return _compose_from_sections(updated_sections), updated_sections


def _build_response_state(state: AgentState) -> Dict[str, Any]:
    question = str(state.get("question") or "")
    route_decision = str(state.get("route_decision") or "fallback")
    state_response = state.get("response")
    if not isinstance(state_response, dict):
        state_response = state.get("response_contract")

    response_state = (
        dict(state_response)
        if isinstance(state_response, dict)
        else build_response_contract(
            question,
            route_decision=route_decision,
            entities=_primary_entity(state),
            temperature=state.get("response_temperature"),
        )
    )
    if "temperature" in response_state:
        response_state["temperature"] = resolve_response_temperature(
            response_state.get("temperature"),
            fallback=0.2,
        )
    return response_state


def summarize_response_payload(response_payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(response_payload.get("mode") or "").strip().lower()
    summary: Dict[str, Any] = {
        "mode": mode,
        "route_decision": response_payload.get("route_decision"),
    }

    if mode == "data":
        payload = response_payload.get("payload")
        if isinstance(payload, dict):
            observations = payload.get("observations")
            observations = observations if isinstance(observations, dict) else {}
            series = observations.get("series") if isinstance(observations.get("series"), list) else []
            latest_point = _extract_latest_point_summary(
                observations,
                payload.get("entities") if isinstance(payload.get("entities"), dict) else {},
            )
            summary["input"] = {
                "question": payload.get("question"),
                "entities": payload.get("entities"),
                "observations": {
                    "cuadro_name": observations.get("cuadro_name"),
                    "frequency": observations.get("frequency"),
                    "series_count": len(series),
                    "source_url": observations.get("source_url"),
                    "latest_available": observations.get("latest_available"),
                    "latest_point": latest_point,
                    "classification": observations.get("classification"),
                },
            }
        return summary

    if mode in {"rag", "fallback"}:
        history = response_payload.get("history")
        summary["input"] = {
            "question": response_payload.get("question"),
            "history_len": len(history) if isinstance(history, list) else 0,
            "append_methodology_footer": bool(response_payload.get("append_methodology_footer")),
            "intent_info": response_payload.get("intent_info"),
        }
        return summary

    if mode == "prebuilt":
        text = str(response_payload.get("text") or "")
        summary["input"] = {
            "text_preview": text[:240],
            "text_length": len(text),
        }
        return summary

    summary["input"] = {"keys": sorted(response_payload.keys())}
    return summary


def resolve_classification_type(state_snapshot: Dict[str, Any]) -> str:
    response_payload = state_snapshot.get("response_payload") if isinstance(state_snapshot, dict) else None
    if isinstance(response_payload, dict):
        route = str(response_payload.get("route_decision") or "").strip().lower()
        mode = str(response_payload.get("mode") or "").strip().lower()
        if route in {"data", "rag", "fallback"}:
            return route
        if mode in {"data", "rag", "fallback"}:
            return mode

    route_decision = str(state_snapshot.get("route_decision") or "").strip().lower()
    if route_decision in {"data", "rag", "fallback"}:
        return route_decision
    return "fallback"


def serialize_data_store_lookup(state_snapshot: Dict[str, Any]) -> str:
    lookup = state_snapshot.get("data_store_lookup") if isinstance(state_snapshot, dict) else None
    if not isinstance(lookup, dict):
        return "{}"
    return json.dumps(lookup, ensure_ascii=False, separators=(",", ":"), default=str)


def _clean_output_for_length(output: str) -> str:
    cleaned = str(output or "")
    cleaned = re.sub(r"##CSV_DOWNLOAD_START.*?##CSV_DOWNLOAD_END", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"##FOLLOWUP_START.*?##FOLLOWUP_END", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_requested_year(question: str) -> Optional[int]:
    match = _YEAR_RE.search(str(question or ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except (TypeError, ValueError):
        return None


def _extract_requested_period_token(question: str) -> Optional[str]:
    q = str(question or "")
    q_match = _QUARTER_RE.search(q)
    if q_match:
        return str(q_match.group(0)).upper()
    m_match = _MONTH_YEAR_RE.search(q)
    if m_match:
        month_raw = str(m_match.group(1) or "").strip().lower()
        year = str(m_match.group(2) or "").strip()
        month_num = _MONTH_TO_NUM.get(month_raw)
        if month_num:
            return f"{year}-{month_num:02d}"
    y_match = _YEAR_RE.search(q)
    if y_match:
        return str(y_match.group(0))
    return None


def _normalize_frequency_code(freq: Any) -> str:
    raw = str(freq or "").strip().upper()
    if raw in {"Q", "TRIMESTRAL"}:
        return "T"
    if raw in {"M", "T", "A"}:
        return raw
    return ""


def _canonicalize_period(freq: str, token: str) -> Optional[str]:
    text = str(token or "").strip()
    f = _normalize_frequency_code(freq)
    if not text or not f:
        return None
    if f == "M":
        if re.fullmatch(r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])", text):
            return text
    elif f == "T":
        if re.fullmatch(r"(?:19|20)\d{2}-Q[1-4]", text, flags=re.IGNORECASE):
            return text.upper()
    elif f == "A":
        if re.fullmatch(r"(?:19|20)\d{2}", text):
            return text
    return None


def _compare_period_tokens(freq: str, requested: str, latest: str) -> Optional[int]:
    req = _canonicalize_period(freq, requested)
    lat = _canonicalize_period(freq, latest)
    if not req or not lat:
        return None
    if req < lat:
        return -1
    if req > lat:
        return 1
    return 0


def _extract_requested_period_token_from_entities(entities: Dict[str, Any], freq: str) -> Optional[str]:
    values = entities.get("period_ent")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return None
    for item in reversed(values):
        token = _canonicalize_period(freq, str(item or "").strip())
        if token:
            return token
    return None


def _extract_relative_period_token(question: str, freq: str) -> Optional[str]:
    q_norm = _norm_text(question)
    f = _normalize_frequency_code(freq)
    if not q_norm or not f:
        return None

    today = _dt.date.today()
    if f == "M":
        if "mes pasado" in q_norm or "ultimo mes" in q_norm:
            year = today.year if today.month > 1 else today.year - 1
            month = today.month - 1 if today.month > 1 else 12
            return f"{year}-{month:02d}"
        if "este mes" in q_norm:
            return f"{today.year}-{today.month:02d}"

    if f == "T":
        quarter = ((today.month - 1) // 3) + 1
        if "trimestre pasado" in q_norm or "ultimo trimestre" in q_norm:
            if quarter == 1:
                return f"{today.year - 1}-Q4"
            return f"{today.year}-Q{quarter - 1}"
        if "este trimestre" in q_norm:
            return f"{today.year}-Q{quarter}"

    if f == "A":
        if "hace dos anos" in q_norm:
            return str(today.year - 2)
        if "ano pasado" in q_norm or "ultimo ano" in q_norm:
            return str(today.year - 1)
        if "este ano" in q_norm:
            return str(today.year)

    return None


def _resolve_requested_period_token(
    question: str,
    entities: Dict[str, Any],
    freq: str,
) -> Optional[str]:
    f = _normalize_frequency_code(freq)
    if not f:
        return None

    explicit = _extract_requested_period_token(question)
    if explicit:
        explicit_token = _canonicalize_period(f, explicit)
        if explicit_token:
            return explicit_token

    entity_token = _extract_requested_period_token_from_entities(entities, f)
    if entity_token:
        return entity_token

    return _extract_relative_period_token(question, f)


def _natural_period_label(token: str, freq: str) -> str:
    f = _normalize_frequency_code(freq)
    period = _canonicalize_period(f, token)
    if not period:
        return str(token or "").strip()

    if f == "M":
        year, month = period.split("-", 1)
        try:
            month_num = int(month)
        except ValueError:
            return period
        month_label = _NUM_TO_MONTH.get(month_num)
        if not month_label:
            return period
        return f"{month_label} de {year}"

    if f == "T":
        match = re.fullmatch(r"((?:19|20)\d{2})-Q([1-4])", period, flags=re.IGNORECASE)
        if not match:
            return period
        year = match.group(1)
        quarter = int(match.group(2))
        return f"{_QUARTER_TO_LABEL.get(quarter, period)} de {year}"

    return period


def _format_percent_token(value: Any, *, decimals: int = 1) -> str:
    raw = str(value or "").strip().replace("%", "")
    if not raw:
        return ""
    parsed = _to_float(raw)
    if parsed is None:
        return ""
    return f"{_format_decimal(parsed, decimals)}%"


def _format_value_token(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = _to_float(raw)
    if parsed is None:
        return ""
    decimals = 2 if abs(parsed) < 1000 else 1
    return _format_decimal(parsed, decimals)


def _extract_latest_point_summary(
    observations: Dict[str, Any],
    entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not isinstance(observations, dict):
        return {}

    existing = observations.get("latest_point")
    if isinstance(existing, dict):
        has_period = str(existing.get("period") or "").strip() != ""
        has_metric = any(existing.get(key) is not None for key in ("yoy_pct", "pct", "value"))
        if has_period or has_metric:
            return dict(existing)

    freq = _normalize_frequency_code(observations.get("frequency"))
    if not freq and isinstance(entities, dict):
        freq = _normalize_frequency_code(
            entities.get("frequency_ent")
            or entities.get("frequency")
        )

    latest_available = observations.get("latest_available")
    latest_available = latest_available if isinstance(latest_available, dict) else {}
    if not freq:
        for candidate in ("M", "T", "A"):
            if str(latest_available.get(candidate) or "").strip():
                freq = candidate
                break
    if not freq:
        return {}

    latest_token = _canonicalize_period(freq, str(latest_available.get(freq) or "")) or ""
    series = observations.get("series")
    if not isinstance(series, list):
        return {}

    for series_item in series:
        if not isinstance(series_item, dict):
            continue
        data = series_item.get("data")
        if not isinstance(data, dict):
            continue
        freq_block = data.get(freq)
        if not isinstance(freq_block, dict):
            continue
        records = freq_block.get("records")
        if not isinstance(records, list) or not records:
            continue

        selected: Optional[Dict[str, Any]] = None
        if latest_token:
            for record in reversed(records):
                if not isinstance(record, dict):
                    continue
                if str(record.get("period") or "").strip() == latest_token:
                    selected = record
                    break

        if selected is None:
            for record in reversed(records):
                if isinstance(record, dict):
                    selected = record
                    break

        if not isinstance(selected, dict):
            continue

        period = str(
            selected.get("period")
            or latest_token
            or freq_block.get("latest_period")
            or ""
        ).strip()

        return {
            "frequency": freq,
            "period": _canonicalize_period(freq, period) or period,
            "series_id": series_item.get("series_id"),
            "short_title": series_item.get("short_title"),
            "value": selected.get("value"),
            "pct": selected.get("pct"),
            "yoy_pct": selected.get("yoy_pct"),
        }

    return {}


def _pick_text_variant(seed: str, variants: List[str]) -> str:
    if not variants:
        return ""
    digest = hashlib.sha256(str(seed or "").encode("utf-8")).hexdigest()
    idx = int(digest[:8], 16) % len(variants)
    return variants[idx]


def _indicator_label_from_observations(observations: Dict[str, Any], latest_point: Dict[str, Any]) -> str:
    classification = (
        observations.get("classification")
        if isinstance(observations.get("classification"), dict)
        else {}
    )
    indicator = _norm_text(classification.get("indicator") or "")
    if indicator == "imacec":
        return "IMACEC"
    if indicator == "pib":
        return "PIB"
    if indicator == "demanda interna":
        return "demanda interna"

    short_title = str(latest_point.get("short_title") or "").strip()
    if short_title:
        return short_title
    return ""


def _subject_for_unpublished_response(question: str, indicator_label: str) -> str:
    q_norm = _norm_text(question)
    label = str(indicator_label or "").strip()
    if label == "IMACEC":
        if "economia" in q_norm:
            return "la economía, medida por el IMACEC"
        return "el IMACEC"
    if label == "PIB":
        if "economia" in q_norm:
            return "la economía, medida por el PIB"
        return "el PIB"
    if label.lower() == "demanda interna":
        return "la demanda interna"
    return "el indicador consultado"


def _frequency_label(freq: str) -> str:
    code = _normalize_frequency_code(freq)
    if code == "M":
        return "mensual"
    if code == "T":
        return "trimestral"
    if code == "A":
        return "anual"
    return ""


def _build_frequency_note(question: str, freq: str) -> str:
    q_norm = _norm_text(question)
    code = _normalize_frequency_code(freq)
    if not code:
        return ""

    has_month_reference = "mes" in q_norm or _contains_any(q_norm, _MONTH_TOKENS)
    has_quarter_reference = "trimestre" in q_norm
    has_year_reference = bool(_YEAR_RE.search(question)) or "ano" in q_norm or "año" in str(question or "").lower()

    if code == "T" and has_month_reference:
        return "La serie disponible se publica en frecuencia trimestral."
    if code == "M" and has_quarter_reference:
        return "La serie disponible se publica en frecuencia mensual."
    if code == "A" and (has_month_reference or has_quarter_reference):
        return "La serie disponible se publica en frecuencia anual."

    if not has_month_reference and not has_quarter_reference and not has_year_reference:
        label = _frequency_label(code)
        if label:
            return f"La serie se reporta con frecuencia {label}."

    return ""


def _subject_for_data_context(question: str, indicator_label: str, freq: str) -> str:
    q_norm = _norm_text(question)
    label = str(indicator_label or "").strip()
    label_norm = _norm_text(label)

    if "demanda interna" in q_norm:
        return "la demanda interna"
    if "inversion" in q_norm or "inversión" in str(question or "").lower():
        return "la inversión en Chile"
    if label == "IMACEC":
        if "economia" in q_norm:
            return "la economía, medida por el IMACEC"
        return "el IMACEC"
    if label == "PIB":
        if "economia" in q_norm:
            return "la economía, medida por el PIB"
        return "el PIB"
    if label_norm == "demanda interna":
        return "la demanda interna"
    if label:
        if label_norm.startswith(("el ", "la ")):
            return label
        return f"el indicador {label}"

    code = _normalize_frequency_code(freq)
    if "economia" in q_norm:
        if code == "M":
            return "la economía, medida por el IMACEC"
        if code in {"T", "A"}:
            return "la economía, medida por el PIB"
        return "la economía"
    return "el indicador consultado"


def _has_indicator_reference(text: str, question: str, indicator_label: str) -> bool:
    t_norm = _norm_text(text)
    q_norm = _norm_text(question)
    label_norm = _norm_text(indicator_label)

    if label_norm:
        if label_norm in t_norm:
            return True
        if label_norm == "pib" and ("pib" in t_norm or "producto interno bruto" in t_norm):
            return True
        if label_norm == "imacec" and "imacec" in t_norm:
            return True
        if label_norm == "demanda interna" and "demanda interna" in t_norm:
            return True

    if "inversion" in q_norm and "inversion" in t_norm:
        return True
    if "demanda interna" in q_norm and "demanda interna" in t_norm:
        return True
    if "economia" in q_norm and ("pib" in t_norm or "imacec" in t_norm):
        return True

    return False


def _resolve_reference_period(
    question: str,
    entities: Dict[str, Any],
    observations: Dict[str, Any],
    latest_point: Dict[str, Any],
) -> Tuple[str, str, bool]:
    freq = _normalize_frequency_code(
        latest_point.get("frequency")
        or observations.get("frequency")
        or entities.get("frequency_ent")
        or entities.get("frequency")
    )
    if not freq:
        return "", "", False

    latest_available = observations.get("latest_available")
    latest_available = latest_available if isinstance(latest_available, dict) else {}
    latest_token = _canonicalize_period(
        freq,
        str(latest_point.get("period") or latest_available.get(freq) or ""),
    )
    requested_token = _resolve_requested_period_token(question, entities, freq)

    selected_token = latest_token
    requested_non_latest = False
    if requested_token and latest_token:
        cmp_result = _compare_period_tokens(freq, requested_token, latest_token)
        if cmp_result in {-1, 0}:
            selected_token = requested_token
            requested_non_latest = requested_token != latest_token
    elif requested_token and not latest_token:
        selected_token = requested_token
        requested_non_latest = True

    period_label = _natural_period_label(selected_token, freq) if selected_token else ""
    return freq, period_label, requested_non_latest


def _ensure_data_intro_context(
    intro: str,
    response_text: str,
    *,
    question: str,
    entities: Dict[str, Any],
    observations: Dict[str, Any],
) -> str:
    narrative, _source = _split_narrative_and_source(response_text)
    core_paragraphs = [
        paragraph.strip()
        for paragraph in narrative.split("\n\n")
        if paragraph.strip() and not _is_recommendation_paragraph(paragraph)
    ]
    response_core = "\n\n".join(core_paragraphs).strip()

    latest_point = _extract_latest_point_summary(observations, entities)
    indicator_label = _indicator_label_from_observations(observations, latest_point)
    freq, period_label, requested_non_latest = _resolve_reference_period(
        question,
        entities,
        observations,
        latest_point,
    )
    subject = _subject_for_data_context(question, indicator_label, freq)
    combined = f"{intro} {response_core}".strip()

    has_period = _has_period_anchor(intro) or _has_period_anchor(response_core)
    has_indicator = _has_indicator_reference(combined, question, indicator_label)
    needs_context = _is_generic_intro(intro) or not has_period or not has_indicator
    if not needs_context:
        return intro

    if period_label:
        if requested_non_latest:
            rebuilt_intro = f"Para {period_label}, {subject} corresponde al período consultado."
        else:
            rebuilt_intro = f"El último dato disponible para {subject} corresponde a {period_label}."
    else:
        freq_label = _frequency_label(freq)
        if freq_label:
            rebuilt_intro = f"La consulta para {subject} se responde con frecuencia {freq_label}."
        else:
            rebuilt_intro = intro or _BASE_INTRO_TEXT

    frequency_note = _build_frequency_note(question, freq)
    if frequency_note and _norm_text(frequency_note) not in _norm_text(rebuilt_intro):
        rebuilt_intro = f"{rebuilt_intro} {frequency_note}".strip()

    return rebuilt_intro


def _ensure_yoy_metric_sentence_in_response(intro: str, response_text: str) -> str:
    narrative, source = _split_narrative_and_source(response_text)
    if _PERCENT_RE.search(narrative):
        return response_text

    pct_token = _extract_first_percentage(intro)
    if not pct_token:
        return response_text

    intro_norm = _norm_text(intro)
    if not pct_token.strip().startswith("-") and any(
        marker in intro_norm for marker in ("disminuy", "retroced", "baj", "cayo", "cayó")
    ):
        pct_token = f"-{pct_token}"

    metric_sentence = f"La variación anual fue {pct_token} respecto al mismo período del año anterior."
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    if paragraphs and _norm_text(paragraphs[0]) == _norm_text(metric_sentence):
        rebuilt_narrative = "\n\n".join(paragraphs).strip()
    else:
        paragraphs.insert(0, metric_sentence)
        rebuilt_narrative = "\n\n".join(paragraphs).strip()

    parts: List[str] = []
    if rebuilt_narrative:
        parts.append(rebuilt_narrative)
    if source:
        parts.append(source)
    return "\n\n".join(parts).strip()


def _build_unpublished_period_override(
    *,
    question: str,
    entities: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    if not isinstance(observations, dict):
        return None

    freq = _normalize_frequency_code(
        observations.get("frequency")
        or entities.get("frequency_ent")
        or entities.get("frequency")
    )
    if not freq:
        return None

    latest_available = observations.get("latest_available")
    latest_available = latest_available if isinstance(latest_available, dict) else {}
    latest_for_freq = _canonicalize_period(freq, str(latest_available.get(freq) or ""))
    if not latest_for_freq:
        return None

    requested_token = _resolve_requested_period_token(question, entities, freq)
    if not requested_token:
        return None

    period_cmp = _compare_period_tokens(freq, requested_token, latest_for_freq)
    if period_cmp != 1:
        return None

    requested_label = _natural_period_label(requested_token, freq)
    latest_label = _natural_period_label(latest_for_freq, freq)
    latest_point = _extract_latest_point_summary(observations, entities)
    indicator_label = _indicator_label_from_observations(observations, latest_point)
    subject = _subject_for_unpublished_response(question, indicator_label)
    connector = _pick_text_variant(
        _norm_text(question),
        [
            "De acuerdo con lo anterior",
            "En ese contexto",
            "Con esa referencia",
        ],
    )

    yoy_metric = _format_percent_token(latest_point.get("yoy_pct"), decimals=1)
    if not yoy_metric:
        yoy_metric = _format_percent_token(latest_point.get("pct"), decimals=1)

    intro = f"Los datos de {requested_label} aún no han sido publicados según los datos de la BDE."
    if yoy_metric:
        comparison_tail = "respecto al mismo período del año anterior"
        if indicator_label in {"IMACEC", "PIB"}:
            comparison_tail = f"respecto al {indicator_label} del mismo período del año anterior"
        response = (
            f"El último dato disponible corresponde a {latest_label}. "
            f"{connector}, {subject} registró una variación anual de **{yoy_metric}** {comparison_tail}."
        )
    else:
        value_metric = _format_value_token(latest_point.get("value"))
        if value_metric:
            response = (
                f"El último dato disponible corresponde a {latest_label}. "
                f"{connector}, {subject} registró un valor de **{value_metric}**."
            )
        else:
            response = f"El último dato disponible corresponde a {latest_label}."

    return {
        "intro": intro,
        "response": response,
    }


def _extract_observations_summary(generation_logic: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(generation_logic, dict):
        return {}
    return _extract_generation_observations(generation_logic)


def _extract_generation_mode(generation_logic: Optional[Dict[str, Any]]) -> str:
    if not isinstance(generation_logic, dict):
        return ""
    return str(generation_logic.get("mode") or "").strip().lower()


def _strip_control_blocks(output: str) -> str:
    text = str(output or "")
    without_csv, _csv_block = _split_marker_block(text, _CSV_BLOCK_RE)
    without_followup, _followup_block = _split_marker_block(without_csv, _FOLLOWUP_BLOCK_RE)
    return without_followup.strip()


def _split_narrative_paragraphs(output: str) -> List[str]:
    core = _strip_control_blocks(output)
    return [paragraph.strip() for paragraph in core.split("\n\n") if paragraph.strip()]


def _extract_intro_text(output: str) -> str:
    text = str(output or "")
    if not text:
        return ""
    match = _INTRO_SECTION_RE.search(text)
    if match:
        return str(match.group(1) or "").strip()

    paragraphs = _split_narrative_paragraphs(text)
    if len(paragraphs) >= 2:
        return paragraphs[0]
    if not paragraphs:
        return ""

    sentences = _split_sentences(paragraphs[0])
    if sentences:
        return sentences[0].strip()
    return paragraphs[0]


def _extract_response_text(output: str) -> str:
    text = str(output or "")
    if not text:
        return ""
    match = _RESPONSE_SECTION_RE.search(text)
    if match:
        return str(match.group(1) or "").strip()

    paragraphs = _split_narrative_paragraphs(text)
    if len(paragraphs) >= 2:
        return "\n\n".join(paragraphs[1:]).strip()
    if not paragraphs:
        return ""

    sentences = _split_sentences(paragraphs[0])
    if len(sentences) >= 2:
        return " ".join(sentence.strip() for sentence in sentences[1:] if sentence.strip()).strip()
    return paragraphs[0]


def _has_recommendation_final_paragraph(response_text: str) -> bool:
    narrative, _source = _split_narrative_and_source(response_text)
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    if not paragraphs:
        return False
    last_norm = _norm_acceptance(paragraphs[-1])
    return _contains_any(last_norm, _RECOMMENDATION_TOKENS)


def _requires_original_value_case(
    *,
    question_norm: str,
    generation_entities: Dict[str, Any],
    observations: Dict[str, Any],
) -> bool:
    if not _is_pib_context(question_norm, generation_entities, observations):
        return False

    if _contains_any(question_norm, _ORIGINAL_VALUE_TOKENS):
        return True
    if _contains_any(question_norm, _PIB_ORIGINAL_VALUE_HINTS):
        return True

    calc_mode = str(
        generation_entities.get("calc_mode_cls")
        or generation_entities.get("calc_mode")
        or ""
    ).strip().lower()
    if calc_mode in {"original_value", "nominal"}:
        return True

    classification = (
        observations.get("classification")
        if isinstance(observations.get("classification"), dict)
        else {}
    )
    price = _norm_acceptance(classification.get("price") or "")
    if price == "co" and calc_mode == "value" and _contains_any(question_norm, _VALUE_QUERY_HINTS):
        return True

    return False


def _extract_contribution_decimals(response_text: str) -> List[int]:
    pattern = re.compile(
        r"(?:contribuci[oó]n|contribuy[oó]|aport[oó])[^\n]{0,80}?"
        r"(?:con|de)?\s*\*\*(-?\d+(?:[\.,]\d+)?)\*\*"
        r"(?:\s*(?:puntos?\s+porcentuales|pp|%))?",
        re.IGNORECASE,
    )
    decimals: List[int] = []
    for match in pattern.finditer(str(response_text or "")):
        raw = str(match.group(1) or "")
        if "." in raw:
            decimals.append(len(raw.split(".", 1)[1]))
        elif "," in raw:
            decimals.append(len(raw.split(",", 1)[1]))
        else:
            decimals.append(0)
    return decimals


def _validate_intro_context(
    intro_text: str,
    *,
    is_data_mode: bool,
    mentions_unpublished: bool,
) -> Tuple[bool, str]:
    intro = str(intro_text or "").strip()
    intro_norm = _norm_acceptance(intro)
    if not intro_norm:
        return False, "Introducción vacía."
    generic_intro = _contains_any(intro_norm, _GENERIC_INTRO_TOKENS)
    forbidden_tail = _contains_any(intro_norm, _FORBIDDEN_INTRO_TAIL_TOKENS)
    is_base_intro = intro_norm in {
        "con base en los datos disponibles",
        "con base en los datos disponibles.",
    }
    has_period_anchor = (
        bool(_YEAR_RE.search(intro))
        or bool(_MONTH_YEAR_RE.search(intro))
        or bool(_QUARTER_RE.search(intro))
        or "trimestre" in intro_norm
        or "mes" in intro_norm
        or "ano" in intro_norm
        or "año" in intro_norm
    )
    has_numeric_value = bool(_PERCENT_RE.search(intro) or re.search(r"\b\d+(?:[\.,]\d+)?\b", intro))
    intro_mentions_unpublished = "public" in intro_norm and (
        "aun no" in intro_norm
        or "aún no" in intro_norm
        or "no han sido" in intro_norm
        or "no esta" in intro_norm
        or "no está" in intro_norm
    )
    if not is_data_mode:
        ok = not forbidden_tail
        return ok, f"Modo no-data: tail_prohibida={forbidden_tail}."
    if mentions_unpublished:
        ok = (not forbidden_tail) and has_period_anchor and intro_mentions_unpublished
        return (
            ok,
            "Caso no publicado: "
            f"tail_prohibida={forbidden_tail}, periodo={has_period_anchor}, "
            f"aviso_no_publicado={intro_mentions_unpublished}.",
        )
    ok = (not forbidden_tail) and (is_base_intro or has_period_anchor or has_numeric_value)
    return (
        ok,
        "Caso dato directo: "
        f"base_intro={is_base_intro}, tail_prohibida={forbidden_tail}, "
        f"periodo={has_period_anchor}, valor={has_numeric_value}, generica={generic_intro}.",
    )


def _published_quarters_for_year(year: int, latest_t: str) -> Optional[int]:
    match = re.fullmatch(r"((?:19|20)\d{2})-Q([1-4])", str(latest_t or ""), flags=re.IGNORECASE)
    if not match:
        return None
    latest_year = int(match.group(1))
    latest_q = int(match.group(2))
    if year < latest_year:
        return 4
    if year > latest_year:
        return 0
    return latest_q


@dataclass
class _Check:
    check_id: str
    title: str
    ok: bool
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.check_id,
            "title": self.title,
            "ok": bool(self.ok),
            "status": "ok" if self.ok else "no_ok",
            "detail": self.detail,
        }


def _norm_acceptance(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _extract_percentages(text: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for match in _PERCENT_VALUE_RE.finditer(text):
        raw = str(match.group(1) or "").strip()
        if not raw:
            continue
        decimals = 0
        if "." in raw:
            decimals = len(raw.split(".", 1)[1])
        elif "," in raw:
            decimals = len(raw.split(",", 1)[1])
        items.append((raw, decimals))
    return items


# ── RuleContext: contexto compartido pre-computado para todas las reglas ───────
@dataclass
class RuleContext:
    """Todas las señales pre-computadas que las reglas necesitan.

    Se construye UNA vez por invocación de evaluate_acceptance_criteria y se
    pasa a cada función _rule_*.  Así ninguna regla repite extracción de datos.
    """
    question: str
    q_norm: str
    output: str
    out_norm: str
    intro_text: str
    response_text: str
    response_norm: str
    core_text: str
    core_norm: str
    mode: str
    is_data_mode: bool
    generation_entities: Dict[str, Any]
    calc_mode: str
    observations: Dict[str, Any]
    percentages: List[Tuple[str, int]]
    percentage_count: int
    economic_value_count: int
    prev_period_exception: bool
    original_value_exception: bool
    contribution_exception: bool
    default_yoy_applies: bool
    mentions_annual_variation: bool
    has_interanual_word: bool
    has_bde_reference: bool
    mentions_unpublished: bool
    mentions_latest: bool
    freq: str
    latest_available: Dict[str, Any]
    requested_token: Optional[str]
    latest_for_freq: str
    period_cmp: Optional[int]
    unavailable_expected: bool
    requested_year: Optional[int]


def _build_rule_context(
    question: str,
    output: str,
    generation_logic: Optional[Dict[str, Any]],
) -> RuleContext:
    q_norm = _norm_acceptance(question)
    out_norm = _norm_acceptance(output)
    intro_text = _extract_intro_text(output)
    response_text = _extract_response_text(output)
    response_norm = _norm_acceptance(response_text)
    core_text = f"{intro_text}\n{response_text}".strip()
    core_norm = _norm_acceptance(core_text)
    mode = _extract_generation_mode(generation_logic)
    is_data_mode = mode == "data"
    generation_entities = _extract_generation_entities(generation_logic or {})
    calc_mode = str(
        generation_entities.get("calc_mode_cls")
        or generation_entities.get("calc_mode")
        or ""
    ).strip().lower()
    observations = _extract_observations_summary(generation_logic)

    percentages = _extract_percentages(str(output or ""))
    percentage_count = len(percentages)
    economic_value_count = _count_economic_values(core_text)

    prev_period_exception = _contains_any(q_norm, _PREV_PERIOD_TOKENS)
    original_value_exception = _requires_original_value_case(
        question_norm=q_norm,
        generation_entities=generation_entities,
        observations=observations,
    )
    contribution_exception = calc_mode == "contribution" or _contains_any(q_norm, _CONTRIBUTION_TOKENS)
    default_yoy_applies = _is_default_yoy_by_question(question, generation_entities, mode)
    if prev_period_exception or original_value_exception or contribution_exception:
        default_yoy_applies = False

    mentions_annual_variation = _contains_any(core_norm, _INTERANUAL_TOKENS)
    has_interanual_word = bool(_INTERANUAL_WORD_RE.search(core_text))
    has_bde_reference = (
        "fuente" in out_norm
        and (
            "bde" in out_norm
            or "base de datos estad" in out_norm
            or "si3.bcentral.cl" in out_norm
        )
    )
    mentions_unpublished = "public" in core_norm and (
        "aun no" in core_norm or "aún no" in core_norm
        or "no esta" in core_norm or "no está" in core_norm
    )
    mentions_latest = "ultimo" in core_norm or "último" in core_norm

    freq = _normalize_frequency_code(observations.get("frequency"))
    latest_available = (
        observations.get("latest_available")
        if isinstance(observations.get("latest_available"), dict)
        else {}
    )
    requested_token = _resolve_requested_period_token(question, generation_entities, freq)
    latest_for_freq = str(latest_available.get(freq) or "").strip() if freq else ""
    period_cmp = (
        _compare_period_tokens(freq, requested_token or "", latest_for_freq)
        if (freq and requested_token and latest_for_freq)
        else None
    )
    unavailable_expected = period_cmp == 1
    requested_year = _extract_requested_year(question)

    return RuleContext(
        question=question,
        q_norm=q_norm,
        output=output,
        out_norm=out_norm,
        intro_text=intro_text,
        response_text=response_text,
        response_norm=response_norm,
        core_text=core_text,
        core_norm=core_norm,
        mode=mode,
        is_data_mode=is_data_mode,
        generation_entities=generation_entities,
        calc_mode=calc_mode,
        observations=observations,
        percentages=percentages,
        percentage_count=percentage_count,
        economic_value_count=economic_value_count,
        prev_period_exception=prev_period_exception,
        original_value_exception=original_value_exception,
        contribution_exception=contribution_exception,
        default_yoy_applies=default_yoy_applies,
        mentions_annual_variation=mentions_annual_variation,
        has_interanual_word=has_interanual_word,
        has_bde_reference=has_bde_reference,
        mentions_unpublished=mentions_unpublished,
        mentions_latest=mentions_latest,
        freq=freq,
        latest_available=latest_available,
        requested_token=requested_token,
        latest_for_freq=latest_for_freq,
        period_cmp=period_cmp,
        unavailable_expected=unavailable_expected,
        requested_year=requested_year,
    )


def _yoy_exception_reason(ctx: RuleContext) -> str:
    if not ctx.is_data_mode:
        return f"modo_{ctx.mode or 'no_data'}"
    if ctx.contribution_exception:
        return "contribution"
    if ctx.prev_period_exception:
        return "prev_period"
    return "valor_original"


# ── Registro de reglas de aceptación ──────────────────────────────────────────
# Cada regla tiene UNA función _rule_* que recibe un RuleContext y devuelve _Check.
# Para entender qué hace cada regla, basta leer esta tabla:
#
#   RULE_ID                                        → FUNCIÓN EVALUADORA
#   ─────────────────────────────────────────────── ─────────────────────────────
#   1.  dato_unico_variacion_yoy_por_defecto       → _rule_single_yoy_default
#   2.  solo_yoy_salvo_excepciones                 → _rule_only_yoy_no_exceptions
#   3.  variaciones_un_decimal                     → _rule_percent_one_decimal
#   4.  formato_respuesta_tipo                     → _rule_response_format_type
#   5.  razonamiento_a_validacion_fecha            → _rule_reasoning_a_date
#   6.  razonamiento_b_entrega_valores             → _rule_reasoning_b_values
#   7.  razonamiento_c_referencia_cuadro           → _rule_reasoning_c_source
#   8.  razonamiento_d_crecimiento_economia        → _rule_reasoning_d_growth
#   9.  anual_usar_pib_no_imacec                   → _rule_annual_pib_not_imacec
#   10. pib_anual_requiere_4_trimestres            → _rule_pib_annual_4q
#   11. introduccion_contextual_fecha_o_dato       → _rule_intro_contextual
#   12. introduccion_sin_cola_prohibida            → _rule_intro_no_forbidden_tail
#   13. recomendacion_parrafo_final                → _rule_recommendation_paragraph
#   14. sin_palabra_interanual                     → _rule_no_interanual_word
#   15. valor_original_pib_casos                   → _rule_original_value_case
#   16. contribucion_precision_un_decimal          → _rule_contribution_one_decimal
#   17. sin_opinion_de_datos                       → _rule_no_subjective_opinion
#   18. sin_palabra_cayo                           → _rule_no_cayo_word
#   19. largo_respuesta_minimo                     → _rule_minimum_length


def _rule_single_yoy_default(ctx: RuleContext) -> _Check:
    """Regla 1: En modo data sin excepciones, la respuesta debe tener exactamente
    UN dato económico visible y debe ser una variación anual (no interanual).
    Cuenta TODOS los valores económicos (intro + respuesta), no solo %."""
    rid = "dato_unico_variacion_yoy_por_defecto"
    title = "Entrega un solo dato por defecto como variación anual"
    if not ctx.default_yoy_applies:
        return _Check(rid, title, True,
                      f"No aplica: excepción ({_yoy_exception_reason(ctx)}).")
    ok = (
        ctx.percentage_count == 1
        and ctx.economic_value_count == 1
        and ctx.mentions_annual_variation
        and not ctx.has_interanual_word
    )
    return _Check(rid, title, ok,
        f"porcentajes={ctx.percentage_count}, valores_econ={ctx.economic_value_count}, "
        f"variacion_anual={ctx.mentions_annual_variation}, interanual={ctx.has_interanual_word}.")


def _rule_only_yoy_no_exceptions(ctx: RuleContext) -> _Check:
    """Regla 2: Cuando aplica yoy por defecto, solo debe mencionar variación anual
    (no periodo anterior ni interanual)."""
    rid = "solo_yoy_salvo_excepciones"
    title = "Solo variación del mismo periodo del año anterior salvo excepciones"
    if not ctx.default_yoy_applies:
        return _Check(rid, title, True, "No aplica por excepción permitida.")
    has_prev_period_wording = _contains_any(ctx.out_norm, _PREV_PERIOD_TOKENS)
    ok = (
        ctx.mentions_annual_variation
        and not has_prev_period_wording
        and not ctx.has_interanual_word
    )
    return _Check(rid, title, ok,
        f"variacion_anual={ctx.mentions_annual_variation}, "
        f"menciona_periodo_anterior={has_prev_period_wording}, "
        f"interanual={ctx.has_interanual_word}.")


def _rule_percent_one_decimal(ctx: RuleContext) -> _Check:
    """Regla 3: Todos los porcentajes deben tener exactamente 1 decimal."""
    rid = "variaciones_un_decimal"
    title = "Las variaciones deben salir con exactamente un decimal"
    if ctx.percentage_count == 0:
        ok = not ctx.default_yoy_applies
        return _Check(rid, title, ok,
            "Sin porcentajes en salida; permitido si no aplica yoy por defecto.")
    bad = [raw for raw, dec in ctx.percentages if dec != 1]
    ok = len(bad) == 0
    return _Check(rid, title, ok,
        f"porcentajes={ctx.percentages}; invalidos={bad}.")


def _rule_response_format_type(ctx: RuleContext) -> _Check:
    """Regla 4: Formato correcto según tipo (incluye caso periodo no publicado)."""
    rid = "formato_respuesta_tipo"
    title = "Formato de respuesta según tipo (incluye caso no publicado)"
    if not (ctx.is_data_mode and ctx.unavailable_expected):
        if not ctx.is_data_mode:
            detail = f"No aplica: modo={ctx.mode or 'none'}."
        else:
            detail = "No aplica formato de periodo no publicado."
        return _Check(rid, title, True, detail)
    ok = (
        ctx.mentions_unpublished
        and ctx.mentions_latest
        and ctx.mentions_annual_variation
        and not ctx.has_interanual_word
    )
    return _Check(rid, title, ok,
        f"periodo_solicitado>{ctx.latest_for_freq}; no_publicado={ctx.mentions_unpublished}, "
        f"menciona_ultimo={ctx.mentions_latest}, variacion_anual={ctx.mentions_annual_variation}, "
        f"interanual={ctx.has_interanual_word}.")


def _rule_reasoning_a_date(ctx: RuleContext) -> _Check:
    """Regla 5: Razonamiento A — validación de última fecha de publicación.
    Comparte resultado con regla 4."""
    r4 = _rule_response_format_type(ctx)
    return _Check(
        "razonamiento_a_validacion_fecha",
        "Razonamiento A: validación de última fecha de publicación",
        r4.ok,
        r4.detail,
    )


def _rule_reasoning_b_values(ctx: RuleContext) -> _Check:
    """Regla 6: Razonamiento B — entrega de valores según reglas.
    Composición de reglas 1 + 2 + 3."""
    r1 = _rule_single_yoy_default(ctx)
    r2 = _rule_only_yoy_no_exceptions(ctx)
    r3 = _rule_percent_one_decimal(ctx)
    ok = r1.ok and r2.ok and r3.ok
    return _Check(
        "razonamiento_b_entrega_valores",
        "Razonamiento B: entrega de valores según reglas",
        ok,
        f"R1={r1.ok}, R2={r2.ok}, R3={r3.ok}.",
    )


def _rule_reasoning_c_source(ctx: RuleContext) -> _Check:
    """Regla 7: Razonamiento C — referencia al cuadro/fuente (BDE)."""
    rid = "razonamiento_c_referencia_cuadro"
    title = "Razonamiento C: referencia al cuadro/fuente (BDE)"
    if not ctx.is_data_mode:
        return _Check(rid, title, True, f"No aplica: modo={ctx.mode or 'none'}.")
    return _Check(rid, title, ctx.has_bde_reference,
        f"referencia_BDE={ctx.has_bde_reference}.")


def _rule_reasoning_d_growth(ctx: RuleContext) -> _Check:
    """Regla 8: Razonamiento D — caso 'cuánto creció la economía el último trimestre'."""
    rid = "razonamiento_d_crecimiento_economia_ultimo_trimestre"
    title = "Razonamiento D: caso 'Cuánto creció la economía el último trimestre'"
    growth_q = (
        ("cuanto crecio la economia" in ctx.q_norm or "cuánto creció la economía" in ctx.q_norm)
        and ("ultimo trimestre" in ctx.q_norm or "último trimestre" in ctx.q_norm)
    )
    if not (ctx.is_data_mode and growth_q):
        return _Check(rid, title, True, "No aplica (pregunta no corresponde al caso D).")
    mentions_economy = _contains_any(ctx.out_norm, _ECONOMY_TOKENS)
    ok = mentions_economy and ctx.percentage_count >= 1
    return _Check(rid, title, ok,
        f"porcentajes={ctx.percentage_count}, menciona_economia={mentions_economy}.")


def _rule_annual_pib_not_imacec(ctx: RuleContext) -> _Check:
    """Regla 9: Si es cifra anual, usar PIB y no IMACEC."""
    rid = "anual_usar_pib_no_imacec"
    title = "Si es cifra anual, usar PIB y no IMACEC"
    annual_context = ctx.requested_year is not None and (
        "anual" in ctx.q_norm or "año" in ctx.q_norm or "ano" in ctx.q_norm
        or _contains_any(ctx.q_norm, _ECONOMY_TOKENS)
        or "pib" in ctx.q_norm or "imacec" in ctx.q_norm
    )
    if not (ctx.is_data_mode and annual_context):
        return _Check(rid, title, True, "No aplica (consulta no anual).")
    has_pib = "pib" in ctx.out_norm
    has_imacec = (
        "imacec" in ctx.out_norm
        and "no imacec" not in ctx.out_norm
        and "no el imacec" not in ctx.out_norm
    )
    ok = has_pib and not has_imacec
    return _Check(rid, title, ok,
        f"anual={annual_context}, pib={has_pib}, imacec={has_imacec}.")


def _rule_pib_annual_4q(ctx: RuleContext) -> _Check:
    """Regla 10: PIB anual solo válido con 4/4 trimestres publicados."""
    rid = "pib_anual_requiere_4_trimestres"
    title = "PIB anual solo válido con 4/4 trimestres publicados"
    pib_annual_query = ctx.is_data_mode and ctx.requested_year is not None and "pib" in ctx.q_norm
    if not pib_annual_query:
        return _Check(rid, title, True, "No aplica (consulta no es PIB anual).")
    latest_t = str(ctx.latest_available.get("T") or "").strip()
    published_q = _published_quarters_for_year(ctx.requested_year, latest_t) if ctx.requested_year is not None else None
    if published_q is not None:
        if published_q < 4:
            has_fraction = f"{published_q}/4" in ctx.out_norm
            ok = ctx.mentions_unpublished and (has_fraction or "trimestre" in ctx.out_norm)
            return _Check(rid, title, ok,
                f"PIB {ctx.requested_year}: {published_q}/4 publicados, "
                f"no_publicado={ctx.mentions_unpublished}, fraccion={has_fraction}.")
        ok = not ctx.mentions_unpublished
        return _Check(rid, title, ok,
            f"PIB {ctx.requested_year}: 4/4 publicados; no_publicado={ctx.mentions_unpublished}.")
    latest_a_raw = str(ctx.latest_available.get("A") or "").strip()
    latest_a = int(latest_a_raw) if latest_a_raw.isdigit() else None
    if latest_a is None:
        return _Check(rid, title, True,
            "No aplica validación 4/4: contexto trimestral no disponible.")
    if ctx.requested_year is not None and ctx.requested_year > latest_a:
        ok = ctx.mentions_unpublished and ctx.mentions_latest
        return _Check(rid, title, ok,
            f"PIB {ctx.requested_year}: ultimo_anual={latest_a}; "
            f"no_publicado={ctx.mentions_unpublished}, menciona_ultimo={ctx.mentions_latest}.")
    return _Check(rid, title, True,
        f"PIB {ctx.requested_year}: dato anual publicado; no requiere 4/4 explícito.")


def _rule_intro_contextual(ctx: RuleContext) -> _Check:
    """Regla 11: Introducción contextual — valida fecha o dato directo."""
    rid = "introduccion_contextual_fecha_o_dato"
    title = "Introducción contextual: valida fecha cuando aplica y si no entrega el dato directo"
    ok, detail = _validate_intro_context(
        ctx.intro_text,
        is_data_mode=ctx.is_data_mode,
        mentions_unpublished=ctx.mentions_unpublished,
    )
    return _Check(rid, title, ok, detail)


def _rule_intro_no_forbidden_tail(ctx: RuleContext) -> _Check:
    """Regla 12: Introducción no incluye 'se reporta el resultado solicitado'."""
    rid = "introduccion_sin_cola_prohibida"
    title = "Introducción no incluye 'se reporta el resultado solicitado'"
    has_tail = _contains_any(_norm_acceptance(ctx.intro_text), _FORBIDDEN_INTRO_TAIL_TOKENS)
    return _Check(rid, title, not has_tail, f"cola_prohibida={has_tail}.")


def _rule_recommendation_paragraph(ctx: RuleContext) -> _Check:
    """Regla 13: Incluye recomendación como párrafo final."""
    rid = "recomendacion_parrafo_final"
    title = "Incluye recomendación como párrafo final de la respuesta"
    if not ctx.is_data_mode:
        return _Check(rid, title, True, f"No aplica: modo={ctx.mode or 'none'}.")
    ok = _has_recommendation_final_paragraph(ctx.response_text)
    return _Check(rid, title, ok, f"recomendacion_detectada={ok}.")


def _rule_no_interanual_word(ctx: RuleContext) -> _Check:
    """Regla 14: No usar la palabra 'interanual'; usar 'variación anual fue'."""
    rid = "sin_palabra_interanual"
    title = "No usar la palabra 'interanual'; usar 'variación anual fue'"
    return _Check(rid, title, not ctx.has_interanual_word,
        f"contiene_interanual={ctx.has_interanual_word}.")


def _rule_original_value_case(ctx: RuleContext) -> _Check:
    """Regla 15: En casos PIB nominal/per cápita/precios corrientes o valor IMACEC,
    responde con valor original (sin %, sin lenguaje de variación)."""
    rid = "valor_original_pib_casos"
    title = "En casos PIB nominal/per cápita/precios corrientes, responde con valor original"
    if ctx.unavailable_expected:
        return _Check(rid, title, True, "No aplica: período solicitado aún no publicado.")
    if not (ctx.is_data_mode and ctx.original_value_exception):
        return _Check(rid, title, True, "No aplica excepción de valor original.")
    # Combinar intro + respuesta para buscar valor numérico y unidad.
    # La unidad (ej. "promedio 2018=100") suele estar en la introducción.
    combined_text = f"{ctx.intro_text} {ctx.response_text}"
    combined_norm = _norm_acceptance(combined_text)
    has_percent = "%" in ctx.response_text
    has_value_number = bool(re.search(r"\b\d+(?:[\.,]\d+)?\b", combined_text))
    has_value_units = _contains_any(combined_norm, _VALUE_UNIT_TOKENS)
    has_annual_lang = (
        _contains_any(ctx.response_norm, _INTERANUAL_TOKENS)
        or bool(_INTERANUAL_WORD_RE.search(ctx.response_text))
    )
    ok = has_value_number and has_value_units and not has_percent and not has_annual_lang
    return _Check(rid, title, ok,
        f"valor={has_value_number}, unidad={has_value_units}, "
        f"porcentaje={has_percent}, variacion={has_annual_lang}.")


def _rule_contribution_one_decimal(ctx: RuleContext) -> _Check:
    """Regla 16: En contribuciones, usar máximo un decimal."""
    rid = "contribucion_precision_un_decimal"
    title = "En contribuciones, usar máximo un decimal"
    if not (ctx.is_data_mode and ctx.contribution_exception):
        return _Check(rid, title, True, "No aplica a consultas de contribución.")
    contribution_decimals = _extract_contribution_decimals(ctx.response_text)
    bad = [d for d in contribution_decimals if d > 1]
    ok = len(bad) == 0
    return _Check(rid, title, ok,
        f"decimales={contribution_decimals}, invalidos={bad}.")


def _rule_no_subjective_opinion(ctx: RuleContext) -> _Check:
    """Regla 17: No realizar opinión de los datos."""
    rid = "sin_opinion_de_datos"
    title = "No realizar opinión de los datos"
    hits = sorted({t for t in _SUBJECTIVE_TOKENS if t in ctx.core_norm})
    ok = len(hits) == 0
    return _Check(rid, title, ok, f"subjetivos={hits}.")


def _rule_no_cayo_word(ctx: RuleContext) -> _Check:
    """Regla 18: No usar la palabra 'cayó'; usar redacción neutral."""
    rid = "sin_palabra_cayo"
    title = "No usar la palabra 'cayo/cayó'; usar redacción neutral de variación"
    has_cayo = bool(_CAYO_RE.search(ctx.core_text))
    return _Check(rid, title, not has_cayo, f"contiene_cayo={has_cayo}.")


def _rule_minimum_length(ctx: RuleContext) -> _Check:
    """Regla 19: Largo mínimo de respuesta."""
    rid = "largo_respuesta_minimo"
    title = "Largo de respuesta mínimo (evitar respuestas demasiado cortas)"
    min_len = int(os.getenv("RESPONSE_ACCEPTANCE_MIN_LEN", "180"))
    cleaned = _clean_output_for_length(ctx.output)
    ok = len(cleaned) >= min_len
    return _Check(rid, title, ok, f"length={len(cleaned)}, min={min_len}.")


# Registro ordenado: (rule_id, título legible, función evaluadora)
ACCEPTANCE_RULES: List[tuple] = [
    ("dato_unico_variacion_yoy_por_defecto",
     "Entrega un solo dato por defecto como variación anual",
     _rule_single_yoy_default),
    ("solo_yoy_salvo_excepciones",
     "Solo variación del mismo periodo del año anterior salvo excepciones",
     _rule_only_yoy_no_exceptions),
    ("variaciones_un_decimal",
     "Las variaciones deben salir con exactamente un decimal",
     _rule_percent_one_decimal),
    ("formato_respuesta_tipo",
     "Formato de respuesta según tipo (incluye caso no publicado)",
     _rule_response_format_type),
    ("razonamiento_a_validacion_fecha",
     "Razonamiento A: validación de última fecha de publicación",
     _rule_reasoning_a_date),
    ("razonamiento_b_entrega_valores",
     "Razonamiento B: entrega de valores según reglas",
     _rule_reasoning_b_values),
    ("razonamiento_c_referencia_cuadro",
     "Razonamiento C: referencia al cuadro/fuente (BDE)",
     _rule_reasoning_c_source),
    ("razonamiento_d_crecimiento_economia_ultimo_trimestre",
     "Razonamiento D: caso 'Cuánto creció la economía el último trimestre'",
     _rule_reasoning_d_growth),
    ("anual_usar_pib_no_imacec",
     "Si es cifra anual, usar PIB y no IMACEC",
     _rule_annual_pib_not_imacec),
    ("pib_anual_requiere_4_trimestres",
     "PIB anual solo válido con 4/4 trimestres publicados",
     _rule_pib_annual_4q),
    ("introduccion_contextual_fecha_o_dato",
     "Introducción contextual: valida fecha cuando aplica y si no entrega el dato directo",
     _rule_intro_contextual),
    ("introduccion_sin_cola_prohibida",
     "Introducción no incluye 'se reporta el resultado solicitado'",
     _rule_intro_no_forbidden_tail),
    ("recomendacion_parrafo_final",
     "Incluye recomendación como párrafo final de la respuesta",
     _rule_recommendation_paragraph),
    ("sin_palabra_interanual",
     "No usar la palabra 'interanual'; usar 'variación anual fue'",
     _rule_no_interanual_word),
    ("valor_original_pib_casos",
     "En casos PIB nominal/per cápita/precios corrientes, responde con valor original",
     _rule_original_value_case),
    ("contribucion_precision_un_decimal",
     "En contribuciones, usar máximo un decimal",
     _rule_contribution_one_decimal),
    ("sin_opinion_de_datos",
     "No realizar opinión de los datos",
     _rule_no_subjective_opinion),
    ("sin_palabra_cayo",
     "No usar la palabra 'cayo/cayó'; usar redacción neutral de variación",
     _rule_no_cayo_word),
    ("largo_respuesta_minimo",
     "Largo de respuesta mínimo (evitar respuestas demasiado cortas)",
     _rule_minimum_length),
]


def evaluate_acceptance_criteria(
    *,
    question: str,
    output: str,
    generation_logic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evalúa las 19 reglas de aceptación contra la salida generada.

    Cada regla se ejecuta via su función _rule_* registrada en ACCEPTANCE_RULES.
    """
    ctx = _build_rule_context(question, output, generation_logic)
    checks = [rule_fn(ctx) for _rid, _title, rule_fn in ACCEPTANCE_RULES]

    as_dict = [c.as_dict() for c in checks]
    ok_count = sum(1 for item in as_dict if item["ok"])
    no_ok_count = len(as_dict) - ok_count
    overall_ok = no_ok_count == 0
    return {
        "overall_ok": overall_ok,
        "overall_status": "ok" if overall_ok else "no_ok",
        "summary": {
            "ok_count": ok_count,
            "no_ok_count": no_ok_count,
            "total": len(as_dict),
        },
        "checks": as_dict,
    }


def normalize_acceptance_checks(acceptance: Dict[str, Any]) -> Tuple[List[Dict[str, str]], int, int, str]:
    checks = acceptance.get("checks") if isinstance(acceptance.get("checks"), list) else []
    normalized: List[Dict[str, str]] = []
    ok_count = 0

    for item in checks:
        if not isinstance(item, dict):
            continue
        is_ok = bool(item.get("ok"))
        if is_ok:
            ok_count += 1
        title = str(item.get("title") or item.get("id") or "Criterio").strip()
        detail = str(item.get("detail") or "").strip()
        normalized.append({
            "title": title,
            "status": "Ok" if is_ok else "No Ok",
            "detail": detail,
        })

    total = len(normalized)
    final_status = "Ok" if (total > 0 and ok_count == total) else "No Ok"
    return normalized, ok_count, total, final_status


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_detail_log_path(file_name: Optional[str] = None) -> Path:
    logs_dir = _project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = (
        str(file_name or "").strip()
        or os.getenv("GRAPH_DETAIL_LOG_FILE", "").strip()
        or _DEFAULT_TRACE_FILE
    )
    return logs_dir / resolved_name


def _detail_log_enabled() -> bool:
    return os.getenv("GRAPH_DETAIL_LOG_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _session_id_from_state(state: AgentState) -> str:
    context = state.get("context")
    if isinstance(context, dict):
        sid = str(context.get("session_id") or "").strip()
        if sid:
            return sid
    return str(state.get("session_id") or "").strip() or "unknown"


def _escape_table_cell(value: str) -> str:
    cell = str(value or "").replace("\n", " ").strip()
    return cell.replace("|", "\\|")


def _render_acceptance_table_rows(checks: List[Dict[str, str]]) -> List[str]:
    rows: List[str] = [
        "| N | Criterio | Estado | Detalle |",
        "| --- | --- | --- | --- |",
    ]

    if not checks:
        rows.append("| 1 | Criterio 1 | No Ok | - |")
        return rows

    for idx, item in enumerate(checks, start=1):
        title = _escape_table_cell(str(item.get("title") or f"Criterio {idx}"))
        status = _escape_table_cell(str(item.get("status") or "No Ok"))
        detail = _escape_table_cell(str(item.get("detail") or "-"))
        rows.append(f"| {idx} | {title} | {status} | {detail} |")
    return rows


def _render_run_detail_block(
    *,
    session_id: str,
    question: str,
    classification_type: str,
    data_store_json: str,
    final_response: str,
    criteria_table_lines: List[str],
    criteria_final: str,
) -> str:
    lines: List[str] = [
        "-" * 100,
        f"Session: {session_id}",
        f"Timestamp: {_dt.datetime.now(_dt.timezone.utc).isoformat()}",
        f"Pregunta: {question}",
        f"Tipo de clasificacion: {classification_type}",
        f"Json_data_store: {data_store_json}",
        f"Respuesta final: {final_response}",
        "Criterios de aceptacion:",
    ]

    lines.extend(criteria_table_lines)

    lines.append(f"Criterio final: {criteria_final}")
    lines.append("-" * 100)
    return "\n".join(lines)


def _write_response_detail_log(
    *,
    state: AgentState,
    question: str,
    output: str,
    acceptance: Dict[str, Any],
) -> str:
    if not _detail_log_enabled():
        return ""

    checks, ok_count, total, final_status = normalize_acceptance_checks(acceptance)
    criteria_table_lines = _render_acceptance_table_rows(checks)

    merged_state: Dict[str, Any] = dict(state)
    merged_state["output"] = output
    classification_type = resolve_classification_type(merged_state)
    data_store_json = serialize_data_store_lookup(merged_state)
    criteria_final = f"{ok_count}/{total} criterios OK. {final_status}"

    block = _render_run_detail_block(
        session_id=_session_id_from_state(state),
        question=str(question or merged_state.get("question") or ""),
        classification_type=classification_type,
        data_store_json=data_store_json,
        final_response=str(output or ""),
        criteria_table_lines=criteria_table_lines,
        criteria_final=criteria_final,
    )

    try:
        path = _resolve_detail_log_path()
        with path.open("a", encoding="utf-8") as fp:
            fp.write(block)
            fp.write("\n\n")
        return str(path)
    except Exception:
        return ""


def _generate_data_output(response_payload: Dict[str, Any], *, writer=None) -> str:
    payload = response_payload.get("payload")
    if not isinstance(payload, dict):
        return "No pude preparar el payload de datos para generar la respuesta."

    chunks: List[str] = []
    for chunk in stream_data_response(payload):
        text = str(chunk or "")
        if not text:
            continue
        chunks.append(text)
        _emit_stream_chunk(text, writer)
    return "".join(chunks)


def _generate_llm_output(response_payload: Dict[str, Any], *, adapter, writer=None) -> str:
    llm_state: AgentState = {
        "question": str(response_payload.get("question") or ""),
        "conversation_history": list(response_payload.get("history") or []),
        "intent_info": response_payload.get("intent_info") or {},
    }
    result = run_llm_stream(llm_state, adapter, writer=writer)
    output = str(result.get("output") or "")

    if bool(response_payload.get("append_methodology_footer")):
        footer = _build_methodology_footer(adapter, max_sources=2)
        if footer and not _is_generation_error_output(output) and not _has_existing_methodology_footer(output, footer):
            output = f"{output}{footer}"
            _emit_stream_chunk(footer, writer)

    return output


def _resolve_generated_output(
    state: AgentState,
    *,
    rag_llm_adapter=None,
    fallback_llm_adapter=None,
    writer=None,
) -> Tuple[str, Dict[str, Any]]:
    current_output = str(state.get("output") or "")
    if current_output:
        return current_output, {
            "mode": "existing_output",
            "generator": "state.output",
            "reason": "Output already present before response node.",
        }

    response_payload = state.get("response_payload")
    if not isinstance(response_payload, dict):
        return current_output, {
            "mode": "none",
            "generator": "none",
            "reason": "Missing response_payload in state.",
        }

    mode = str(response_payload.get("mode") or "").strip().lower()
    payload_summary = summarize_response_payload(response_payload)

    if mode == "prebuilt":
        text = str(response_payload.get("text") or "")
        if text:
            _emit_stream_chunk(text, writer)
        return text, {
            "mode": "prebuilt",
            "generator": "response_payload.text",
            "input": payload_summary,
        }
    if mode == "data":
        generated = _generate_data_output(response_payload, writer=writer)
        return generated, {
            "mode": "data",
            "generator": "stream_data_response",
            "input": payload_summary,
        }
    if mode == "rag":
        generated = _generate_llm_output(response_payload, adapter=rag_llm_adapter, writer=writer)
        return generated, {
            "mode": "rag",
            "generator": "run_llm_stream",
            "input": payload_summary,
        }
    if mode == "fallback":
        generated = _generate_llm_output(response_payload, adapter=fallback_llm_adapter, writer=writer)
        return generated, {
            "mode": "fallback",
            "generator": "run_llm_stream",
            "input": payload_summary,
        }

    return current_output, {
        "mode": mode or "unknown",
        "generator": "unknown",
        "reason": "Unsupported response_payload mode.",
        "input": payload_summary,
    }


def _build_validation_context(
    *,
    question: str,
    output: str,
    response_state: Dict[str, Any],
    generation_logic: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a single immutable-like context for response validations."""
    return {
        "question": str(question or ""),
        "output": str(output or ""),
        "response_state": dict(response_state or {}),
        "generation_logic": dict(generation_logic or {}),
    }


def _validate_contract_rules(validation_ctx: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_response_contract(
        str(validation_ctx.get("output") or ""),
        dict(validation_ctx.get("response_state") or {}),
    )


def _validate_acceptance_rules(validation_ctx: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_acceptance_criteria(
        question=str(validation_ctx.get("question") or ""),
        output=str(validation_ctx.get("output") or ""),
        generation_logic=dict(validation_ctx.get("generation_logic") or {}),
    )


def _aggregate_validation_report(validation_ctx: Dict[str, Any]) -> Dict[str, Any]:
    contract_validation = _validate_contract_rules(validation_ctx)
    acceptance_validation = _validate_acceptance_rules(validation_ctx)
    return {
        "contract": contract_validation,
        "acceptance": acceptance_validation,
        "ok": bool(contract_validation.get("ok")) and bool(acceptance_validation.get("overall_ok", True)),
    }


def make_response_node(*, rag_llm_adapter=None, fallback_llm_adapter=None):
    def response_node(state: AgentState, *, writer=None) -> AgentState:
        question = str(state.get("question") or "")
        output, generation_logic = _resolve_generated_output(
            state,
            rag_llm_adapter=rag_llm_adapter,
            fallback_llm_adapter=fallback_llm_adapter,
            writer=writer,
        )

        response_state = _build_response_state(state)
        composed_output, sections = compose_modular_response(output, response_state, question=question)
        composed_output, sections = _postprocess_response_sections(sections, generation_logic)

        validation_ctx = _build_validation_context(
            question=question,
            output=composed_output,
            response_state=response_state,
            generation_logic=generation_logic,
        )
        validation_report = _aggregate_validation_report(validation_ctx)
        validation = validation_report.get("contract") or {}
        acceptance_checks = validation_report.get("acceptance") or {}

        response_logic = {
            "generation": generation_logic,
            "postprocess": {
                "type": "deterministic_redaction_guard",
                "llm_output_preserved": True,
                "purpose": "Reducir alucinacion y desalineacion con criterios, priorizando ajustes de redaccion.",
                "note": (
                    "La respuesta base la sigue generando el LLM; esta capa determinista "
                    "disminuye la probabilidad de respuestas erroneas y privilegia mejorar "
                    "redaccion antes que interpretar o generar texto no deseado."
                ),
                "scope": [
                    "intro_contextual",
                    "intro_frase_permitida",
                    "neutralizacion_termino_cayo",
                    "normalizacion_variacion_anual",
                    "recomendacion_final_obligatoria",
                    "referencia_bde_en_modo_data",
                ],
            },
            "contract": {
                "version": response_state.get("version"),
                "route_decision": response_state.get("route_decision"),
                "sections": response_state.get("sections"),
                "temperature": response_state.get("temperature"),
                "semantic": response_state.get("semantic"),
            },
            "validation": {
                "ok": bool(validation.get("ok")),
                "errors": list(validation.get("errors") or []),
                "metrics": dict(validation.get("metrics") or {}),
            },
            "acceptance": acceptance_checks,
        }

        detail_log_path = _write_response_detail_log(
            state=state,
            question=question,
            output=composed_output,
            acceptance=acceptance_checks,
        )
        if detail_log_path:
            response_logic["detail_log_path"] = detail_log_path

        return {
            "output": composed_output,
            "response": response_state,
            "response_structure": response_state.get("sections"),
            "response_sections": sections,
            "response_logic": response_logic,
            "acceptance_checks": acceptance_checks,
            "response_temperature": response_state.get("temperature"),
            "response_ok": bool(validation.get("ok")),
            "response_errors": list(validation.get("errors") or []),
            "response_metrics": dict(validation.get("metrics") or {}),
            "response_classification_type": resolve_classification_type(
                {
                    "route_decision": state.get("route_decision"),
                    "response_payload": state.get("response_payload"),
                }
            ),
            "response_detail_log_path": detail_log_path,
        }

    return response_node


__all__ = [
    "make_response_node",
    "_build_validation_context",
    "_validate_contract_rules",
    "_validate_acceptance_rules",
    "_aggregate_validation_report",
    "normalize_acceptance_checks",
    "resolve_classification_type",
    "serialize_data_store_lookup",
    "summarize_response_payload",
]
