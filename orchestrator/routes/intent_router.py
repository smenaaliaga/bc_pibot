"""
Este módulo maneja:
1. Chart follow-ups (requiere contexto de gráficos previos)
2. Casos edge no clasificados por JointBERT

Lo demás (detección de series, entidades, períodos) lo maneja JointBERT + data_node.
"""

import json
import logging
import re
import unicodedata
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

GENTLE_NUDGE_MESSAGE = (
    "Mi base de conocimiento no está diseñada para responder respecto a ese tópico. "
    "Puedo responder preguntas respecto al PIB e IMACEC."
)

NAME_PATTERNS = [
    r"\bmi nombre es\s+(?P<name>[^,.!?]+)",
    r"\byo soy\s+(?P<name>[^,.!?]+)",
    r"\bme llaman\s+(?P<name>[^,.!?]+)",
    r"\bme llamo\s+(?P<name>[^,.!?]+)",
    r"\bme dicen\s+(?P<name>[^,.!?]+)",
]


def _wrap(text: str) -> Iterable[str]:
    return [text]


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFD", value)
    stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return stripped.lower()


def _coerce_macro(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.isdigit():
            return 1 if int(cleaned) != 0 else 0
    return None


def _normalize_title_case(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip())
    return cleaned.lower().title()


def _extract_user_name(question: str) -> Optional[str]:
    if not question:
        return None
    lowered = _normalize_text(question)
    for pattern in NAME_PATTERNS:
        match = re.search(pattern, lowered, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group("name") if match.groupdict().get("name") else match.group(1)
        if not raw:
            continue
        name = _normalize_title_case(raw)
        if name:
            return name
    return None


def _append_user_name_fact(memory: Optional[Any], session_id: Optional[str], name: str) -> bool:
    if not memory or not session_id or not name:
        return False
    if not hasattr(memory, "get_facts") or not hasattr(memory, "set_facts"):
        return False
    try:
        facts = memory.get_facts(session_id) or {}
    except Exception:
        facts = {}
    existing = facts.get("user_name")
    names: list[str] = []
    if isinstance(existing, list):
        names = [str(item) for item in existing if str(item).strip()]
    elif isinstance(existing, str) and existing.strip():
        try:
            parsed = json.loads(existing)
            if isinstance(parsed, list):
                names = [str(item) for item in parsed if str(item).strip()]
            else:
                names = [existing]
        except Exception:
            names = [existing]
    if name:
        names.append(name)
    try:
        memory.set_facts(session_id, {"user_name": names})
        return True
    except Exception:
        logger.debug("Failed to persist user_name fact", exc_info=True)
    return False


def _extract_chart_domain_hint(question: str) -> Optional[str]:
    """Extrae hints de dominio para gráficos (imacec, pib)."""
    question_lower = _normalize_text(question)
    if "imacec" in question_lower:
        return "IMACEC"
    if "pib" in question_lower or "producto" in question_lower:
        return "PIB"
    return None


def _last_chart_turn_metadata(memory: Optional[Any], session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Obtiene metadata del último gráfico generado."""
    if not memory or not session_id:
        return None
    try:
        if not hasattr(memory, "get_turn_metadata"):
            return None
        last = memory.get_turn_metadata(session_id, turn=-1)
        if last and last.get("chart_domain"):
            return last
    except Exception:
        logger.debug("Could not retrieve last chart metadata", exc_info=True)
    return None


def _looks_like_chart_command(question: str) -> bool:
    """Detecta comandos de gráficos (grafica, visualiza, chart)."""
    question_lower = _normalize_text(question).strip()
    chart_keywords = [
        "grafica", "graficar", "graficalo", "grafiquemos", "otro grafico",
        "muestra", "muestrame", "mostrar",
        "visualiza", "chart", "plot", "dibuja",
        "genera un grafico", "generar un grafico",
    ]
    return any(keyword in question_lower for keyword in chart_keywords)


def _handle_chart_followup(
    question: str,
    domain_upper: str,
    memory: Optional[Any],
    session_id: Optional[str],
) -> Optional[Iterable[str]]:
    """
    Maneja follow-ups de gráficos usando contexto de la conversación.
    
    Ejemplos:
    - "grafícalo" → usa domain del último chart
    - "muéstrame eso en gráfico" → usa domain + series del contexto
    """
    if not _looks_like_chart_command(question):
        return None

    last_meta = _last_chart_turn_metadata(memory, session_id)
    last_domain = ((last_meta or {}).get("chart_domain") or "").strip().upper()

    chart_hint: Optional[str] = None
    explicit_hint = _extract_chart_domain_hint(question)

    if explicit_hint:
        candidate = explicit_hint.strip().upper()
        if not last_domain or last_domain != candidate:
            logger.debug(
                "[INTENT_ROUTER] Chart follow-up rejected (last=%s, requested=%s)",
                last_domain or "",
                candidate,
            )
            return None
        chart_hint = candidate
    else:
        if last_domain:
            chart_hint = last_domain
        else:
            normalized_domain = (domain_upper or "").strip().upper()
            if normalized_domain:
                chart_hint = normalized_domain

    if not chart_hint:
        return None

    # Retornar metadata para que el graph node genere el gráfico
    class _IterWithMetadata:
        def __init__(self, iterable: Iterable[str], metadata: Optional[Dict[str, Any]] = None):
            self._iter = iter(iterable)
            self.metadata = metadata or {}

        def __iter__(self):
            return self._iter

    # Mensaje + metadata para chart
    chart_marker = f"[CHART:{chart_hint}]"
    msg = f"Generando gráfico de {chart_hint}...\n{chart_marker}"
    
    metadata = {
        "chart_domain": chart_hint,
        "chart_request": True,
    }

    return _IterWithMetadata([msg], metadata)


def route_intents(
    classification: Any,
    question: str,
    history_text: str,
    intent_classifier: Optional[Any] = None,
    memory: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> Optional[Iterable[str]]:
    """
    Router simplificado que solo maneja casos específicos que requieren contexto.
    
    Retorna:
        - Iterable[str]: Respuesta directa (short-circuit)
        - None: Continuar con el flujo normal (data_node o rag_node)
    
    Casos manejados:
    1. Chart follow-ups (requiere contexto de gráficos previos)
    2. Queries metodológicas (delega a RAG)
    
    TODO lo demás lo maneja JointBERT + data_node.
    """
    try:
        macro = _coerce_macro(getattr(classification, "macro", None))
        # Extraer indicador desde normalized
        indicator = None
        normalized = getattr(classification, "normalized", None)
        if normalized and isinstance(normalized, dict):
            indicator_data = normalized.get('indicator', {})
            if isinstance(indicator_data, dict):
                indicator = indicator_data.get('standard_name') or indicator_data.get('normalized')
        
        intent = (getattr(classification, "intent", "") or "").lower()
    except Exception:
        macro = None
        indicator = None
        intent = ""

    # 0. Macro=0 → no económico: guardar facts si aplica y nudgear
    if macro == 0:
        name = _extract_user_name(question)
        if name:
            _append_user_name_fact(memory, session_id, name)
        return _wrap(GENTLE_NUDGE_MESSAGE)

    # 1. Chart follow-ups (usa contexto)
    chart_iter = _handle_chart_followup(question, str(indicator or "").upper(), memory, session_id)
    if chart_iter:
        logger.info("[INTENT_ROUTER] Manejando chart follow-up")
        return chart_iter

    # 2. Metodología → delega a RAG
    if intent in ('methodology', 'definition', 'greeting'):
        logger.debug("[INTENT_ROUTER] Consulta metodológica, delegando a RAG")
        return None

    # 3. Si JointBERT detectó entidades → delega a data_node
    if normalized and isinstance(normalized, dict) and intent in ('value', 'data', 'last', 'table'):
        logger.debug("[INTENT_ROUTER] Consulta de datos, delegando a data_node")
        return None

    # 4. Todo lo demás → continuar con flujo normal
    logger.debug("[INTENT_ROUTER] Sin casos específicos, continuando flujo normal")
    return None


__all__ = [
    "route_intents",
]
