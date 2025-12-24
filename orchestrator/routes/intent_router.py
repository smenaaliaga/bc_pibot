"""
Este módulo maneja:
1. Chart follow-ups (requiere contexto de gráficos previos)
2. Casos edge no clasificados por JointBERT

Lo demás (detección de series, entidades, períodos) lo maneja JointBERT + data_node.
"""

import logging
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


def _extract_chart_domain_hint(question: str) -> Optional[str]:
    """Extrae hints de dominio para gráficos (imacec, pib)."""
    question_lower = question.lower()
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
    question_lower = question.lower().strip()
    chart_keywords = [
        "grafica", "gráfica", "grafíca",
        "muestra", "muéstrame", "mostrar",
        "visualiza", "chart", "plot", "dibuja",
        "genera un grafico", "genera un gráfico"
    ]
    return any(keyword in question_lower for keyword in chart_keywords)


def _handle_chart_followup(
    question: str,
    domain_upper: str,
    memory: Optional[Any],
    session_id: Optional[str],
    facts: Optional[Dict[str, str]],
) -> Optional[Iterable[str]]:
    """
    Maneja follow-ups de gráficos usando contexto de la conversación.
    
    Ejemplos:
    - "grafícalo" → usa domain del último chart
    - "muéstrame eso en gráfico" → usa domain + series del contexto
    """
    if not _looks_like_chart_command(question):
        return None

    # Intentar obtener domain de la pregunta
    chart_hint = _extract_chart_domain_hint(question)
    
    # Si no hay hint en la pregunta, usar el último gráfico
    if not chart_hint:
        last_meta = _last_chart_turn_metadata(memory, session_id)
        if last_meta:
            chart_hint = last_meta.get("chart_domain")

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
    
    # Incluir series_id si está en facts
    if facts and "series_id" in facts:
        metadata["series_id"] = facts["series_id"]
    
    return _IterWithMetadata([msg], metadata)


def route_intents(
    classification: Any,
    question: str,
    history_text: str,
    intent_classifier: Optional[Any] = None,
    memory: Optional[Any] = None,
    session_id: Optional[str] = None,
    facts: Optional[Dict[str, str]] = None,
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
        # Extraer indicador desde normalized
        indicator = None
        normalized = getattr(classification, "normalized", None)
        if normalized and isinstance(normalized, dict):
            indicator_data = normalized.get('indicator', {})
            if isinstance(indicator_data, dict):
                indicator = indicator_data.get('standard_name') or indicator_data.get('normalized')
        
        intent = (getattr(classification, "intent", "") or "").lower()
    except Exception:
        indicator = None
        intent = ""

    # 1. Chart follow-ups (usa contexto)
    chart_iter = _handle_chart_followup(question, str(indicator or "").upper(), memory, session_id, facts)
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
