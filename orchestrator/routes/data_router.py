"""Self-contained data flow router (adapted from original orchestrator)."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

DATA_BANNER = "\n---\nProcesando los datos solicitados, esto puede tomar unos segundos...\n"


def _wrap(text: str) -> Iterable[str]:
    yield text


def _default_data_reply(domain: str, year: Optional[int] = None) -> Iterable[str]:
    msg = (
        f"No tengo habilitado el flujo de datos completo para {domain}. "
        "Puedo explicarte el indicador y cómo obtener los datos desde la serie oficial."
    )
    return _wrap(msg)


def _simple_methodological_intro(domain: str) -> str:
    return (
        f"Estás consultando por {domain}. En este modo puedo darte la definición y la metodología, "
        "pero no generaré cifras ni tablas. "
    )


def _extract_year(question: str) -> Optional[int]:
    try:
        m = re.search(r"\b(19|20)\d{2}\b", question)
        if m:
            return int(m.group(0))
    except Exception:
        return None
    return None


def can_handle_data(classification: Any) -> bool:
    try:
        return getattr(classification, "intent", "").lower() in ('value', 'data', 'last', 'table')
    except Exception:
        return False


def stream_data_flow(
    classification: Any,
    question: str,
    history_text: str,
    *,
    indicator_context: Optional[Dict[str, str]] = None,
) -> Iterable[str]:
    """
    Intenta usar el flujo DATA completo (fase 1 + fetch + tabla).

    Si falla, usa el placeholder metodológico+banner.
    """
    # Extraer indicador desde normalized
    indicator = "OTRO"
    normalized = getattr(classification, "normalized", None)
    if normalized and isinstance(normalized, dict):
        indicator_data = normalized.get('indicator', {})
        if isinstance(indicator_data, dict):
            ind = indicator_data.get('standard_name') or indicator_data.get('normalized')
            if ind:
                indicator = str(ind).upper()
    
    domain = indicator
    year = _extract_year(question)

    try:
        from orchestrator.data import flow_data

        for chunk in flow_data.stream_data_flow(
            classification,
            question,
            history_text,
            indicator_context=indicator_context,
        ):
            if chunk:
                yield chunk
        return
    except Exception as e:
        logger.error(f"[DATA_DELEGATE] fallo flujo DATA: {e}")

    yield _simple_methodological_intro(domain)
    yield DATA_BANNER
    yield from _default_data_reply(domain, year)
