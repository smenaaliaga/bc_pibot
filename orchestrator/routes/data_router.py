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
    context_payload: Optional[Dict[str, Any]] = None,
) -> Iterable[str]:
    """
    Delegado del flujo de datos simplificado.

    - Extrae `session_id` del payload para permitir a `flow_data` acceder a Redis.
    - Delegar completamente a `flow_data.stream_data_flow` (sin question/history).
    """
    try:
        from orchestrator.data import flow_data

        session_id = None
        if isinstance(context_payload, dict):
            session_id = context_payload.get("session_id")

        for chunk in flow_data.stream_data_flow(
            classification,
            session_id=session_id,
        ):
            if chunk:
                yield chunk
        return
    except Exception as e:
        logger.error(f"[DATA_DELEGATE] fallo flujo DATA: {e}")
        # Fallback mínimo si el flujo falla
        indicator = "OTRO"
        normalized = getattr(classification, "normalized", None)
        if normalized and isinstance(normalized, dict):
            indicator_data = normalized.get('indicator', {})
            if isinstance(indicator_data, dict):
                ind = indicator_data.get('standard_name') or indicator_data.get('normalized')
                if ind:
                    indicator = str(ind).upper()
        yield _simple_methodological_intro(indicator)
        yield DATA_BANNER
        yield from _default_data_reply(indicator, None)
