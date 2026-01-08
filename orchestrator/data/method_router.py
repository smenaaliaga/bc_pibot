"""Methodological routing utilities extracted from orchestrator_old.

Provides a minimal wrapper for methodological queries if you want to reuse
legacy streaming. Currently unused by default; kept for symmetry.
"""
from __future__ import annotations

from typing import Iterable, Any
import logging

logger = logging.getLogger(__name__)

try:
    from orchestrator_old import stream as legacy_stream  # type: ignore
    LEGACY_AVAILABLE = True
except Exception:
    LEGACY_AVAILABLE = False


def can_handle_method(classification: Any) -> bool:
    try:
        return getattr(classification, "intent", "").lower() in ('methodology', 'definition')
    except Exception:
        return False


def stream_method_flow(classification: Any, question: str, history_text: str) -> Iterable[str]:
    if not LEGACY_AVAILABLE:
        yield "El flujo metodológico legacy no está disponible en este entorno."
        return
    try:
        yield from legacy_stream(question, history=[])  # type: ignore
    except Exception as exc:
        logger.exception("Legacy method flow failed: %s", exc)
        yield "No fue posible completar la ruta metodológica con el flujo legacy."

