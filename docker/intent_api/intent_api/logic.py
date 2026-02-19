"""Heuristics for macro/intent/context classification."""
from __future__ import annotations

from typing import Dict


_MACRO_TERMS = ("pib", "imacec")
_METHOD_TERMS = ("metodolog", "metodología", "metodologia", "definicion", "definición")
_VALUE_TERMS = (
    "valor",
    "ultimo",
    "último",
    "cuanto",
    "cuánto",
    "cifra",
    "nivel",
    "tasa",
    "variacion",
    "variación",
)
_FOLLOWUP_TERMS = ("eso", "lo mismo", "esa", "ese", "anterior", "otra vez", "siguiente", "igual")


def classify_intent(text: str) -> Dict[str, str | int]:
    normalized = (text or "").strip().lower()
    macro = 1 if any(term in normalized for term in _MACRO_TERMS) else 0

    intent = "other"
    if any(term in normalized for term in _METHOD_TERMS):
        intent = "method"
    elif macro == 1:
        if any(term in normalized for term in _VALUE_TERMS):
            intent = "value"
        else:
            intent = "value"

    context = "followup" if any(term in normalized for term in _FOLLOWUP_TERMS) else "standalone"

    return {
        "macro": macro,
        "intent": intent,
        "context": context,
    }
