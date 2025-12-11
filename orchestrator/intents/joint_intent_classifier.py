"""Stub for a joint BERT intent+slot classifier using BIO tags.

This is a placeholder to integrate the real model once it is trained.
It exposes the same predict API as IntentClassifierProtocol and returns:
    intent: str
    score: float
    spans: list of dicts with text/label/start/end
    entities: dict with normalized slots (indicator, frequency, sector, etc.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re

from .intent_classifier import IntentClassifierProtocol


# Minimal vocabularies to mock BIO tagging until the real model is wired.
VOCABS = {
    "indicator": ["pib", "imacec", "crecimiento económico", "pib regional"],
    "frequency": ["mensual", "trimestral", "anual"],
    "sector": [
        "minero",
        "minería",
        "no minero",
        "comercio",
        "construcción",
        "transporte",
        "servicios financieros",
        "pesca",
        "industria manufacturera",
    ],
    "activity": [
        "producción de bienes",
        "industria",
        "manufactura",
        "servicios",
        "comercio",
    ],
    "seasonality": ["desestacionalizado", "no desestacionalizado"],
    "unit": ["índices", "niveles", "deflactor", "serie original"],
    "calculation": [
        "variación mensual",
        "variación anual",
        "variación respecto al periodo anterior",
    ],
    "region": ["metropolitana", "valparaíso", "biobío", "los lagos", "antofagasta"],
    "period": ["último", "trimestre", "2024", "2025"],
    "rank": ["aportó", "restó", "impulsó"],
}


def _find_spans(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    entities: Dict[str, Any] = {}
    lower = text.lower()
    for slot, vocab in VOCABS.items():
        for term in vocab:
            m = re.search(re.escape(term), lower)
            if m:
                spans.append(
                    {
                        "text": text[m.start() : m.end()],
                        "label": f"B-{slot}",
                        "start": m.start(),
                        "end": m.end(),
                    }
                )
                entities.setdefault(slot, text[m.start() : m.end()])
                break
    return spans, entities


class JointIntentClassifier(IntentClassifierProtocol):
    """Placeholder for the real joint BERT model with BIO tagging."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, float, List[Dict[str, Any]], Dict[str, Any]]:
        spans, entities = _find_spans(text or "")
        # Heuristic intent until the real model is loaded
        intent = "unknown"
        if entities.get("indicator") in {"pib", "pib regional"}:
            intent = "ask_pib"
        elif entities.get("indicator") == "imacec":
            intent = "ask_imacec"
        elif entities.get("sector"):
            intent = "ask_sector"
        score = 0.8 if intent != "unknown" else 0.3
        return intent, score, spans, entities


__all__ = ["JointIntentClassifier"]
