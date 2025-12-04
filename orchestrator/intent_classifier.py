"""Intent classifier interface and pluggable implementations.

Default is a simple regex stub; if JOINT_INTENT_ENABLED=true, use the joint
BERT placeholder (BIO tagging) in `joint_intent_classifier.py`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re
import os

try:
    from .joint_intent_classifier import JointIntentClassifier  # type: ignore
except Exception:
    JointIntentClassifier = None  # type: ignore


class IntentClassifierProtocol:
    """Protocol for intent classifiers."""

    def predict(self, text: str, history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, float, List[Dict[str, Any]], Dict[str, Any]]:
        raise NotImplementedError


class SimpleIntentClassifier(IntentClassifierProtocol):
    """Heuristic stub; replace with your joint BERT model."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, text: str, history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, float, List[Dict[str, Any]], Dict[str, Any]]:
        t = (text or "").lower()
        if "nombre" in t:
            intent = "ask_name"
        elif any(k in t for k in ["imacec", "actividad economica"]):
            intent = "ask_imacec"
        elif any(k in t for k in ["inflacion", "ipc"]):
            intent = "ask_inflation"
        else:
            intent = "unknown"
        # Fake BIO spans: full text as a single span with label O
        spans = [{"text": text, "label": "O", "start": 0, "end": len(text)}]
        entities: Dict[str, Any] = {}
        score = 0.51 if intent != "unknown" else 0.2
        return intent, score, spans, entities


def get_intent_classifier() -> IntentClassifierProtocol:
    """Factory to choose between joint BIO model and simple stub."""
    use_joint = os.getenv("JOINT_INTENT_ENABLED", "false").lower() in {"1", "true", "yes"}
    if use_joint and JointIntentClassifier is not None:
        try:
            return JointIntentClassifier()
        except Exception:
            pass
    return SimpleIntentClassifier()
