"""Classification node for the PIBot LangGraph."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from orchestrator.classifier.classifier_agent import ClassificationResult

from ..state import AgentState

logger = logging.getLogger(__name__)


def _build_history_text_from_turns(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    try:
        return "\n".join(f"{turn.get('role', '')}: {turn.get('content', '')}" for turn in history)
    except Exception:
        logger.debug("[GRAPH] Unable to build history_text", exc_info=True)
        return ""


def _record_intent_event(
    intent_store: Any,
    state: AgentState,
    classification: Optional[ClassificationResult],
    intent_info: Optional[Dict[str, Any]],
) -> None:
    if not intent_store or not hasattr(intent_store, "record"):
        return
    session_id = state.get("session_id")
    turn_id = state.get("user_turn_id")
    if not session_id or turn_id is None:
        return
    payload = intent_info if isinstance(intent_info, dict) else {}
    raw_intent = payload.get("intent")
    if not raw_intent and classification:
        raw_intent = getattr(classification, "intent", None)
    intent = str(raw_intent).strip() if raw_intent else ""
    if not intent:
        return
    score_raw = payload.get("score")
    if score_raw is None and classification and getattr(classification, "confidence", None) is not None:
        score_raw = classification.confidence
    try:
        score = float(score_raw) if score_raw is not None else 1.0
    except (TypeError, ValueError):
        score = 1.0
    spans = payload.get("spans") or []
    base_entities = payload.get("entities") or {}
    entities = dict(base_entities) if isinstance(base_entities, dict) else {}
    extra_jointbert: Dict[str, Any] = {}
    if classification:
        if getattr(classification, "intent", None):
            extra_jointbert["raw_intent"] = classification.intent
        if getattr(classification, "confidence", None) is not None:
            extra_jointbert["confidence"] = classification.confidence
        if getattr(classification, "entities", None):
            extra_jointbert["entities"] = classification.entities
        if getattr(classification, "normalized", None):
            extra_jointbert["normalized"] = classification.normalized
    if extra_jointbert:
        entities = dict(entities)
        entities["jointbert"] = extra_jointbert
    try:
        intent_store.record(
            session_id,
            intent,
            score,
            spans=spans,
            entities=entities,
            turn_id=int(turn_id),
        )
    except Exception:
        logger.debug("[GRAPH] Unable to persist intent event", exc_info=True)


def make_classify_node(
    intent_store: Any,
    classify_fn: Callable[[str, Optional[List[Dict[str, str]]]], Tuple[ClassificationResult, str]],
    intent_info_builder: Callable[[Optional[ClassificationResult]], Optional[Dict[str, Any]]],
):
    def classify_node(state: AgentState) -> AgentState:
        question = state.get("question", "")
        history = state.get("conversation_history")
        try:
            classification, history_text = classify_fn(question, history)
        except Exception:
            logger.exception("[GRAPH] classify_question_with_history failed")
            classification = None
            history_text = ""
        if not isinstance(history_text, str):
            history_text = ""
        if not history_text:
            history_text = _build_history_text_from_turns(history)
        try:
            intent_info = intent_info_builder(classification)
        except Exception:
            logger.exception("[GRAPH] build_intent_info failed")
            intent_info = None
        _record_intent_event(intent_store, state, classification, intent_info)
        return {
            "classification": classification,
            "history_text": history_text,
            "intent_info": intent_info,
        }

    return classify_node


__all__ = ["make_classify_node"]
