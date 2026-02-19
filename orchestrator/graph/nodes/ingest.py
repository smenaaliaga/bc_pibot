"""Ingest and routing nodes for the PIBot LangGraph."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import uuid

from ..state import (
    AgentState,
    _clone_entities,
    _ensure_entity_slot,
    _ensure_list,
    _ensure_text,
    _merge_entity_fields,
)
from ..session import extract_latest_entity_from_history, load_previous_agent_state

logger = logging.getLogger(__name__)


def make_ingest_node(memory_adapter: Any):
    def ingest_node(state: AgentState) -> AgentState:
        question = _ensure_text(state.get("question", "")).strip()
        incoming_history = _ensure_list(state.get("conversation_history") or state.get("history"))
        context = dict(state.get("context") or {})
        session_id = context.get("session_id") or state.get("session_id") or f"graph-{uuid.uuid4().hex}"
        context["session_id"] = session_id
        conversation_history: List[Dict[str, str]] = incoming_history
        previous_agent_state: Optional[AgentState] = None
        user_turn_id: Optional[int] = None
        if memory_adapter and session_id:
            try:
                if question:
                    user_turn_id = memory_adapter.on_user_turn(session_id, question)
                memory_history = memory_adapter.get_window_for_llm(session_id)
                if memory_history:
                    conversation_history = memory_history
                previous_agent_state = load_previous_agent_state(memory_adapter, session_id)
            except Exception:
                logger.debug("[GRAPH] Unable to prime memory for ingest", exc_info=True)
        next_state: AgentState = {
            "question": question,
            "history": previous_agent_state,
            "conversation_history": conversation_history,
            "context": context,
            "session_id": session_id,
            "intent": {
                "macro_cls": None,
                "intent_cls": None,
                "context_cls": None,
                "intent": None,
                "context_mode": None,
            },
            "entities": [],
        }
        if user_turn_id is not None:
            next_state["user_turn_id"] = user_turn_id
        return next_state

    return ingest_node


def make_intent_node(memory_adapter: Any, predict_with_router=None):
    def intent_node(state: AgentState) -> AgentState:
        question = state.get("question", "")
        session_id = state.get("session_id", "")

        classification = state.get("classification")
        intent_label = ""
        context_label = ""
        macro_label = None
        if classification is not None:
            intent_label = (getattr(classification, "intent", None) or "")
            context_label = (getattr(classification, "context", None) or "")
            macro_label = getattr(classification, "macro", None)
        elif callable(predict_with_router):
            results = predict_with_router(question)
            intent_label = (
                getattr(getattr(results, "intent_cls", None), "label", "")
                or getattr(getattr(results, "intent", None), "label", "")
                or ""
            )
            context_label = (
                getattr(getattr(results, "context_cls", None), "label", "")
                or getattr(getattr(results, "context_mode", None), "label", "")
                or ""
            )
            macro_label = getattr(getattr(results, "macro_cls", None), "label", None)

        normalized_intent = "method" if intent_label == "methodology" else intent_label

        if macro_label in (0, "0", False):
            decision = "fallback"
        elif normalized_intent == "other" and context_label == "standalone":
            decision = "fallback"
        elif normalized_intent == "value":
            decision = "data"
        elif normalized_intent == "method":
            decision = "rag"
        else:
            decision = "fallback"

        logger.info(
            "[INTENT_NODE] PIBOT_INTENT_ROUTE | intent_cls=%s | context_cls=%s | decision=%s",
            normalized_intent,
            context_label,
            decision,
        )

        intent_envelope = dict(state.get("intent") or {})
        intent_envelope["macro_cls"] = macro_label
        intent_envelope["intent_cls"] = normalized_intent
        intent_envelope["context_cls"] = context_label
        intent_envelope["intent"] = normalized_intent
        intent_envelope["context_mode"] = decision

        entities = _clone_entities(state.get("entities"))
        primary_entity = _ensure_entity_slot(entities, 0)

        prior_entity = extract_latest_entity_from_history(state.get("history"))
        if prior_entity:
            _merge_entity_fields(
                primary_entity,
                {
                    "indicator": prior_entity.get("indicator") or prior_entity.get("indicador"),
                    "activity": prior_entity.get("activity"),
                    "seasonality": prior_entity.get("seasonality"),
                    "region": prior_entity.get("region"),
                    "period": prior_entity.get("period"),
                },
            )

        for key in ("indicator", "activity", "seasonality", "region", "period"):
            primary_entity.setdefault(key, None)

        return {"route_decision": decision, "intent": intent_envelope, "entities": entities}

    return intent_node


def make_router_node():
    def router_node(state: AgentState) -> AgentState:
        decision = state.get("route_decision", "fallback")
        entities = _clone_entities(state.get("entities"))
        primary_entity = _ensure_entity_slot(entities, 0)
        history_entity = extract_latest_entity_from_history(state.get("history"))
        history_payload = {}
        if history_entity:
            history_payload = {
                "indicator": history_entity.get("indicator") or history_entity.get("indicador"),
                "activity": history_entity.get("activity"),
                "seasonality": history_entity.get("seasonality"),
                "region": history_entity.get("region"),
                "period": history_entity.get("period"),
            }
        _merge_entity_fields(primary_entity, history_payload)

        logger.info("[ROUTER_NODE] routing=%s", decision)
        return {"route_decision": decision, "entities": entities}

    return router_node


__all__ = [
    "make_ingest_node",
    "make_intent_node",
    "make_router_node",
]
