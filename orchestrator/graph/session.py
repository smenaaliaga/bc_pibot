"""Checkpoint helpers for agent state persistence."""

from __future__ import annotations

import copy
import datetime
import logging
from typing import Any, Dict, Optional

from .state import AgentState

logger = logging.getLogger(__name__)


def _coerce_agent_state(value: Any) -> Optional[AgentState]:
    if not isinstance(value, dict):
        return None
    snapshot: Dict[str, Any] = {}
    allowed = {
        "question",
        "session_id",
        "intent",
        "entities",
        "history",
        "conversation_history",
        "output",
    }
    for key in allowed:
        if key in value:
            snapshot[key] = copy.deepcopy(value[key])
    return snapshot  # type: ignore[return-value]


def load_previous_agent_state(memory_adapter: Any, session_id: Optional[str]) -> Optional[AgentState]:
    if not session_id or not memory_adapter:
        return None
    try:
        payload = memory_adapter.load_checkpoint(session_id)
    except Exception:
        logger.debug("[GRAPH] Unable to load checkpoint for session %s", session_id, exc_info=True)
        return None
    if not payload or not isinstance(payload, dict):
        return None
    checkpoint = payload.get("checkpoint") or payload.get("values") or payload
    if not isinstance(checkpoint, dict):
        return None
    candidate = checkpoint.get("agent_state")
    return _coerce_agent_state(candidate)


def _snapshot_agent_state(state: AgentState) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "question": state.get("question"),
        "session_id": state.get("session_id"),
        "intent": copy.deepcopy(state.get("intent")),
        "entities": [copy.deepcopy(ent) for ent in state.get("entities", []) if isinstance(ent, dict)],
        "conversation_history": [dict(item) for item in state.get("conversation_history", [])],
        "output": state.get("output"),
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    history = state.get("history")
    if isinstance(history, dict):
        snapshot["history"] = copy.deepcopy(history)
    else:
        snapshot["history"] = None
    return snapshot


def persist_agent_state(memory_adapter: Any, session_id: Optional[str], state: AgentState) -> None:
    if not session_id or not memory_adapter:
        return
    snapshot = _snapshot_agent_state(state)
    try:
        memory_adapter.save_checkpoint(
            session_id,
            {"agent_state": snapshot},
            metadata={"source": "agent_state"},
        )
    except Exception:
        logger.debug("[GRAPH] Unable to persist agent snapshot", exc_info=True)


def extract_latest_entity_from_history(history: Optional[AgentState]) -> Optional[Dict[str, Any]]:
    cursor: Any = history
    visited = 0
    while isinstance(cursor, dict) and visited < 5:
        entities = cursor.get("entities") if isinstance(cursor, dict) else None
        if isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, dict):
                    return entity
        cursor = cursor.get("history") if isinstance(cursor, dict) else None
        visited += 1
    return None


__all__ = [
    "load_previous_agent_state",
    "persist_agent_state",
    "extract_latest_entity_from_history",
]
