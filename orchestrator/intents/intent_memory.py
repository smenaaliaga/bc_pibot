"""Lightweight intent memory store for per-session intent/slot records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import threading
import time


@dataclass
class IntentRecord:
    intent: str
    score: float
    spans: List[Dict[str, Any]] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    turn_id: int = 0
    ts: float = field(default_factory=time.time)


class IntentMemory:
    """Thread-safe in-process intent store (per session)."""

    def __init__(self, max_history: int = 50):
        self._lock = threading.RLock()
        self._sessions: Dict[str, List[IntentRecord]] = {}
        self.max_history = max_history

    def record(
        self,
        session_id: str,
        intent: str,
        score: float,
        spans: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[Dict[str, Any]] = None,
        turn_id: int = 0,
        model_version: Optional[str] = None,
    ) -> IntentRecord:
        rec = IntentRecord(
            intent=intent or "",
            score=float(score or 0.0),
            spans=list(spans or []),
            entities=dict(entities or {}),
            turn_id=turn_id,
        )
        with self._lock:
            lst = self._sessions.setdefault(session_id, [])
            lst.append(rec)
            if len(lst) > self.max_history:
                self._sessions[session_id] = lst[-self.max_history :]
        return rec

    def last(self, session_id: str) -> Optional[IntentRecord]:
        with self._lock:
            lst = self._sessions.get(session_id) or []
            return lst[-1] if lst else None

    def history(self, session_id: str, k: int = 10) -> List[IntentRecord]:
        with self._lock:
            lst = self._sessions.get(session_id) or []
            return lst[-int(k) :]
