"""Backend bootstrap helpers for PIBot graph."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from orchestrator.classifier.intent_store import IntentStoreBase, create_intent_store
from orchestrator.llm.llm_adapter import LLMAdapter, build_llm
from orchestrator.memory.memory_adapter import MemoryAdapter
from orchestrator.rag.rag_factory import create_retriever

logger = logging.getLogger(__name__)


class _InProcessMemoryAdapter:
    """Minimal drop-in replacement when persistent memory is unavailable."""

    def __init__(self, max_turns: int = 8):
        self.saver = None
        self._history: Dict[str, List[Dict[str, str]]] = {}
        self._max_turns = max_turns

    def on_user_turn(self, session_id: str, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._append_history(session_id, "user", message, metadata=metadata)

    def on_assistant_turn(self, session_id: str, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._append_history(session_id, "assistant", message, metadata=metadata)

    def _append_history(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not session_id or not content:
            return
        history = self._history.setdefault(session_id, [])
        entry = {"role": role, "content": str(content)}
        if metadata:
            entry["metadata"] = dict(metadata)
        history.append(entry)
        if len(history) > 200:
            self._history[session_id] = history[-200:]

    def get_window_for_llm(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        limit = max_turns or self._max_turns
        turns = self._history.get(session_id, [])
        return turns[-limit:]

    def get_backend_status(self) -> Dict[str, Any]:
        return {
            "using_pg": False,
            "require_pg": False,
            "backend": "in-process",
        }

    def clear_session(self, session_id: str) -> bool:
        if not session_id:
            return False
        try:
            self._history.pop(session_id, None)
            return True
        except Exception:
            return False


def _safe_memory_adapter() -> Any:
    require_pg = os.getenv("REQUIRE_PG_MEMORY", "0").lower() in {"1", "true", "yes", "on"}
    force_local = os.getenv("PG_FORCE_LOCALHOST", "1").lower() in {"1", "true", "yes", "on"}
    pg_dsn = None
    if force_local:
        pg_dsn = os.getenv("PG_LOCALHOST_DSN") or "postgresql://postgres:postgres@localhost:5432/pibot"
    else:
        pg_dsn = os.getenv("PG_DSN") or os.getenv("DATABASE_URL")
    try:
        if pg_dsn:
            return MemoryAdapter(pg_dsn=pg_dsn)
        return MemoryAdapter()
    except Exception as exc:  # pragma: no cover - best-effort init
        if require_pg:
            logger.error(
                "[GRAPH] MemoryAdapter require_pg enabled but failed: %s",
                exc,
            )
            raise
        logger.warning("[GRAPH] MemoryAdapter unavailable, using in-process fallback: %s", exc)
        return _InProcessMemoryAdapter()


def _safe_retriever():
    try:
        return create_retriever()
    except Exception:  # pragma: no cover - retriever optional
        logger.exception("[GRAPH] Retriever initialization failed")
        return None


def _safe_intent_store() -> Optional[IntentStoreBase]:
    try:
        return create_intent_store()
    except Exception:
        logger.exception("[GRAPH] IntentStore initialization failed")
        return None


def _safe_llm(*, retriever=None, mode: str = "rag") -> Optional[LLMAdapter]:
    try:
        return build_llm(streaming=True, retriever=retriever, mode=mode)
    except Exception:  # pragma: no cover - avoid crashing import
        logger.exception("[GRAPH] LLMAdapter initialization failed")
        return None


__all__ = [
    "_safe_memory_adapter",
    "_safe_retriever",
    "_safe_intent_store",
    "_safe_llm",
    "_InProcessMemoryAdapter",
]
