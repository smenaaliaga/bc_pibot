"""
Shared helpers for throttled Postgres logging.
"""
from __future__ import annotations

import time
from typing import Optional, Dict, Any


def throttled_pg_log(
    logger,
    state: Dict[str, Any],
    msg: str,
    *,
    session_id: Optional[str] = None,
    op: Optional[str] = None,
    table: Optional[str] = None,
    pool: Optional[str] = None,
    period: float = 60.0,
) -> None:
    """
    Log Postgres-related errors with throttling and structured context.

    state: mutable dict holding 'last_time', 'last_msg', 'repeat'.
    period: minimum seconds between repeated identical logs at error level.
    """
    now = time.time()
    last_time = state.get("last_time", 0.0)
    last_msg = state.get("last_msg", "")
    repeat = state.get("repeat", 0)

    same_msg = msg == last_msg
    fields = {
        "session_id": session_id or "",
        "op": op or "",
        "table": table or "",
        "repeat": repeat,
        "pool": pool or "",
    }

    if same_msg and (now - last_time) < period:
        state["repeat"] = repeat + 1
        logger.debug(msg, extra=fields)
        return

    state["last_time"] = now
    state["last_msg"] = msg
    state["repeat"] = 0
    # Use exception to keep stacktrace when available; callers may call from except blocks.
    logger.exception(msg, extra=fields)
