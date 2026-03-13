from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

_DEFAULT_FILE_NAME = "run_detail.log"


def _resolve_log_path() -> Path:
    custom_path = str(os.getenv("RUN_DETAIL_LOG_PATH", "")).strip()
    if custom_path:
        return Path(custom_path)

    file_name = str(os.getenv("RUN_DETAIL_LOG", "")).strip() or _DEFAULT_FILE_NAME
    custom_dir = str(os.getenv("RUN_DETAIL_LOG_DIR", "")).strip()
    if custom_dir:
        return Path(custom_dir) / file_name

    # Default: mantener el archivo dentro de orchestrator/logs.
    return Path(__file__).resolve().parents[1] / "logs" / file_name


def safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        return str(data)


def append_run_detail(event: str, payload: Any, *, session_id: str | None = None) -> None:
    path = _resolve_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat()
    sid = str(session_id or "-")
    line = f"{ts} | session={sid} | event={event} | payload={safe_json(payload)}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _detail_ts() -> str:
    # Formato legible alineado con ejemplos de trazabilidad operativa.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]


def append_detail_trace(message: str, *, session_id: str | None = None) -> None:
    """Agrega una línea humana de diagnóstico en run_detail.log.

    Formato:
      2026-03-12 12:34:56,789 | [DETAIL_TRACE] <message>
    """
    path = _resolve_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{_detail_ts()} | [DETAIL_TRACE]"
    sid = str(session_id or "").strip()
    if sid:
        prefix += f" session={sid}"
    line = f"{prefix} {str(message or '').rstrip()}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def append_detail_trace_lines(lines: Iterable[str], *, session_id: str | None = None) -> None:
    for line in lines:
        append_detail_trace(str(line), session_id=session_id)
