from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


_CONFIG_PATH = Path(__file__).resolve().parent / "series_context_config.json"
_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "store_path": "qa/series_context.json",
    "max_history": 5,
}


def _load_config() -> Dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return dict(_DEFAULT_CONFIG)
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except Exception:
        raw = {}
    config = dict(_DEFAULT_CONFIG)
    config.update(raw)
    return config


def _has_valid_series(series_id: Any) -> bool:
    text = str(series_id or "").strip().lower()
    return bool(text and text not in {"none", "null", "nan"})


def _resolve_store_path(config: Dict[str, Any]) -> Path:
    root = Path(__file__).resolve().parents[1]
    return (root / str(config.get("store_path") or _DEFAULT_CONFIG["store_path"])).resolve()


def _read_store(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
        return content if isinstance(content, dict) else {}
    except Exception:
        return {}


def _write_store(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _session_key(payload: Dict[str, Any], session_id: Optional[str]) -> str:
    explicit = str(session_id or "").strip()
    if explicit:
        return explicit
    for key in ("session_id", "thread_id", "conversation_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return "default"


def _normalize_optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _extract_tracking_item(payload: Dict[str, Any]) -> Dict[str, Any]:
    classification = payload.get("classification") or {}
    if not isinstance(classification, dict):
        classification = {}

    req_form_value = classification.get("req_form")
    if req_form_value is None:
        req_form_value = classification.get("req_form_cls")

    return {
        "series_id": _normalize_optional_text(payload.get("series")),
        "indicator": _normalize_optional_text(classification.get("indicator")),
        "freq": _normalize_optional_text(classification.get("frequency")),
        "req_form": _normalize_optional_text(req_form_value),
        "parsed_point": _normalize_optional_text(payload.get("parsed_point")),
        "parsed_range_start": None,
        "parsed_range_end": None,
        "reference_period": _normalize_optional_text(payload.get("reference_period")),
        "calc_mode": _normalize_optional_text(classification.get("calc_mode_cls")),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _fill_range(item: Dict[str, Any], parsed_range: Any) -> Dict[str, Any]:
    if isinstance(parsed_range, (list, tuple)) and parsed_range:
        item["parsed_range_start"] = _normalize_optional_text(parsed_range[0])
        item["parsed_range_end"] = _normalize_optional_text(parsed_range[-1])
    return item


def apply_series_context_before(payload: Dict[str, Any], *, session_id: Optional[str] = None) -> Dict[str, Any]:
    config = _load_config()
    if not bool(config.get("enabled", False)):
        return payload

    if _has_valid_series(payload.get("series")):
        return payload

    store_path = _resolve_store_path(config)
    store = _read_store(store_path)
    skey = _session_key(payload, session_id)
    session_data = store.get(skey) if isinstance(store.get(skey), dict) else {}
    active_series = session_data.get("active_series") if isinstance(session_data, dict) else {}
    if not isinstance(active_series, dict):
        active_series = {}

    active_series_id = active_series.get("series_id")
    if not _has_valid_series(active_series_id):
        return payload

    merged = dict(payload)
    merged["series"] = active_series_id

    classification = merged.get("classification") or {}
    if not isinstance(classification, dict):
        classification = {}

    if not classification.get("frequency") and active_series.get("freq"):
        classification["frequency"] = active_series.get("freq")
    if not classification.get("calc_mode_cls") and active_series.get("calc_mode"):
        classification["calc_mode_cls"] = active_series.get("calc_mode")
    if not classification.get("req_form_cls") and active_series.get("req_form"):
        classification["req_form_cls"] = active_series.get("req_form")
    if not classification.get("req_form") and active_series.get("req_form"):
        classification["req_form"] = active_series.get("req_form")

    merged["classification"] = classification

    if not merged.get("parsed_point") and active_series.get("parsed_point"):
        merged["parsed_point"] = active_series.get("parsed_point")

    if not merged.get("parsed_range") and (
        active_series.get("parsed_range_start") or active_series.get("parsed_range_end")
    ):
        start_value = active_series.get("parsed_range_start")
        end_value = active_series.get("parsed_range_end")
        if start_value and end_value:
            merged["parsed_range"] = (start_value, end_value)

    if not merged.get("reference_period") and active_series.get("reference_period"):
        merged["reference_period"] = active_series.get("reference_period")

    merged["series_context_reused"] = True
    return merged


def apply_series_context_after(payload: Dict[str, Any], *, session_id: Optional[str] = None) -> None:
    config = _load_config()
    if not bool(config.get("enabled", False)):
        return

    item = _extract_tracking_item(payload)
    item = _fill_range(item, payload.get("parsed_range"))

    if not _has_valid_series(item.get("series_id")):
        return

    store_path = _resolve_store_path(config)
    store = _read_store(store_path)
    skey = _session_key(payload, session_id)
    session_data = store.get(skey) if isinstance(store.get(skey), dict) else {}
    history = session_data.get("history") if isinstance(session_data, dict) else []
    if not isinstance(history, list):
        history = []

    active_series = session_data.get("active_series") if isinstance(session_data, dict) else None
    previous_series_id = active_series.get("series_id") if isinstance(active_series, dict) else None
    item["series_changed"] = bool(previous_series_id and previous_series_id != item.get("series_id"))

    history.insert(0, item)
    try:
        max_history = max(1, int(config.get("max_history", 5)))
    except Exception:
        max_history = 5

    store[skey] = {
        "active_series": item,
        "history": history[:max_history],
    }

    _write_store(store_path, store)


def get_series_context_snapshot(
    payload: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    config = _load_config()
    if not bool(config.get("enabled", False)):
        return {"active_series": None, "history": []}

    context_payload = payload if isinstance(payload, dict) else {}
    skey = _session_key(context_payload, session_id)
    store = _read_store(_resolve_store_path(config))
    session_data = store.get(skey) if isinstance(store.get(skey), dict) else {}
    active_series = session_data.get("active_series") if isinstance(session_data, dict) else None
    history = session_data.get("history") if isinstance(session_data, dict) else []
    return {
        "active_series": active_series if isinstance(active_series, dict) else None,
        "history": history if isinstance(history, list) else [],
    }


def get_recent_series_ids(
    payload: Optional[Dict[str, Any]] = None,
    *,
    session_id: Optional[str] = None,
    limit: int = 5,
) -> List[str]:
    snapshot = get_series_context_snapshot(payload, session_id=session_id)
    items: List[str] = []

    active = snapshot.get("active_series")
    if isinstance(active, dict):
        active_series_id = _normalize_optional_text(active.get("series_id"))
        if active_series_id:
            items.append(active_series_id)

    history = snapshot.get("history")
    if isinstance(history, list):
        for row in history:
            if not isinstance(row, dict):
                continue
            series_id = _normalize_optional_text(row.get("series_id"))
            if series_id and series_id not in items:
                items.append(series_id)

    max_items = max(1, int(limit or 5))
    return items[:max_items]


__all__ = [
    "apply_series_context_after",
    "apply_series_context_before",
    "get_recent_series_ids",
    "get_series_context_snapshot",
]
