from __future__ import annotations

from typing import Any, Dict


def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def label(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("label")
    return value


def is_empty_value(value: Any) -> bool:
    if value in (None, "", "none", "None", "null", "NULL"):
        return True
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return True
    return False


def first_non_empty(value: Any) -> Any:
    if isinstance(value, list):
        for item in value:
            if not is_empty_value(item):
                return item
        return None
    return None if is_empty_value(value) else value


def normalize_intent_label(intent_label: Any) -> str:
    raw = str(intent_label or "").strip().lower()
    if raw == "methodology":
        return "method"
    return raw


def predict_payload_root(predict_raw: Any) -> Dict[str, Any]:
    payload = as_dict(predict_raw)
    interpretation = payload.get("interpretation")
    if isinstance(interpretation, dict):
        return interpretation
    return payload


def routing_label(payload: Dict[str, Any], key: str) -> Any:
    if not isinstance(payload, dict):
        return None
    direct = payload.get(key)
    if not is_empty_value(direct):
        return label(as_dict(direct).get("label") if isinstance(direct, dict) else direct)
    routing = as_dict(payload.get("routing"))
    nested = routing.get(key)
    if is_empty_value(nested):
        return None
    return label(as_dict(nested).get("label") if isinstance(nested, dict) else nested)


def has_explicit_indicator(payload_root: Dict[str, Any]) -> bool:
    if not isinstance(payload_root, dict):
        return False
    entities = as_dict(payload_root.get("entities"))
    if not is_empty_value(entities.get("indicator")):
        return True
    slot_tags = payload_root.get("slot_tags")
    if isinstance(slot_tags, list):
        for tag in slot_tags:
            if str(tag or "").strip().lower() == "b-indicator":
                return True
    return False
