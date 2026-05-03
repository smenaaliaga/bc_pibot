from __future__ import annotations

from typing import Any


def resolve_standalone_route(*, normalized_intent: str, context_label: str, macro_label: Any) -> str:
    if macro_label in (0, "0", False):
        return "fallback"
    if normalized_intent == "other" and context_label == "standalone":
        return "fallback"
    if normalized_intent == "value":
        return "data"
    if normalized_intent == "method":
        return "rag"
    return "fallback"
