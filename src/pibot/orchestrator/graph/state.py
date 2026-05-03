"""Shared typed structures and helpers for PIBot agent graph."""

from __future__ import annotations

import datetime
from typing import Annotated, Any, Dict, Iterable, List, Optional, Tuple, TypedDict

from langgraph.channels.topic import Topic
from langgraph.runtime import get_runtime
from langgraph.types import StreamWriter


class IntentEnvelope(TypedDict, total=False):
    # Nueva taxonomía
    macro_cls: Optional[str]
    intent_cls: Optional[str]
    context_cls: Optional[str]
    # Compatibilidad legacy
    intent: Optional[str]
    context_mode: Optional[str]


class EntityPayload(TypedDict, total=False):
    id: Optional[str]
    role: Optional[str]
    indicador: Optional[str]
    indicator: Optional[str]
    activity: Optional[str]
    activity_cls: Optional[str]
    seasonality: Optional[str]
    region: Optional[str]
    region_cls: Optional[str]
    period: Optional[str]
    metric_type_cls: Optional[str]
    calc_mode_cls: Optional[str]
    req_form_cls: Optional[str]
    frequency_cls: Optional[str]
    timeseries: Dict[str, float]


class AgentState(TypedDict, total=False):
    question: str
    history: Optional["AgentState"]
    conversation_history: List[Dict[str, str]]
    context: Dict[str, Any]
    session_id: str
    user_turn_id: int
    classification: Any  # ClassificationResult, but typed late to avoid cycle
    history_text: str
    intent_info: Dict[str, Any]
    intent: IntentEnvelope
    entities: List[EntityPayload]
    route_decision: str
    output: str
    stream_chunks: Annotated[List[str], Topic(str, accumulate=True)]
    parsed_point: Optional[str]
    parsed_range: Optional[Tuple[str, str]]
    series: Optional[str]
    data_classification: Dict[str, Any]
    data_params: Dict[str, Any]
    data_params_status: Dict[str, str]
    metadata_response: Dict[str, Any]
    metadata_key: Optional[str]
    series_fetch_args: Dict[str, Any]
    series_fetch_result: Dict[str, Any]


def _emit_stream_chunk(chunk_text: str, writer: Optional[StreamWriter]) -> None:
    if not chunk_text:
        return

    resolved_writer: Optional[StreamWriter] = writer
    if resolved_writer is None:
        try:
            runtime = get_runtime()
            resolved_writer = getattr(runtime, "stream_writer", None)
        except Exception:  # pragma: no cover - runtime may be unavailable
            resolved_writer = None

    if resolved_writer:
        try:
            resolved_writer({"stream_chunks": chunk_text})
        except Exception:  # pragma: no cover - defensive logging only
            pass



def _ensure_list(history: Optional[Iterable[Dict[str, str]]]) -> List[Dict[str, str]]:
    if not history:
        return []
    try:
        return [dict(item) for item in history]
    except Exception:
        return list(history)  # type: ignore[arg-type]


def _ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_choice_content(choice: Any) -> Iterable[str]:
    if choice is None:
        return []
    snippets: List[str] = []
    delta = getattr(choice, "delta", None)
    if delta is not None:
        content = getattr(delta, "content", None)
        if isinstance(content, list):
            for item in content:
                snippets.append(_ensure_text(getattr(item, "text", getattr(item, "content", item))))
        elif content:
            snippets.append(_ensure_text(content))
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content:
            snippets.append(_ensure_text(content))
    text = getattr(choice, "text", None)
    if text:
        snippets.append(_ensure_text(text))
    return [s for s in snippets if s]


def _yield_openai_stream_chunks(stream: Any) -> Iterable[str]:
    if stream is None:
        return []
    if isinstance(stream, (str, bytes)):
        return [str(stream)]

    try:
        iterator = iter(stream)
    except TypeError:
        iterator = None

    chunks: List[str] = []
    if iterator is not None and not isinstance(stream, dict):
        for chunk in iterator:
            choices = getattr(chunk, "choices", None)
            if choices:
                for choice in choices:
                    chunks.extend(_extract_choice_content(choice))
            else:
                chunks.append(_ensure_text(getattr(chunk, "content", getattr(chunk, "text", chunk))))
        return [c for c in chunks if c]

    choices = getattr(stream, "choices", None)
    if choices:
        for choice in choices:
            chunks.extend(_extract_choice_content(choice))
    else:
        content = getattr(stream, "content", getattr(stream, "text", None))
        if content:
            chunks.append(_ensure_text(content))
    return [c for c in chunks if c]


def _clone_entities(raw_entities: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_entities, list):
        return []
    cloned: List[Dict[str, Any]] = []
    for item in raw_entities:
        if isinstance(item, dict):
            cloned.append(dict(item))
    return cloned


def _ensure_entity_slot(entities: List[Dict[str, Any]], index: int = 0) -> Dict[str, Any]:
    while len(entities) <= index:
        entities.append({})
    slot = entities[index]
    if not isinstance(slot, dict):
        slot = {}
        entities[index] = slot
    return slot


def _merge_entity_fields(
    entity: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    for key, value in payload.items():
        if value is None:
            continue
        if not overwrite and entity.get(key) not in (None, ""):
            continue
        entity[key] = value
        if key == "indicator" and value is not None:
            entity["indicador"] = value


def _build_timeseries_map(series: Iterable[Tuple[datetime.datetime, float]]) -> Dict[str, float]:
    ts: Dict[str, float] = {}
    for dt_value, val in series:
        try:
            key = dt_value.strftime("%Y-%m-%d")
        except Exception:
            continue
        try:
            ts[key] = float(val)
        except (TypeError, ValueError):
            continue
    return ts


__all__ = [
    "AgentState",
    "IntentEnvelope",
    "EntityPayload",
    "_emit_stream_chunk",
    "_ensure_list",
    "_ensure_text",
    "_yield_openai_stream_chunks",
    "_clone_entities",
    "_ensure_entity_slot",
    "_merge_entity_fields",
    "_build_timeseries_map",
]
