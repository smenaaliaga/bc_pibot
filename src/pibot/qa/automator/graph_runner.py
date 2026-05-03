"""Ejecuta una pregunta sobre el grafo, replicando el flujo de main.stream_fn.

Importante: este módulo NO usa qa_batch._run_trace_silent. Construye el
grafo y consume el stream con la misma firma que main.py para garantizar
paridad con Streamlit.
"""
from __future__ import annotations
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ._model import _load_dotenv_once

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_graph_singleton = None


def _get_graph():
    global _graph_singleton
    if _graph_singleton is not None:
        return _graph_singleton
    _load_dotenv_once()
    from orchestrator.graph.agent_graph import build_graph  # type: ignore
    _graph_singleton = build_graph()
    return _graph_singleton


def _extract_field(value: Any, field: str):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == field:
                yield nested
            else:
                yield from _extract_field(nested, field)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _extract_field(item, field)


def _iter_strings(payload: Any):
    if payload is None:
        return
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _iter_strings(item)


def _split_event(raw_event):
    mode = None
    payload = raw_event
    if isinstance(raw_event, tuple):
        if len(raw_event) == 2 and isinstance(raw_event[0], str):
            mode, payload = raw_event
        elif len(raw_event) == 3 and isinstance(raw_event[1], str):
            mode, payload = raw_event[1], raw_event[2]
    return mode, payload


def run_question(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Ejecuta una pregunta sobre el grafo y devuelve estado final."""
    graph = _get_graph()
    thread_id = session_id or f"qa-auto-{uuid.uuid4().hex}"
    checkpoint_ns = os.getenv("LANGGRAPH_CHECKPOINT_NS", "memory")
    state: Dict[str, Any] = {
        "question": question,
        "history": [],
        "context": {"session_id": thread_id},
    }
    cfg = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        },
    }
    current_state: Dict[str, Any] = dict(state)
    final_text = ""
    seen_chunk_count = 0
    error: Optional[str] = None

    try:
        for raw_event in graph.stream(state, config=cfg, stream_mode=["updates", "custom"]):
            mode, payload = _split_event(raw_event)

            if mode in (None, "updates", "values") and isinstance(payload, dict):
                for _, delta in payload.items():
                    if isinstance(delta, dict):
                        current_state.update(delta)

            for chunk_payload in _extract_field(payload, "stream_chunks"):
                items = chunk_payload
                if isinstance(chunk_payload, (list, tuple)):
                    start_idx = seen_chunk_count
                    if start_idx < 0 or start_idx > len(chunk_payload):
                        start_idx = len(chunk_payload)
                    items = chunk_payload[start_idx:]
                    seen_chunk_count = len(chunk_payload)
                for chunk_text in _iter_strings(items):
                    if chunk_text:
                        final_text += chunk_text

            if mode in (None, "updates", "values"):
                for out_payload in _extract_field(payload, "output"):
                    for out_text in _iter_strings(out_payload):
                        if isinstance(out_text, str) and out_text:
                            final_text = out_text
    except Exception as exc:  # pragma: no cover
        error = f"{type(exc).__name__}: {exc}"

    classification = current_state.get("classification") or {}
    if hasattr(classification, "intent"):
        intent_label = getattr(classification, "intent", None)
        macro_label = getattr(classification, "macro", None)
        calc_mode = getattr(classification, "calc_mode", None)
    else:
        cls_dict = classification if isinstance(classification, dict) else {}
        intent_label = cls_dict.get("intent")
        macro_label = cls_dict.get("macro")
        calc_mode = cls_dict.get("calc_mode")

    return {
        "response": (final_text or str(current_state.get("output") or "")).strip(),
        "route_decision": current_state.get("route_decision"),
        "metadata_key": current_state.get("metadata_key"),
        "calc_mode": calc_mode,
        "intent_label": intent_label,
        "macro_label": macro_label,
        "error": error,
    }
