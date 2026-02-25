"""Memory node helpers for PIBot graph."""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List

from ..session import persist_agent_state
from ..state import AgentState
from ..suggestions import extract_chart_metadata_from_output, generate_suggested_questions

logger = logging.getLogger(__name__)


def make_memory_node(memory_adapter: Any, intent_store) -> Any:
    def memory_node(state: AgentState) -> AgentState:
        session_id = state.get("session_id")
        output = state.get("output", "")
        metadata = dict(state.get("memory_metadata") or {})
        chart_meta = extract_chart_metadata_from_output(output)
        if chart_meta:
            metadata.update(chart_meta)
        if memory_adapter and session_id:
            try:
                if output:
                    if metadata:
                        memory_adapter.on_assistant_turn(session_id, output, metadata=metadata)
                    else:
                        memory_adapter.on_assistant_turn(session_id, output)

                _persist_fallback_payload(memory_adapter, session_id, state)
                _persist_classification_facts(memory_adapter, session_id, state)
                persist_agent_state(memory_adapter, session_id, state)
            except Exception:
                logger.debug("[GRAPH] Unable to persist assistant turn", exc_info=True)

        output = _append_followups(output, state, intent_store)

        return {"output": output}

    return memory_node


def _append_followups(output: str, state: AgentState, intent_store) -> str:
    suggested_questions = generate_suggested_questions(state, intent_store)
    if "##FOLLOWUP_START" in output:
        # Do not append a second followup block, but ensure suggestions/state
        # have been updated by generate_suggested_questions above.
        return output
    if not suggested_questions:
        suggested_questions = [
            "¿Quieres que busque los datos más recientes?",
            "¿Te muestro un gráfico con la última variación?",
            "¿Prefieres consultar IMACEC o PIB?",
        ]
    followup_block = "\n\n##FOLLOWUP_START\n"
    for i, question in enumerate(suggested_questions[:3], start=1):
        followup_block += f"suggestion_{i}={question}\n"
    followup_block += "##FOLLOWUP_END"
    return output + followup_block


def _persist_classification_facts(memory_adapter: Any, session_id: str, state: AgentState) -> None:
    try:
        intent_payload = state.get("intent") or {}
        entities = state.get("entities") or []
        primary = entities[0] if isinstance(entities, list) and entities else {}
        facts = {
            "macro_cls": intent_payload.get("macro_cls"),
            "intent_cls": intent_payload.get("intent_cls"),
            "context_cls": intent_payload.get("context_cls"),
            "indicator": primary.get("indicator") or primary.get("indicador"),
            "seasonality": primary.get("seasonality"),
            "activity": primary.get("activity"),
            "region": primary.get("region"),
            "period": primary.get("period"),
            "activity_cls": primary.get("activity_cls"),
            "region_cls": primary.get("region_cls"),
            "calc_mode_cls": primary.get("calc_mode_cls"),
            "frequency_cls": primary.get("frequency_cls"),
            "req_form_cls": primary.get("req_form_cls"),
        }
        facts = {k: v for k, v in facts.items() if v not in (None, "")}
        if facts:
            memory_adapter.set_facts(session_id, facts)
            logger.info(
                "[MEMORY_NODE] Clasificación guardada en memoria | session=%s | keys=%s",
                session_id,
                list(facts.keys()),
            )
    except Exception:
        logger.debug("[MEMORY_NODE] Unable to persist classification facts", exc_info=True)


def _persist_fallback_payload(memory_adapter: Any, session_id: str, state: AgentState) -> None:
    intent_payload = state.get("intent") or {}
    intent_cls = intent_payload.get("intent_cls")
    context_cls = intent_payload.get("context_cls")
    if intent_cls != "other" or context_cls != "standalone":
        return
    question = (state.get("question") or "").strip()
    if not question:
        return
    items = _extract_session_payload(question)
    if not items:
        return
    _append_session_payload(memory_adapter, session_id, items)


def _extract_session_payload(question: str) -> List[Dict[str, Any]]:
    lowered = question.lower().strip()
    items: List[Dict[str, Any]] = []

    name = _extract_name_from_text(question)
    if name:
        items.append({"type": "profile", "key": "name", "value": name})

    methodology_only_tokens = (
        "only want methodological",
        "only methodological",
        "solo quiero respuestas metodologicas",
        "solo quiero respuestas metodológicas",
        "solo quiero metodologicas",
        "solo quiero metodológicas",
        "solo quiero metodologia",
        "solo quiero metodología",
    )
    if any(token in lowered for token in methodology_only_tokens):
        items.append({"type": "preference", "key": "methodology_only", "value": True})

    if not items:
        items.append({"type": "fallback_note", "value": question})
    return items


def _extract_name_from_text(text: str) -> str:
    lowered = text.lower()
    markers = (
        "my name is ",
        "mi nombre es ",
        "me llamo ",
    )
    for marker in markers:
        if marker in lowered:
            start = lowered.index(marker) + len(marker)
            return _extract_value_tail(text, start)
    if lowered.startswith("soy "):
        return _extract_value_tail(text, len("soy "))
    return ""


def _extract_value_tail(text: str, start_index: int) -> str:
    tail = text[start_index:]
    for sep in (".", ",", "!", "?", ";", ":"):
        if sep in tail:
            tail = tail.split(sep, 1)[0]
    return tail.strip().strip("\"' ")


def _append_session_payload(memory_adapter: Any, session_id: str, items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    payload_entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "items": items,
    }
    existing = {}
    if hasattr(memory_adapter, "get_facts"):
        try:
            existing = memory_adapter.get_facts(session_id) or {}
        except Exception:
            existing = {}
    current_payload = existing.get("session_payload") if isinstance(existing, dict) else None
    payload_list: List[Dict[str, Any]] = []
    if isinstance(current_payload, list):
        payload_list = list(current_payload)
    elif isinstance(current_payload, str):
        try:
            parsed = json.loads(current_payload)
            if isinstance(parsed, list):
                payload_list = parsed
        except Exception:
            payload_list = []
    payload_list.append(payload_entry)
    memory_adapter.set_facts(session_id, {"session_payload": payload_list})


__all__ = ["make_memory_node"]
