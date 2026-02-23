"""Ingest and routing nodes for the PIBot LangGraph."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import uuid

from ..state import (
    AgentState,
    _clone_entities,
    _ensure_entity_slot,
    _ensure_list,
    _ensure_text,
    _merge_entity_fields,
)
from ..session import extract_latest_entity_from_history, load_previous_agent_state

logger = logging.getLogger(__name__)


def make_ingest_node(memory_adapter: Any):
    def ingest_node(state: AgentState) -> AgentState:
        question = _ensure_text(state.get("question", "")).strip()
        incoming_history = _ensure_list(state.get("conversation_history") or state.get("history"))
        context = dict(state.get("context") or {})
        session_id = context.get("session_id") or state.get("session_id") or f"graph-{uuid.uuid4().hex}"
        context["session_id"] = session_id
        conversation_history: List[Dict[str, str]] = incoming_history
        previous_agent_state: Optional[AgentState] = None
        user_turn_id: Optional[int] = None
        if memory_adapter and session_id:
            try:
                if question:
                    user_turn_id = memory_adapter.on_user_turn(session_id, question)
                memory_history = memory_adapter.get_window_for_llm(session_id)
                if memory_history:
                    conversation_history = memory_history
                previous_agent_state = load_previous_agent_state(memory_adapter, session_id)
            except Exception:
                logger.debug("[GRAPH] Unable to prime memory for ingest", exc_info=True)
        next_state: AgentState = {
            "question": question,
            "history": previous_agent_state,
            "conversation_history": conversation_history,
            "context": context,
            "session_id": session_id,
            "intent": {
                "macro_cls": None,
                "intent_cls": None,
                "context_cls": None,
                "intent": None,
                "context_mode": None,
            },
            "entities": [],
        }
        if user_turn_id is not None:
            next_state["user_turn_id"] = user_turn_id
        return next_state

    return ingest_node


def _coerce_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()

def _extract_label(value: Any) -> Any:
    if isinstance(value, dict):
        if "label" in value:
            return value.get("label")
        return None
    return value

def _get_nested(payload: Any, *keys: str) -> Any:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _ensure_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _ensure_nested_dict(payload: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    current = payload
    for key in keys:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    return current


def _set_nested(payload: Dict[str, Any], value: Any, *keys: str) -> None:
    if not keys:
        return
    parent = _ensure_nested_dict(payload, *keys[:-1]) if len(keys) > 1 else payload
    parent[keys[-1]] = value


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return bool(cleaned) and cleaned != "none"
    if isinstance(value, (list, tuple, set)):
        return any(_has_value(item) for item in value)
    if isinstance(value, dict):
        return bool(value)
    return True


def _first_value(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if _has_value(item):
                return str(item).strip()
        return ""
    if isinstance(value, str) and _has_value(value):
        return value.strip()
    return ""


def _has_explicit_indicator_signal(predict_payload: Dict[str, Any]) -> bool:
    entities = _get_nested(predict_payload, "entities_normalized")
    if isinstance(entities, dict) and _has_value(entities.get("indicator")):
        return True

    slot_tags = _get_nested(predict_payload, "slot_tags") or _get_nested(predict_payload, "slots")
    if isinstance(slot_tags, list):
        for tag in slot_tags:
            if isinstance(tag, str) and "indicator" in tag.lower():
                return True
    return False


def _get_previous_intent_record(intent_store: Any, session_id: str, user_turn_id: Optional[int]) -> Any:
    if not intent_store or not hasattr(intent_store, "history"):
        return None
    if not session_id or user_turn_id is None:
        return None
    try:
        history = intent_store.history(session_id, k=25)
    except Exception:
        logger.debug("[INTENT_NODE] Unable to load intent history", exc_info=True)
        return None
    latest = None
    latest_turn = None
    for record in history or []:
        turn_id = getattr(record, "turn_id", None)
        if turn_id is None:
            continue
        if int(turn_id) >= int(user_turn_id):
            continue
        if latest_turn is None or int(turn_id) > int(latest_turn):
            latest = record
            latest_turn = turn_id
    return latest


def make_intent_node(memory_adapter: Any, intent_store: Any = None, predict_with_router=None):
    def intent_node(state: AgentState) -> AgentState:
        question = state.get("question", "")
        session_id = state.get("session_id", "")

        classification = state.get("classification")
        intent_label = ""
        context_label = ""
        macro_label = None
        intent_raw: Dict[str, Any] = {}
        predict_raw: Dict[str, Any] = {}
        if classification is not None:
            intent_label = _coerce_label(_extract_label(getattr(classification, "intent", None)))
            context_label = _coerce_label(_extract_label(getattr(classification, "context", None)))
            macro_label = _extract_label(getattr(classification, "macro", None))
            intent_raw = _ensure_dict(getattr(classification, "intent_raw", None))
            predict_raw = _ensure_dict(getattr(classification, "predict_raw", None))
        elif callable(predict_with_router):
            results = predict_with_router(question)
            intent_label = (
                getattr(getattr(results, "intent_cls", None), "label", "")
                or getattr(getattr(results, "intent", None), "label", "")
                or ""
            )
            context_label = (
                getattr(getattr(results, "context_cls", None), "label", "")
                or getattr(getattr(results, "context_mode", None), "label", "")
                or ""
            )
            macro_label = getattr(getattr(results, "macro_cls", None), "label", None)
            intent_label = _coerce_label(intent_label)
            context_label = _coerce_label(context_label)
            macro_label = _extract_label(macro_label)

        user_turn_id = state.get("user_turn_id")
        followup_label = _coerce_label(_get_nested(intent_raw, "context", "label")) or _coerce_label(context_label)
        is_followup = followup_label == "followup"
        force_fallback = False

        if is_followup:
            prev_record = _get_previous_intent_record(intent_store, session_id, user_turn_id)
            if prev_record is None:
                force_fallback = True
            else:
                prev_intent_raw = _ensure_dict(getattr(prev_record, "intent_raw", None))
                prev_predict_raw = _ensure_dict(getattr(prev_record, "predict_raw", None))

                macro_label_raw = _get_nested(intent_raw, "macro", "label")
                intent_label_raw = _coerce_label(_get_nested(intent_raw, "intent", "label")) or _coerce_label(intent_label)

                if macro_label_raw in (0, "0", False):
                    prev_macro = _get_nested(prev_intent_raw, "macro", "label")
                    if prev_macro is not None:
                        _set_nested(intent_raw, prev_macro, "macro", "label")
                        macro_label_raw = prev_macro
                        macro_label = prev_macro

                if intent_label_raw in ("none", "", "other"):
                    prev_intent = _coerce_label(_get_nested(prev_intent_raw, "intent", "label"))
                    if not prev_intent:
                        prev_intent = _coerce_label(getattr(prev_record, "intent", None))
                    if prev_intent:
                        _set_nested(intent_raw, prev_intent, "intent", "label")
                        intent_label_raw = prev_intent
                        intent_label = prev_intent

                resolved_intent = "method" if intent_label_raw == "methodology" else intent_label_raw
                entities_normalized = _ensure_dict(_get_nested(predict_raw, "entities_normalized"))
                prev_entities_normalized = _ensure_dict(_get_nested(prev_predict_raw, "entities_normalized"))

                def _prev_indicator() -> str:
                    prev_value = prev_entities_normalized.get("indicator")
                    if _has_value(prev_value):
                        return _first_value(prev_value) or str(prev_value)
                    prev_value = _get_nested(prev_intent_raw, "entities_normalized", "indicator")
                    if _has_value(prev_value):
                        return _first_value(prev_value) or str(prev_value)
                    return ""

                def _backfill_entity_norm(keys: Iterable[str]) -> None:
                    for key in keys:
                        if _has_value(entities_normalized.get(key)):
                            continue
                        prev_value = prev_entities_normalized.get(key)
                        if _has_value(prev_value):
                            entities_normalized[key] = prev_value

                if resolved_intent == "value":
                    matched = False
                    indicator_missing = not _has_value(entities_normalized.get("indicator"))

                    region_specific = _coerce_label(_get_nested(predict_raw, "intents", "region", "label")) == "specific"
                    region_value = entities_normalized.get("region")
                    if region_specific and _has_value(region_value) and indicator_missing:
                        prev_indicator = _prev_indicator()
                        if prev_indicator:
                            entities_normalized["indicator"] = prev_indicator
                            matched = True
                        else:
                            force_fallback = True

                    activity_specific = _coerce_label(_get_nested(predict_raw, "intents", "activity", "label")) == "specific"
                    activity_value = _coerce_label(_first_value(entities_normalized.get("activity")))
                    if activity_specific and activity_value and indicator_missing and not matched:
                        pib_activities = {
                            "agropecuario",
                            "pesca",
                            "industria",
                            "electricidad",
                            "construccion",
                            "restaurantes",
                            "transporte",
                            "comunicaciones",
                            "servicio_financieros",
                            "servicios_empresariales",
                            "servicio_viviendas",
                            "servicio_personales",
                            "admin_publica",
                            "impuestos",
                        }
                        imacec_activities = {
                            "bienes",
                            "mineria",
                            "industria",
                            "resto_bienes",
                            "servicios",
                            "no_mineria",
                        }
                        prev_activities = {"comercio", "impuesto"}
                        if activity_value in pib_activities:
                            entities_normalized["indicator"] = "pib"
                            matched = True
                        elif activity_value in imacec_activities:
                            entities_normalized["indicator"] = "imacec"
                            matched = True
                        elif activity_value in prev_activities:
                            prev_indicator = _prev_indicator()
                            if prev_indicator:
                                entities_normalized["indicator"] = prev_indicator
                                matched = True
                            else:
                                force_fallback = True

                    investment_specific = _coerce_label(_get_nested(predict_raw, "intents", "investment", "label")) == "specific"
                    investment_value = entities_normalized.get("investment")
                    if investment_specific and _has_value(investment_value) and indicator_missing and not matched:
                        prev_indicator = _prev_indicator()
                        if prev_indicator:
                            entities_normalized["indicator"] = prev_indicator
                            matched = True
                        else:
                            force_fallback = True

                    if matched and not force_fallback:
                        _backfill_entity_norm(["period", "frequency", "seasonality"])
                    if not matched and not force_fallback:
                        force_fallback = True

                elif resolved_intent == "method":
                    explicit_indicator = _has_explicit_indicator_signal(predict_raw)
                    if not explicit_indicator or not _has_value(entities_normalized.get("indicator")):
                        prev_indicator = _prev_indicator()
                        if prev_indicator:
                            entities_normalized["indicator"] = prev_indicator
                else:
                    force_fallback = True

                if entities_normalized:
                    _set_nested(predict_raw, entities_normalized, "entities_normalized")
                    if classification is not None and isinstance(getattr(classification, "normalized", None), dict):
                        classification.normalized.update({
                            "indicator": entities_normalized.get("indicator"),
                            "activity": entities_normalized.get("activity"),
                            "region": entities_normalized.get("region"),
                            "period": entities_normalized.get("period"),
                            "frequency": entities_normalized.get("frequency"),
                            "seasonality": entities_normalized.get("seasonality"),
                        })

        normalized_intent = "method" if intent_label == "methodology" else intent_label

        if force_fallback:
            decision = "fallback"
        elif macro_label in (0, "0", False):
            decision = "fallback"
        elif normalized_intent == "other" and context_label == "standalone":
            decision = "fallback"
        elif normalized_intent == "value":
            decision = "data"
        elif normalized_intent == "method":
            decision = "rag"
        else:
            decision = "fallback"

        logger.info(
            "[INTENT_NODE] PIBOT_INTENT_ROUTE | intent_cls=%s | context_cls=%s | decision=%s",
            normalized_intent,
            context_label,
            decision,
        )

        intent_envelope = dict(state.get("intent") or {})
        intent_envelope["macro_cls"] = macro_label
        intent_envelope["intent_cls"] = normalized_intent
        intent_envelope["context_cls"] = context_label
        intent_envelope["intent"] = normalized_intent
        intent_envelope["context_mode"] = decision

        entities = _clone_entities(state.get("entities"))
        primary_entity = _ensure_entity_slot(entities, 0)

        prior_entity = extract_latest_entity_from_history(state.get("history"))
        if prior_entity:
            _merge_entity_fields(
                primary_entity,
                {
                    "indicator": prior_entity.get("indicator") or prior_entity.get("indicador"),
                    "activity": prior_entity.get("activity"),
                    "seasonality": prior_entity.get("seasonality"),
                    "region": prior_entity.get("region"),
                    "period": prior_entity.get("period"),
                },
            )

        for key in ("indicator", "activity", "seasonality", "region", "period"):
            primary_entity.setdefault(key, None)

        if classification is not None:
            classification.intent_raw = intent_raw
            classification.predict_raw = predict_raw
            classification.intent = normalized_intent or classification.intent
            classification.macro = macro_label

        intent_info = dict(state.get("intent_info") or {})
        intent_info.update(
            {
                "intent": normalized_intent or intent_info.get("intent"),
                "macro": macro_label,
                "intent_raw": intent_raw,
                "predict_raw": predict_raw,
            }
        )

        entities_normalized_final = _ensure_dict(_get_nested(predict_raw, "entities_normalized"))
        if entities_normalized_final:
            _merge_entity_fields(
                primary_entity,
                {
                    "indicator": entities_normalized_final.get("indicator"),
                    "activity": _first_value(entities_normalized_final.get("activity")) or None,
                    "region": _first_value(entities_normalized_final.get("region")) or None,
                    "period": _first_value(entities_normalized_final.get("period")) or None,
                    "seasonality": _first_value(entities_normalized_final.get("seasonality")) or None,
                },
            )

        return {
            "route_decision": decision,
            "intent": intent_envelope,
            "entities": entities,
            "classification": classification,
            "intent_info": intent_info,
        }

    return intent_node


def make_router_node():
    def router_node(state: AgentState) -> AgentState:
        decision = state.get("route_decision", "fallback")
        entities = _clone_entities(state.get("entities"))
        primary_entity = _ensure_entity_slot(entities, 0)
        history_entity = extract_latest_entity_from_history(state.get("history"))
        history_payload = {}
        if history_entity:
            history_payload = {
                "indicator": history_entity.get("indicator") or history_entity.get("indicador"),
                "activity": history_entity.get("activity"),
                "seasonality": history_entity.get("seasonality"),
                "region": history_entity.get("region"),
                "period": history_entity.get("period"),
            }
        _merge_entity_fields(primary_entity, history_payload)

        logger.info("[ROUTER_NODE] routing=%s", decision)
        return {"route_decision": decision, "entities": entities}

    return router_node


__all__ = [
    "make_ingest_node",
    "make_intent_node",
    "make_router_node",
]
