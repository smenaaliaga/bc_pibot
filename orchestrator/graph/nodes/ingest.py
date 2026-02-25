"""Ingest and routing nodes for the PIBot LangGraph."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

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

_PIB_ACTIVITY_HINTS = {
    "agropecuario",
    "pesca",
    "industria",
    "electricidad",
    "construccion",
    "construcción",
    "restaurantes",
    "transporte",
    "comunicaciones",
    "servicio_financieros",
    "servicios_financieros",
    "servicios_empresariales",
    "servicio_viviendas",
    "servicios_vivienda",
    "servicio_personales",
    "servicios_personales",
    "admin_publica",
    "administracion_publica",
    "administración_pública",
    "impuestos",
}

_IMACEC_ACTIVITY_HINTS = {
    "bienes",
    "mineria",
    "minería",
    "industria",
    "resto_bienes",
    "servicios",
    "no_mineria",
    "no_minería",
}

_PREVIOUS_ACTIVITY_HINTS = {
    "comercio",
    "impuesto",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _label(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("label")
    return value


def _is_empty_value(value: Any) -> bool:
    if value in (None, "", "none", "None", "null", "NULL"):
        return True
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return True
    return False


def _first_non_empty(value: Any) -> Any:
    if isinstance(value, list):
        for item in value:
            if not _is_empty_value(item):
                return item
        return None
    return None if _is_empty_value(value) else value


def _normalize_intent_label(intent_label: Any) -> str:
    raw = str(intent_label or "").strip().lower()
    if raw == "methodology":
        return "method"
    return raw


def _predict_payload_root(predict_raw: Any) -> Dict[str, Any]:
    payload = _as_dict(predict_raw)
    interpretation = payload.get("interpretation")
    if isinstance(interpretation, dict):
        return interpretation
    return payload


def _has_explicit_indicator(payload_root: Dict[str, Any]) -> bool:
    if not isinstance(payload_root, dict):
        return False
    entities = _as_dict(payload_root.get("entities"))
    if not _is_empty_value(entities.get("indicator")):
        return True
    slot_tags = payload_root.get("slot_tags")
    if isinstance(slot_tags, list):
        for tag in slot_tags:
            if str(tag or "").strip().lower() == "b-indicator":
                return True
    return False


def _extract_previous_turn_payload(
    intent_store: Any,
    session_id: str,
    current_turn_id: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not intent_store or not hasattr(intent_store, "history") or not session_id:
        return {}, {}
    try:
        records = intent_store.history(session_id, k=10) or []
    except Exception:
        logger.debug("[INTENT_NODE] Unable to read intent history", exc_info=True)
        return {}, {}
    if not isinstance(records, list) or not records:
        return {}, {}

    for rec in reversed(records):
        turn_id = getattr(rec, "turn_id", None)
        if current_turn_id is not None and isinstance(turn_id, int) and turn_id >= current_turn_id:
            continue
        prev_intent_raw = _as_dict(getattr(rec, "intent_raw", None))
        prev_predict_raw = _as_dict(getattr(rec, "predict_raw", None))
        if prev_intent_raw or prev_predict_raw:
            return prev_intent_raw, prev_predict_raw
    return {}, {}


def _backfill_time_fields(current_norm: Dict[str, Any], prev_norm: Dict[str, Any]) -> None:
    for key in ("period", "frequency", "seasonality"):
        if _is_empty_value(current_norm.get(key)) and not _is_empty_value(prev_norm.get(key)):
            current_norm[key] = prev_norm.get(key)


def _build_intent_info_from_state(
    state: AgentState,
    classification: Any,
    intent_raw: Dict[str, Any],
    predict_raw: Dict[str, Any],
    normalized_intent: str,
    context_label: str,
    macro_label: Any,
) -> Dict[str, Any]:
    base = state.get("intent_info") if isinstance(state.get("intent_info"), dict) else {}
    intent_info = dict(base)
    if "intent" not in intent_info:
        intent_info["intent"] = normalized_intent or getattr(classification, "intent", None) or "unknown"
    if "score" not in intent_info:
        intent_info["score"] = getattr(classification, "confidence", 0.0) or 0.0
    intent_info["intent_raw"] = intent_raw
    intent_info["predict_raw"] = predict_raw
    intent_info["macro"] = macro_label
    intent_info["context"] = context_label
    return intent_info


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


def make_intent_node(memory_adapter: Any, intent_store: Any = None, predict_with_router=None):
    if predict_with_router is None and callable(intent_store) and not hasattr(intent_store, "history"):
        predict_with_router = intent_store
        intent_store = None

    def intent_node(state: AgentState) -> AgentState:
        question = state.get("question", "")
        session_id = state.get("session_id", "")
        current_turn_id = state.get("user_turn_id")

        classification = state.get("classification")
        intent_label: Any = ""
        context_label: Any = ""
        macro_label = None
        intent_raw: Dict[str, Any] = {}
        predict_raw: Dict[str, Any] = {}
        if classification is not None:
            intent_label = (getattr(classification, "intent", None) or "")
            context_label = (getattr(classification, "context", None) or "")
            macro_label = getattr(classification, "macro", None)
            intent_raw = _as_dict(getattr(classification, "intent_raw", None))
            predict_raw = _as_dict(getattr(classification, "predict_raw", None))
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

        if not intent_raw:
            current_intent_info = state.get("intent_info") if isinstance(state.get("intent_info"), dict) else {}
            intent_raw = _as_dict(current_intent_info.get("intent_raw"))
        if not predict_raw:
            current_intent_info = state.get("intent_info") if isinstance(state.get("intent_info"), dict) else {}
            predict_raw = _as_dict(current_intent_info.get("predict_raw"))

        intent_raw_context_label = _label(_as_dict(intent_raw.get("context")).get("label") if isinstance(intent_raw.get("context"), dict) else intent_raw.get("context"))
        intent_raw_intent_label = _label(_as_dict(intent_raw.get("intent")).get("label") if isinstance(intent_raw.get("intent"), dict) else intent_raw.get("intent"))
        intent_raw_macro_label = _label(_as_dict(intent_raw.get("macro")).get("label") if isinstance(intent_raw.get("macro"), dict) else intent_raw.get("macro"))

        if _is_empty_value(context_label):
            context_label = intent_raw_context_label
        if _is_empty_value(intent_label):
            intent_label = intent_raw_intent_label
        if macro_label is None:
            macro_label = intent_raw_macro_label

        context_label = str(context_label or "").strip().lower()
        normalized_intent = _normalize_intent_label(intent_label)

        payload_root = _predict_payload_root(predict_raw)
        current_intents = _as_dict(payload_root.get("intents"))
        current_norm = _as_dict(payload_root.get("entities_normalized"))

        if context_label == "followup":
            prev_intent_raw, prev_predict_raw = _extract_previous_turn_payload(intent_store, session_id, current_turn_id)
            prev_root = _predict_payload_root(prev_predict_raw)
            prev_norm = _as_dict(prev_root.get("entities_normalized"))

            is_first_turn = bool(current_turn_id in (None, 0, 1))
            if is_first_turn or (not prev_intent_raw and not prev_predict_raw):
                decision = "fallback"
            else:
                if macro_label in (0, "0") or normalized_intent in {"", "none", "other"}:
                    prev_macro = _label(_as_dict(prev_intent_raw.get("macro")).get("label") if isinstance(prev_intent_raw.get("macro"), dict) else prev_intent_raw.get("macro"))
                    prev_intent = _label(_as_dict(prev_intent_raw.get("intent")).get("label") if isinstance(prev_intent_raw.get("intent"), dict) else prev_intent_raw.get("intent"))
                    if macro_label in (0, "0") and prev_macro is not None:
                        macro_label = prev_macro
                    if normalized_intent in {"", "none", "other"} and not _is_empty_value(prev_intent):
                        normalized_intent = _normalize_intent_label(prev_intent)

                indicator_missing = _is_empty_value(current_norm.get("indicator"))
                prev_indicator = _first_non_empty(prev_norm.get("indicator"))
                explicit_indicator = _has_explicit_indicator(payload_root)

                if normalized_intent == "value":
                    region_label = str(_label(current_intents.get("region")) or "").strip().lower()
                    activity_label = str(_label(current_intents.get("activity")) or "").strip().lower()
                    investment_label = str(_label(current_intents.get("investment")) or "").strip().lower()
                    activity_value = str(_first_non_empty(current_norm.get("activity")) or "").strip().lower()

                    applied_rule = False

                    if region_label == "specific" and not _is_empty_value(current_norm.get("region")) and indicator_missing:
                        if not _is_empty_value(prev_indicator):
                            current_norm["indicator"] = prev_indicator
                            applied_rule = True
                    elif activity_label == "specific" and indicator_missing and activity_value in _PIB_ACTIVITY_HINTS:
                        current_norm["indicator"] = "pib"
                        applied_rule = True
                    elif activity_label == "specific" and indicator_missing and activity_value in _IMACEC_ACTIVITY_HINTS:
                        current_norm["indicator"] = "imacec"
                        applied_rule = True
                    elif activity_label == "specific" and indicator_missing and activity_value in _PREVIOUS_ACTIVITY_HINTS:
                        if not _is_empty_value(prev_indicator):
                            current_norm["indicator"] = prev_indicator
                            applied_rule = True
                    elif investment_label == "specific" and not _is_empty_value(current_norm.get("investment")) and indicator_missing:
                        if not _is_empty_value(prev_indicator):
                            current_norm["indicator"] = prev_indicator
                            applied_rule = True

                    if applied_rule:
                        _backfill_time_fields(current_norm, prev_norm)
                        decision = "data"
                    else:
                        decision = "fallback"

                elif normalized_intent == "method":
                    if not explicit_indicator:
                        if _is_empty_value(prev_indicator):
                            decision = "fallback"
                        else:
                            current_norm["indicator"] = prev_indicator
                            decision = "rag"
                    elif indicator_missing:
                        if _is_empty_value(prev_indicator):
                            decision = "fallback"
                        else:
                            current_norm["indicator"] = prev_indicator
                            decision = "rag"
                    else:
                        decision = "rag"
                else:
                    decision = "fallback"

        else:
            if macro_label in (0, "0", False):
                decision = "fallback"
            elif normalized_intent == "other" and context_label == "standalone":
                decision = "fallback"
            elif normalized_intent == "value":
                decision = "data"
            elif normalized_intent == "method":
                decision = "rag"
            else:
                decision = "fallback"

        if isinstance(payload_root, dict):
            payload_root["entities_normalized"] = current_norm

        if classification is not None:
            try:
                classification.intent = normalized_intent
                classification.context = context_label
                classification.macro = macro_label
                classification.intent_raw = intent_raw
                classification.predict_raw = predict_raw
            except Exception:
                logger.debug("[INTENT_NODE] Unable to mutate classification payload", exc_info=True)

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

        indicator_value = current_norm.get("indicator")
        if not _is_empty_value(indicator_value):
            primary_entity["indicator"] = indicator_value
            primary_entity["indicador"] = indicator_value
        for key in ("activity", "seasonality", "region", "period", "frequency"):
            value = current_norm.get(key)
            if not _is_empty_value(value):
                primary_entity[key] = value

        intent_info = _build_intent_info_from_state(
            state,
            classification,
            intent_raw,
            predict_raw,
            normalized_intent,
            context_label,
            macro_label,
        )

        return {
            "route_decision": decision,
            "intent": intent_envelope,
            "entities": entities,
            "intent_info": intent_info,
            "classification": classification,
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
