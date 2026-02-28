from __future__ import annotations

from typing import Any, Dict

from .routing_utils import (
    as_dict,
    first_non_empty,
    has_explicit_indicator,
    is_empty_value,
    label,
    normalize_intent_label,
    predict_payload_root,
    routing_label,
)


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


def _backfill_time_fields(current_norm: Dict[str, Any], prev_norm: Dict[str, Any]) -> None:
    for key in ("period", "frequency", "seasonality"):
        if is_empty_value(current_norm.get(key)) and not is_empty_value(prev_norm.get(key)):
            current_norm[key] = prev_norm.get(key)


def resolve_followup_route(
    *,
    normalized_intent: str,
    context_label: str,
    macro_label: Any,
    current_turn_id: Any,
    payload_root: Dict[str, Any],
    current_intents: Dict[str, Any],
    current_norm: Dict[str, Any],
    prev_intent_raw: Dict[str, Any],
    prev_predict_raw: Dict[str, Any],
) -> Dict[str, Any]:
    current_norm = dict(as_dict(current_norm))
    prev_intent_raw = as_dict(prev_intent_raw)
    prev_predict_raw = as_dict(prev_predict_raw)

    prev_root = predict_payload_root(prev_predict_raw)
    prev_norm = as_dict(prev_root.get("entities_normalized"))

    is_first_turn = bool(current_turn_id in (None, 0, 1))
    if is_first_turn:
        explicit_indicator = has_explicit_indicator(payload_root)
        entities_payload = as_dict(payload_root.get("entities"))
        has_explicit_region = not is_empty_value(entities_payload.get("region"))
        has_explicit_signal = explicit_indicator or has_explicit_region

        if has_explicit_signal and macro_label not in (0, "0", False):
            context_label = "standalone"
            if normalized_intent == "value":
                decision = "data"
            elif normalized_intent == "method":
                decision = "rag"
            else:
                decision = "fallback"
        else:
            decision = "fallback"

        return {
            "decision": decision,
            "normalized_intent": normalized_intent,
            "context_label": context_label,
            "macro_label": macro_label,
            "current_norm": current_norm,
        }

    if not prev_intent_raw and not prev_predict_raw:
        return {
            "decision": "fallback",
            "normalized_intent": normalized_intent,
            "context_label": context_label,
            "macro_label": macro_label,
            "current_norm": current_norm,
        }

    macro_from_api_is_zero = macro_label in (0, "0")
    if macro_from_api_is_zero or normalized_intent in {"", "none", "other"}:
        prev_macro = routing_label(prev_intent_raw, "macro")
        prev_intent = routing_label(prev_intent_raw, "intent")
        if macro_from_api_is_zero and prev_macro is not None:
            macro_label = prev_macro
        if (macro_from_api_is_zero or normalized_intent in {"", "none", "other"}) and not is_empty_value(prev_intent):
            normalized_intent = normalize_intent_label(prev_intent)

    indicator_missing = is_empty_value(current_norm.get("indicator"))
    prev_indicator = first_non_empty(prev_norm.get("indicator"))
    explicit_indicator = has_explicit_indicator(payload_root)

    if normalized_intent == "value":
        region_label = str(label(current_intents.get("region")) or "").strip().lower()
        activity_label = str(label(current_intents.get("activity")) or "").strip().lower()
        investment_label = str(label(current_intents.get("investment")) or "").strip().lower()
        activity_value = str(first_non_empty(current_norm.get("activity")) or "").strip().lower()

        applied_rule = False

        if region_label == "specific" and not is_empty_value(current_norm.get("region")) and indicator_missing:
            if not is_empty_value(prev_indicator):
                current_norm["indicator"] = prev_indicator
                applied_rule = True
        elif activity_label == "specific" and indicator_missing and activity_value in _PIB_ACTIVITY_HINTS:
            current_norm["indicator"] = "pib"
            applied_rule = True
        elif activity_label == "specific" and indicator_missing and activity_value in _IMACEC_ACTIVITY_HINTS:
            current_norm["indicator"] = "imacec"
            applied_rule = True
        elif activity_label == "specific" and indicator_missing and activity_value in _PREVIOUS_ACTIVITY_HINTS:
            if not is_empty_value(prev_indicator):
                current_norm["indicator"] = prev_indicator
                applied_rule = True
        elif investment_label == "specific" and not is_empty_value(current_norm.get("investment")) and indicator_missing:
            if not is_empty_value(prev_indicator):
                current_norm["indicator"] = prev_indicator
                applied_rule = True

        if applied_rule:
            _backfill_time_fields(current_norm, prev_norm)
            decision = "data"
        else:
            decision = "fallback"

    elif normalized_intent == "method":
        if not explicit_indicator:
            if is_empty_value(prev_indicator):
                decision = "fallback"
            else:
                current_norm["indicator"] = prev_indicator
                decision = "rag"
        elif indicator_missing:
            if is_empty_value(prev_indicator):
                decision = "fallback"
            else:
                current_norm["indicator"] = prev_indicator
                decision = "rag"
        else:
            decision = "rag"
    else:
        decision = "fallback"

    return {
        "decision": decision,
        "normalized_intent": normalized_intent,
        "context_label": context_label,
        "macro_label": macro_label,
        "current_norm": current_norm,
    }
