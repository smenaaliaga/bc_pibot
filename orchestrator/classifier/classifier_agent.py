"""Clasificador modular basado en el flujo legacy de original orchestrator"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.normalizer.normalizer import (
    coerce_specific_class_labels,
    coerce_req_form_from_period_and_frequency,
    normalize_entities,
)
from orchestrator.utils.http_client import post_json
from config import PREDICT_TIMEOUT_SECONDS, PREDICT_URL

logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ClassificationResult:
    """Resultado de la clasificación de una consulta económica."""
    intent: Optional[str] = None  # Intención ('value', 'methodology', 'ambiguous', etc.)
    confidence: Optional[float] = None  # Confianza del modelo
    entities: Optional[dict] = None  # Entidades raw extraídas
    normalized: Optional[dict] = None  # Entidades normalizadas
    text: Optional[str] = None
    words: Optional[List[str]] = None
    slot_tags: Optional[List[str]] = None
    calc_mode: Optional[dict] = None
    activity: Optional[dict] = None
    region: Optional[dict] = None
    investment: Optional[dict] = None
    req_form: Optional[dict] = None
    macro: Optional[int] = None
    context: Optional[str] = None
    intent_raw: Optional[dict] = None
    predict_raw: Optional[dict] = None



# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _build_history_text(history: Optional[List[Dict[str, str]]]) -> str:
    """Concatena el historial en formato 'role: content' por línea."""
    if not history:
        return ""
    try:
        return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in history)
    except Exception:
        logger.exception("Failed to build history_text")
        return ""


def _extract_normalized_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    if isinstance(value, dict):
        for key in ("standard_name", "normalized", "label", "value", "text_normalized"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
    return None


def _extract_label(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("label")
    return value


def _extract_confidence(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        confidence = value.get("confidence")
        return confidence if isinstance(confidence, (int, float)) else None
    return value if isinstance(value, (int, float)) else None


def _get_intents_dict(predict_source: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae el dict de intents desde predict_source de forma segura."""
    intents = predict_source.get("intents") if isinstance(predict_source, dict) else {}
    return intents if isinstance(intents, dict) else {}


def _intent_label_from_interpretation(predict_source: Dict[str, Any], key: str) -> Any:
    return _extract_label(_get_intents_dict(predict_source).get(key))


def _intent_payload_from_interpretation(predict_source: Dict[str, Any], key: str) -> Any:
    payload = _get_intents_dict(predict_source).get(key)
    if isinstance(payload, dict):
        return payload
    label = _extract_label(payload)
    return {"label": label} if label is not None else None


def _coerce_entities_payload(entities_payload: Any) -> Dict[str, List[str]]:
    if not isinstance(entities_payload, dict):
        return {}

    coerced: Dict[str, List[str]] = {}
    for key, value in entities_payload.items():
        key_text = str(key).strip()
        if not key_text:
            continue

        if isinstance(value, list):
            values = [str(v).strip() for v in value if v is not None and str(v).strip()]
        elif value is not None:
            text = str(value).strip()
            values = [text] if text else []
        else:
            values = []

        coerced[key_text] = values

    return coerced


# ============================================================
# CLASSIFICATION VIA ENDPOINT
# ============================================================

def _classify_via_endpoint(question: str) -> ClassificationResult:
    """Clasificación usando el endpoint remoto PREDICT_URL."""
    # 1. Llamada al endpoint
    predict_result = post_json(PREDICT_URL, {"text": question}, timeout=PREDICT_TIMEOUT_SECONDS)
    logger.debug("[PREDICT_API] response=%s", predict_result)
    predict_result_dict: Dict[str, Any] = predict_result if isinstance(predict_result, dict) else {}

    # 2. Resolver source e interpretation
    interpretation = predict_result_dict.get("interpretation")
    predict_raw_for_state: Dict[str, Any] = dict(predict_result_dict)
    if isinstance(interpretation, dict):
        predict_source = interpretation
        if not isinstance(predict_raw_for_state.get("interpretation"), dict):
            predict_raw_for_state["interpretation"] = interpretation
    else:
        predict_source = predict_result_dict
        predict_raw_for_state["interpretation"] = predict_source if isinstance(predict_source, dict) else {}

    # 3. Entidades
    entities_raw = predict_source.get("entities")
    if entities_raw is None and isinstance(predict_result_dict.get("entities"), dict):
        entities_raw = predict_result_dict["entities"]
    entities = _coerce_entities_payload(entities_raw)

    # 4. Routing
    routing = predict_result_dict.get("routing")
    if not isinstance(routing, dict):
        routing = predict_source.get("routing")
    routing = routing if isinstance(routing, dict) else {}

    routing_intent = routing.get("intent") if isinstance(routing.get("intent"), dict) else {}
    routing_macro = routing.get("macro") if isinstance(routing.get("macro"), dict) else {}
    routing_context = routing.get("context") if isinstance(routing.get("context"), dict) else {}

    intent = _extract_label(routing_intent)
    macro = _extract_label(routing_macro)
    context = _extract_label(routing_context)
    confidence = _extract_confidence(routing_intent)

    # 5. Clasificadores específicos
    calc_mode_label = _intent_label_from_interpretation(predict_source, "calc_mode")
    req_form_label = _intent_label_from_interpretation(predict_source, "req_form")
    activity_payload = _intent_payload_from_interpretation(predict_source, "activity")
    region_payload = _intent_payload_from_interpretation(predict_source, "region")
    investment_payload = _intent_payload_from_interpretation(predict_source, "investment")
    activity_label = _extract_label(activity_payload)
    region_label = _extract_label(region_payload)
    investment_label = _extract_label(investment_payload)

    logger.info("[CLASSIFIER_API] intent=%s confidence=%s macro=%s context=%s", intent, confidence, macro, context)
    logger.debug("[CLASSIFIER_API] Raw classifier: calc_mode=%s, activity_cls=%s, region_cls=%s, investment_cls=%s, req_form_cls=%s",
                 calc_mode_label, activity_label, region_label, investment_label, req_form_label)
    logger.debug("[CLASSIFIER_API] Raw entities: %s", entities)

    # 6. Normalización
    try:
        normalized = normalize_entities(
            entities=entities,
            calc_mode=calc_mode_label,
            req_form=req_form_label,
            intents={
                "intent": intent,
                "context": context,
                "activity": activity_payload,
                "region": region_payload,
                "investment": investment_payload,
            },
        )
    except Exception as exc:
        logger.exception("[CLASSIFIER_API] Failed to recompute entities_normalized")
        raise RuntimeError("Failed to recompute entities_normalized from /predict response") from exc

    # 7. Coerción post-normalización
    req_form_label = coerce_req_form_from_period_and_frequency(req_form_label, normalized)
    activity_label, region_label, investment_label = coerce_specific_class_labels(
        activity_label=activity_label,
        region_label=region_label,
        investment_label=investment_label,
        normalized_entities=normalized,
    )

    logger.debug("[CLASSIFIER_API] Normalized classifier: calc_mode=%s, activity_cls=%s, region_cls=%s, investment_cls=%s, req_form_cls=%s",
                 calc_mode_label, activity_label, region_label, investment_label, req_form_label)
    logger.debug("[CLASSIFIER_API] Normalized entities: %s", normalized)

    # 8. Armar predict_raw
    interpretation_state = dict(predict_raw_for_state.get("interpretation")) if isinstance(predict_raw_for_state.get("interpretation"), dict) else {}
    interpretation_state["entities"] = entities
    interpretation_state["entities_normalized"] = normalized
    predict_raw_for_state["interpretation"] = interpretation_state

    # 9. Texto
    text = (
        interpretation.get("text") if isinstance(interpretation, dict) else None
    ) or predict_result_dict.get("text") or question

    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        entities=entities,
        normalized=normalized,
        text=text,
        words=predict_source.get("words") or [],
        slot_tags=predict_source.get("slot_tags") or predict_source.get("slots") or [],
        calc_mode=calc_mode_label,
        activity=activity_payload,
        region=region_payload,
        investment=investment_payload,
        req_form=req_form_label,
        macro=macro,
        context=context,
        intent_raw={"routing": routing},
        predict_raw=predict_raw_for_state,
    )

# ============================================================
# PUBLIC API
# ============================================================

def classify_question_with_history(
    question: str, history: Optional[List[Dict[str, str]]]
) -> Tuple[ClassificationResult, str]:
    """Clasificación y construcción de history_text."""
    
    t_start = time.perf_counter()
    logger.info("[CLASSIFICATION] Iniciando clasificación de la consulta | question='%s'", question)
    try:
        classification = _classify_via_endpoint(question)
    except Exception as exc:
        logger.error("[CLASSIFICATION] ERROR al clasificar | time='%.3f' | error=%s", time.perf_counter() - t_start, exc)
        raise

    indicator = _extract_indicator(classification)

    t_elapsed = time.perf_counter() - t_start
    logger.info(
        "[CLASSIFICATION] Clasificación finalizada (%.3fs) | intent=%s | confidence=%.3f | indicator=%s",
        t_elapsed, classification.intent, classification.confidence or 0.0, indicator,
    )

    norm = classification.normalized or {}
    logger.debug(
        "[CLASSIFIER_FILE] MAPPED_PARAMS | indicator=%s seasonality=%s frequency=%s period=%s "
        "calc_mode_cls=%s activity_cls=%s region_cls=%s investment_cls=%s req_form_cls=%s "
        "macro=%s intent=%s context=%s",
        _extract_normalized_scalar(norm.get("indicator")),
        _extract_normalized_scalar(norm.get("seasonality")),
        _extract_normalized_scalar(norm.get("frequency")),
        norm.get("period"),
        classification.calc_mode,
        _extract_label(classification.activity),
        _extract_label(classification.region),
        _extract_label(classification.investment),
        classification.req_form,
        classification.macro,
        classification.intent,
        classification.context,
    )

    history_text = _build_history_text(history)
    return classification, history_text


def _extract_indicator(cls: ClassificationResult) -> Optional[str]:
    """Extrae el indicador normalizado del ClassificationResult."""
    if cls.normalized and isinstance(cls.normalized, dict):
        return _extract_normalized_scalar(cls.normalized.get("indicator"))
    return None


def build_intent_info(cls: Optional[ClassificationResult]) -> Optional[Dict[str, Any]]:
    """Normaliza el resultado del clasificador al intent_info que consume el LLM."""
    if cls is None:
        return None
    try:
        return {
            "intent": cls.intent or "unknown",
            "score": cls.confidence or 0.0,
            "entities": cls.entities or {},
            "normalized": cls.normalized or {},
            "intent_raw": cls.intent_raw or {},
            "predict_raw": cls.predict_raw or {},
            "indicator": _extract_indicator(cls),
            "spans": [],
            "macro": cls.macro,
            "context": cls.context,
            "calc_mode": cls.calc_mode,
            "activity": cls.activity,
            "region": cls.region,
            "investment": cls.investment,
            "req_form": cls.req_form,
            "words": cls.words or [],
            "slot_tags": cls.slot_tags or [],
            "text": cls.text,
        }
    except Exception:
        logger.exception("build_intent_info failed")
        return None
