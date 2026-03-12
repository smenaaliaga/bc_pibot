"""Clasificador modular basado en el flujo legacy de original orchestrator"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.normalizer.normalizer import (
    coerce_req_form_from_period_and_frequency,
    normalize_entities,
)
from orchestrator.utils.http_client import post_json
from orchestrator.utils.run_detail_log import append_run_detail
from config import PREDICT_URL
from registry import get_intent_router, get_series_interpreter

logger = logging.getLogger(__name__)

# Timeouts
PREDICT_TIMEOUT_SECONDS = float(os.getenv("PREDICT_TIMEOUT_SECONDS", "10"))


def _use_local_jointbert() -> bool:
    return os.getenv("USE_JOINTBERT_CLASSIFIER", "false").lower() in {"1", "true", "yes", "on"}


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
# SHARED MODEL ACCESSORS
# ============================================================

def get_router_model(path: Optional[str] = None):
    """Singleton IntentRouter (cached en registry.py)."""
    return get_intent_router(path)


def get_series_interpreter_model(path: Optional[str] = None):
    """Singleton SeriesInterpreter (cached en registry.py)."""
    return get_series_interpreter(path)


def load_intent_router(path: Optional[str] = None):
    """Alias simple para obtener el IntentRouter compartido."""
    return get_router_model(path)


def load_series_interpreter(path: Optional[str] = None):
    """Alias simple para obtener el SeriesInterpreter compartido."""
    return get_series_interpreter_model(path)


def predict_with_router(query: str) -> Any:
    """
    Predice la intención usando el IntentRouter compartido.

    Args:
        query: Texto a clasificar.

    Returns:
        Resultado retornado por el IntentRouter.
    """
    router = load_intent_router()
    if router is None:
        logger.debug("IntentRouter unavailable; returning None")
        return None
    return router.predict(query)


def predict_with_interpreter(query: str) -> Any:
    """
    Predice interpretaciones de series usando el SeriesInterpreter compartido.

    Args:
        query: Texto a interpretar.

    Returns:
        Resultado retornado por el SeriesInterpreter.
    """
    interpreter = load_series_interpreter()
    if interpreter is None:
        logger.debug("SeriesInterpreter unavailable; returning None")
        return None
    return interpreter.predict(query)


def predict_with_router_and_interpreter(query: str) -> Dict[str, Any]:
    """Ejecuta predicción usando IntentRouter y SeriesInterpreter con caché global.

    Args:
        query: Texto a clasificar/interpetar.

    Returns:
        Dict con llaves `router` y `interpreter` conteniendo las respuestas de cada modelo.
    """
    router_out = predict_with_router(query)
    interpreter_out = predict_with_interpreter(query)
    return {"router": router_out, "interpreter": interpreter_out}


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


def _coalesce_defined(primary: Any, fallback: Any) -> Any:
    if primary is None:
        return fallback
    if isinstance(primary, str) and not primary.strip():
        return fallback
    return primary


def _intent_label_from_interpretation(predict_source: Dict[str, Any], key: str) -> Any:
    intents = predict_source.get("intents") if isinstance(predict_source, dict) else {}
    intents = intents if isinstance(intents, dict) else {}
    payload = intents.get(key)
    return _extract_label(payload)


def _coerce_entities_payload(entities_payload: Any) -> Dict[str, List[str]]:
    if not isinstance(entities_payload, dict):
        return {}

    coerced: Dict[str, List[str]] = {}
    for key, value in entities_payload.items():
        key_text = str(key).strip()
        if not key_text:
            continue

        values: List[str] = []
        if isinstance(value, list):
            for item in value:
                if item is None:
                    continue
                item_text = str(item).strip()
                if item_text:
                    values.append(item_text)
        elif value is not None:
            value_text = str(value).strip()
            if value_text:
                values.append(value_text)

        coerced[key_text] = values

    return coerced


def _format_mapped_params_table(mapped: Dict[str, Any], statuses: Dict[str, str]) -> List[str]:
    rows: List[Tuple[str, str, str]] = []
    for key in sorted(mapped.keys()):
        status = str(statuses.get(key, "MISSING"))
        raw_value = mapped.get(key)
        value = "(empty)" if raw_value in (None, "", [], {}) else str(raw_value)
        rows.append((str(key), status, value))

    key_width = max(3, max((len(row[0]) for row in rows), default=0))
    status_width = max(6, max((len(row[1]) for row in rows), default=0))
    lines = [
        f"{'Key':<{key_width}} | {'Status':<{status_width}} | Value",
        f"{'-' * key_width}-+-{'-' * status_width}-+-{'-' * 5}",
    ]
    for key, status, value in rows:
        lines.append(f"{key:<{key_width}} | {status:<{status_width}} | {value}")
    return lines


def _format_classification_header_table(
    *,
    intent: Any,
    confidence: Any,
    indicator: Any,
    mapped: Dict[str, Any],
) -> List[str]:
    def _display(value: Any) -> str:
        if value in (None, "", [], {}):
            return "(empty)"
        return str(value)

    confidence_display = "(empty)"
    if isinstance(confidence, (int, float)):
        confidence_display = f"{float(confidence):.3f}"
    elif confidence not in (None, ""):
        confidence_display = str(confidence)

    columns: List[Tuple[str, str]] = [
        ("Intent", _display(intent)),
        ("Confidence", confidence_display),
        ("Indicator", _display(indicator)),
        ("Macro", _display(mapped.get("macro_cls"))),
        ("Context", _display(mapped.get("context_cls"))),
        ("CalcMode", _display(mapped.get("calc_mode_cls"))),
        ("ReqForm", _display(mapped.get("req_form_cls"))),
        ("Frequency", _display(mapped.get("frequency"))),
        ("Seasonality", _display(mapped.get("seasonality"))),
        ("Activity", _display(mapped.get("activity_cls"))),
        ("Region", _display(mapped.get("region_cls"))),
        ("Investment", _display(mapped.get("investment_cls"))),
    ]

    widths = [max(len(header), len(value)) for header, value in columns]
    header_line = " | ".join(
        f"{header:<{width}}" for (header, _), width in zip(columns, widths)
    )
    separator = "-+-".join("-" * width for width in widths)
    value_line = " | ".join(
        f"{value:<{width}}" for (_, value), width in zip(columns, widths)
    )
    return [header_line, separator, value_line]


def _classify_with_jointbert(question: str) -> ClassificationResult:
    """
    Clasificación usando APIs remotas (o modelo local si está habilitado).
    Clasificación usando APIs remotas (o modelo local si está habilitado).
    """
    if _use_local_jointbert():
        logger.warning("USE_JOINTBERT_CLASSIFIER enabled but local classifier is unavailable; using API flow")

    predict_payload = {"text": question}
    predict_result = post_json(PREDICT_URL, predict_payload, timeout=PREDICT_TIMEOUT_SECONDS)
    logger.debug("[PREDICT_API] response=%s", predict_result)
    predict_result_dict: Dict[str, Any] = predict_result if isinstance(predict_result, dict) else {}
    interpretation = predict_result_dict.get("interpretation")
    predict_raw_for_state: Dict[str, Any] = dict(predict_result_dict) if isinstance(predict_result_dict, dict) else {}
    if isinstance(interpretation, dict):
        predict_source = interpretation
        if not isinstance(predict_raw_for_state.get("interpretation"), dict):
            predict_raw_for_state["interpretation"] = interpretation
    else:
        predict_source = predict_result_dict
        fallback_interpretation = predict_source if isinstance(predict_source, dict) else {}
        predict_raw_for_state = {
            **predict_raw_for_state,
            "interpretation": fallback_interpretation,
        }
    predict_result = predict_result_dict

    entities_api_raw = predict_source.get("entities")
    if entities_api_raw is None and isinstance(predict_result_dict.get("entities"), dict):
        entities_api_raw = predict_result_dict.get("entities")
    entities_api = _coerce_entities_payload(entities_api_raw)

    routing_payload = predict_result_dict.get("routing")
    if not isinstance(routing_payload, dict):
        routing_payload = predict_source.get("routing")
    routing_payload = routing_payload if isinstance(routing_payload, dict) else {}
    routing_intent = routing_payload.get("intent") if isinstance(routing_payload.get("intent"), dict) else {}
    routing_macro = routing_payload.get("macro") if isinstance(routing_payload.get("macro"), dict) else {}
    routing_context = routing_payload.get("context") if isinstance(routing_payload.get("context"), dict) else {}
    routing_intent_label = _extract_label(routing_intent)
    routing_context_label = _extract_label(routing_context)

    calc_mode_label = _intent_label_from_interpretation(predict_source, "calc_mode")
    req_form_label = _intent_label_from_interpretation(predict_source, "req_form")
    activity_label = _intent_label_from_interpretation(predict_source, "activity")
    region_label = _intent_label_from_interpretation(predict_source, "region")
    investment_label = _intent_label_from_interpretation(predict_source, "investment")
    try:
        normalized = normalize_entities(
            entities=entities_api,
            calc_mode=calc_mode_label,
            req_form=req_form_label,
            intents={
                "intent": routing_intent_label,
                "context": routing_context_label,
                "activity": activity_label,
                "region": region_label,
                "investment": investment_label,
            },
        )
    except Exception as exc:
        logger.exception("[CLASSIFIER_API] Failed to recompute entities_normalized")
        raise RuntimeError("Failed to recompute entities_normalized from /predict response") from exc

    req_form_label = coerce_req_form_from_period_and_frequency(req_form_label, normalized)

    interpretation_state = predict_raw_for_state.get("interpretation")
    if isinstance(interpretation_state, dict):
        interpretation_state = dict(interpretation_state)
    else:
        interpretation_state = {}
    interpretation_state["entities"] = entities_api
    interpretation_state["entities_normalized"] = normalized
    predict_raw_for_state["interpretation"] = interpretation_state

    # Extraer intención y entidades desde el payload unificado de PREDICT_URL
    intent = routing_intent_label
    entities = entities_api
    macro = _extract_label(routing_macro)
    context = routing_context_label
    confidence = _extract_confidence(routing_intent)

    # Mostrar predicción completa
    logger.info("[CLASSIFIER_API] intent=%s confidence=%s macro=%s context=%s", intent, confidence, macro, context)
    logger.debug("[CLASSIFIER_API] Raw entities: %s", entities)
    logger.debug("[CLASSIFIER_API] Normalized entities: %s", normalized)
    
    text = (
        interpretation.get("text")
        if isinstance(interpretation, dict)
        else None
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
        activity=activity_label,
        region=region_label,
        investment=investment_label,
        req_form=req_form_label,
        macro=macro,
        context=context,
        intent_raw={"routing": routing_payload},
        predict_raw=predict_raw_for_state,
    )

def classify_question_with_history(
    question: str, history: Optional[List[Dict[str, str]]]
) -> Tuple[ClassificationResult, str]:
    """Clasificación y construcción de history_text."""
    
    t_start = time.perf_counter()
    logger.info("[CLASSIFICATION] Iniciando clasificación de la consulta | question='%s'", question)
    try:
        classification = _classify_with_jointbert(question)
    except Exception as exc:
        t_end = time.perf_counter()
        logger.error("[CLASSIFICATION] ERROR al clasificar | time='%s' | error=%s", t_end - t_start, exc)
        raise
    
    # Extraer indicador normalizado
    indicator: Optional[str] = None
    if classification.normalized and isinstance(classification.normalized, dict):
        indicator = _extract_normalized_scalar(classification.normalized.get('indicator'))
    
    t_end = time.perf_counter()
    logger.info(
        "[CLASSIFICATION] Clasificación finalizada (%.3fs) | intent=%s | confidence=%.3f | indicator=%s",
        t_end - t_start,
        classification.intent,
        classification.confidence or 0.0,
        indicator,
    )
    
    
    # Registrar detalle de clasificación usando el logger principal (sin escrituras manuales a run_main.log).
    try:
        predict_raw = classification.predict_raw or {}
        interpretation = predict_raw.get("interpretation") if isinstance(predict_raw.get("interpretation"), dict) else predict_raw
        interpretation = interpretation if isinstance(interpretation, dict) else {}
        interpretation_intents = interpretation.get("intents") or {}
        entities_normalized = interpretation.get("entities_normalized") or {}
        routing = predict_raw.get("routing") or {}

        def _first_or_none(value: Any) -> Any:
            if isinstance(value, list):
                return next((item for item in value if item not in (None, "")), None)
            return value

        def _intent_label(key: str) -> Any:
            intent_payload = interpretation_intents.get(key)
            if isinstance(intent_payload, dict):
                return intent_payload.get("label")
            return intent_payload

        mapped = {
            "indicator": _first_or_none(entities_normalized.get("indicator"))
            or _first_or_none((classification.normalized or {}).get("indicator")),
            "seasonality": _first_or_none(entities_normalized.get("seasonality"))
            or _first_or_none((classification.normalized or {}).get("seasonality")),
            "frequency": _first_or_none(entities_normalized.get("frequency"))
            or _first_or_none((classification.normalized or {}).get("frequency")),
            "period": entities_normalized.get("period")
            or (classification.normalized or {}).get("period"),
            "calc_mode_cls": _intent_label("calc_mode"),
            "activity_cls": _intent_label("activity"),
            "region_cls": _intent_label("region"),
            "investment_cls": _intent_label("investment"),
            "req_form_cls": _intent_label("req_form"),
            "macro_cls": (routing.get("macro") or {}).get("label"),
            "intent_cls": (routing.get("intent") or {}).get("label"),
            "context_cls": (routing.get("context") or {}).get("label"),
        }
        statuses = {
            key: "PRESENT" if value not in (None, "", [], {}) else "MISSING"
            for key, value in mapped.items()
        }
        summary = {
            "intent": classification.intent,
            "confidence": classification.confidence,
            "indicator": indicator,
            "mapped": mapped,
            "statuses": statuses,
            "normalized": classification.normalized or {},
            "predict_raw": predict_raw,
        }
        logger.info(
            "[CLASSIFIER_TRACE] intent=%s confidence=%.3f indicator=%s",
            classification.intent,
            classification.confidence or 0.0,
            indicator,
        )
        logger.info("[CLASSIFIER_TRACE] CLASSIFICATION_HEADER_TABLE")
        for line in _format_classification_header_table(
            intent=classification.intent,
            confidence=classification.confidence,
            indicator=indicator,
            mapped=mapped,
        ):
            logger.info("[CLASSIFIER_TRACE] %s", line)
        logger.info("[CLASSIFIER_TRACE] MAPPED_PARAMS_TABLE")
        for line in _format_mapped_params_table(mapped, statuses):
            logger.info("[CLASSIFIER_TRACE] %s", line)
        append_run_detail("classifier_classification", summary)
    except Exception as exc:
        logger.warning("[CLASSIFIER_TRACE] no se pudo registrar tabla de parametros: %s", exc)
    history_text = _build_history_text(history)
    return classification, history_text


def build_intent_info(cls: Optional[ClassificationResult]) -> Optional[Dict[str, Any]]:
    """Normaliza el resultado del clasificador al intent_info que consume el LLM."""
    if cls is None:
        return None
    try:
        # Extraer indicador desde normalized
        indicator: Optional[str] = None
        if cls.normalized and isinstance(cls.normalized, dict):
            indicator = _extract_normalized_scalar(cls.normalized.get('indicator'))
        
        return {
            "intent": cls.intent or "unknown",
            "score": cls.confidence or 0.0,
            "entities": cls.entities or {},
            "normalized": cls.normalized or {},
            "intent_raw": cls.intent_raw or {},
            "predict_raw": cls.predict_raw or {},
            "indicator": indicator,
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
