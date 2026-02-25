"""Clasificador modular basado en el flujo legacy de original orchestrator"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.utils.http_client import post_json
from config import PREDICT_URL, INTENT_CLASSIFIER_URL
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
        logger.warning("IntentRouter unavailable; returning None")
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
        logger.warning("SeriesInterpreter unavailable; returning None")
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


def _classify_with_jointbert(question: str) -> ClassificationResult:
    """
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

    intent_payload = {"text": question}
    intent_result: Dict[str, Any]
    intent_raw: Dict[str, Any]
    try:
        intent_result = post_json(INTENT_CLASSIFIER_URL, intent_payload, timeout=PREDICT_TIMEOUT_SECONDS)
        logger.debug("[INTENT_API] response=%s", intent_result)
        intent_raw = intent_result if isinstance(intent_result, dict) else {"raw": intent_result}
        if not isinstance(intent_result, dict):
            intent_result = {}
    except Exception as exc:
        logger.warning("[INTENT_API] fallback to routing data | error=%s", exc)
        intent_result = {}
        intent_raw = {"error": str(exc)}

    entities_api = predict_source.get("entities", {}) or {}
    normalized_api = predict_source.get("entities_normalized")
    if not isinstance(normalized_api, dict):
        normalized_api = predict_result_dict.get("entities_normalized")
    normalized = normalized_api if isinstance(normalized_api, dict) else {}
    routing_payload = predict_source.get("routing")
    if not isinstance(routing_payload, dict):
        routing_payload = predict_result_dict.get("routing")
    routing_payload = routing_payload if isinstance(routing_payload, dict) else {}
    routing_intent = routing_payload.get("intent") if isinstance(routing_payload.get("intent"), dict) else {}
    routing_macro = routing_payload.get("macro") if isinstance(routing_payload.get("macro"), dict) else {}
    routing_context = routing_payload.get("context") if isinstance(routing_payload.get("context"), dict) else {}
     
    # Extraer intentención y entidades
    intent = _coalesce_defined(_extract_label(intent_result.get("intent")), routing_intent.get("label"))
    entities = entities_api
    macro = _coalesce_defined(_extract_label(intent_result.get("macro")), routing_macro.get("label"))
    context = _coalesce_defined(_extract_label(intent_result.get("context")), routing_context.get("label"))
    confidence = _extract_confidence(intent_result.get("intent"))
    if confidence is None:
        confidence = _extract_confidence(intent_result.get("confidence"))
    if confidence is None:
        confidence = routing_intent.get("confidence")

    # Mostrar predicción completa
    logger.info("[CLASSIFIER_API] intent=%s confidence=%s macro=%s context=%s", intent, confidence, macro, context)
    logger.info("[CLASSIFIER_API] Raw entities: %s", entities)
    logger.info("[CLASSIFIER_API] Normalized entities: %s", normalized)
    
    text = (
        interpretation.get("text")
        if isinstance(interpretation, dict)
        else None
    ) or predict_result.get("text") or question

    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        entities=normalized,
        normalized=normalized,
        text=text,
        words=predict_source.get("words") or [],
        slot_tags=predict_source.get("slot_tags") or predict_source.get("slots") or [],
        calc_mode=_intent_label_from_interpretation(predict_source, "calc_mode"),
        activity=_intent_label_from_interpretation(predict_source, "activity"),
        region=_intent_label_from_interpretation(predict_source, "region"),
        investment=_intent_label_from_interpretation(predict_source, "investment"),
        req_form=_intent_label_from_interpretation(predict_source, "req_form"),
        macro=macro,
        context=context,
        intent_raw=intent_raw,
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
    
    
    # Fallback: asegurarse de que quede registrado en el archivo de log configurado por RUN_MAIN_LOG,
    # pero sin truncar ni reescribir; solo append.
    try:
        logs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        fixed = os.getenv("RUN_MAIN_LOG", "").strip() or "run_main.log"
        path = os.path.join(logs_dir, fixed)
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
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                "[CLASSIFIER_FILE] intent=%s confidence=%.3f indicator=%s\n"
                % (classification.intent, classification.confidence or 0.0, indicator)
            )
            f.write("[CLASSIFIER_FILE] PREDICT_RAW=%s\n" % (predict_raw,))
            f.write("[CLASSIFIER_FILE] MAPPED_PARAMS=\n")
            f.write("Key                                  | Status     | Value\n")
            f.write("-------------------------------------+------------+--------------------------\n")
            for key in sorted(mapped.keys()):
                status = statuses.get(key, "MISSING")
                raw_value = mapped.get(key)
                value = "(empty)" if raw_value in (None, "", [], {}) else str(raw_value)
                f.write(f"{key:<37} | {status:<10} | {value}\n")
    except Exception:
        pass
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
