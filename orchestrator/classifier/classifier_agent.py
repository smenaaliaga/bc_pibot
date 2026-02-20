"""Clasificador modular basado en el flujo legacy de original orchestrator"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.classifier.entity_normalizer import normalize_entities
from orchestrator.utils.http_client import post_json
from config import PREDICT_URL, INTENT_CLASSIFIER_URL
from registry import get_intent_router, get_series_interpreter

logger = logging.getLogger(__name__)

# Timeouts
PREDICT_TIMEOUT_SECONDS = float(os.getenv("PREDICT_TIMEOUT_SECONDS", "10"))


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


def _flatten_api_entities(entities: Any) -> Dict[str, str]:
    if not entities or not isinstance(entities, dict):
        return {}
    flattened: Dict[str, str] = {}
    for key, value in entities.items():
        if isinstance(value, list):
            first = next((v for v in value if isinstance(v, str) and v.strip()), None)
            if first:
                flattened[str(key)] = first
        elif isinstance(value, str) and value.strip():
            flattened[str(key)] = value
    return flattened


def _get_first_entity_value(entities: Any, key: str) -> str:
    if not entities or not isinstance(entities, dict):
        return ""
    value = entities.get(key)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
        return ""
    if isinstance(value, str):
        return value
    return ""


def _classify_with_jointbert(question: str) -> ClassificationResult:
    """
    Clasificación usando APIs remotas.
    """
    predict_payload = {"text": question}
    predict_result = post_json(PREDICT_URL, predict_payload, timeout=PREDICT_TIMEOUT_SECONDS)
    logger.debug("[PREDICT_API] response=%s", predict_result)
    predict_result_dict: Dict[str, Any] = predict_result if isinstance(predict_result, dict) else {}
    interpretation = predict_result_dict.get("interpretation")
    if isinstance(interpretation, dict):
        predict_raw = interpretation
        predict_source = interpretation
    else:
        predict_raw = predict_result_dict if isinstance(predict_result, dict) else {"raw": predict_result}
        predict_source = predict_result_dict
    predict_result = predict_result_dict

    intent_payload = {"text": question}
    intent_result = post_json(INTENT_CLASSIFIER_URL, intent_payload, timeout=PREDICT_TIMEOUT_SECONDS)
    logger.debug("[INTENT_API] response=%s", intent_result)
    intent_raw: Dict[str, Any] = intent_result if isinstance(intent_result, dict) else {"raw": intent_result}
    if not isinstance(intent_result, dict):
        intent_result = {}

    entities_api = predict_source.get("entities", {}) or {}
    entities_flat = _flatten_api_entities(entities_api)
    
    # Aplicar normalización de entidades
    normalized = normalize_entities(entities_flat)
     
    # Extraer intentención y entidades
    intent = intent_result.get("intent")
    entities = entities_api
    macro = intent_result.get("macro")
    context = intent_result.get("context")
    confidence = intent_result.get("confidence")
    
    # Mostrar predicción completa
    logger.info("[CLASSIFIER_API] intent=%s confidence=%s macro=%s context=%s", intent, confidence, macro, context)
    logger.info("[CLASSIFIER_API] Raw entities: %s", entities)
    logger.info("[CLASSIFIER_API] Normalized entities: %s", normalized)
    
    # Usar entidades normalizadas si están disponibles, sino usar raw
    indicator_norm = normalized.get('indicator', {})
    
    # Extraer indicador (priorizar normalizado)
    if indicator_norm and indicator_norm.get('standard_name'):
        indicator = indicator_norm['standard_name'].lower()
    else:
        indicator = _get_first_entity_value(entities, "indicator").lower()
    
    text = (
        interpretation.get("text")
        if isinstance(interpretation, dict)
        else None
    ) or predict_result.get("text") or question

    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        entities=entities,
        normalized=normalized,
        text=text,
        words=predict_source.get("words") or [],
        slot_tags=predict_source.get("slot_tags") or predict_source.get("slots") or [],
        calc_mode=predict_source.get("calc_mode"),
        activity=predict_source.get("activity"),
        region=predict_source.get("region"),
        investment=predict_source.get("investment"),
        req_form=predict_source.get("req_form"),
        macro=macro,
        context=context,
        intent_raw=intent_raw,
        predict_raw=predict_raw,
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
    indicator = None
    if classification.normalized and isinstance(classification.normalized, dict):
        indicator_data = classification.normalized.get('indicator', {})
        if isinstance(indicator_data, dict):
            indicator = indicator_data.get('standard_name') or indicator_data.get('normalized')
    
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
        with open(path, "a", encoding="utf-8") as f:
            f.write(
                "[CLASSIFIER_FILE] intent=%s confidence=%.3f indicator=%s\n"
                % (classification.intent, classification.confidence or 0.0, indicator)
            )
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
        indicator = None
        if cls.normalized and isinstance(cls.normalized, dict):
            indicator_data = cls.normalized.get('indicator', {})
            if isinstance(indicator_data, dict):
                indicator = indicator_data.get('standard_name') or indicator_data.get('normalized')
        
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
