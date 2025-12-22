"""Clasificador modular basado en el flujo legacy de original orchestrator"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

# JointBERT imports
from orchestrator import get_predictor
from orchestrator.classifier.entity_normalizer import normalize_entities

logger = logging.getLogger(__name__)

# MODEL
MODEL_JOINTBERT_NAME = os.getenv("JOINT_BERT_MODEL_NAME", "pibot_model_beto")

# Data classes
from orchestrator.classifier.classifier_dataclass import ClassificationResult

def _build_history_text(history: Optional[List[Dict[str, str]]]) -> str:
    """Concatena el historial en formato 'role: content' por línea."""
    if not history:
        return ""
    try:
        return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in history)
    except Exception:
        logger.exception("Failed to build history_text")
        return ""


def _classify_with_jointbert(question: str) -> ClassificationResult:
    """
    Clasificación usando JointBERT
    """
    # Obtener predictor JointBERT
    predictor = get_predictor()
    result = predictor.predict(question)
    
    # Aplicar normalización de entidades
    normalized = normalize_entities(result.get('entities', {}))
    
    # Extraer intentención y entidades
    intent = result.get('intent', 'unknown')
    entities = result.get('entities', {})
    
    # Mostrar predicción completa
    logger.info(f"[JOINTBERT PREDICTION] question='{question}'")
    logger.info(f"[JOINTBERT PREDICTION] intent={intent}, confidence={result.get('confidence', 0.0):.3f}")
    logger.info(f"[JOINTBERT PREDICTION] Raw entities: {entities}")
    logger.info(f"[JOINTBERT PREDICTION] Normalized entities: {normalized}")
    
    # Si no hay entidades, tratar como consulta metodológica
    if not entities or all(not v for v in entities.values()):
        logger.info("[JOINTBERT] No entities detected, treating as METHODOLOGICAL query")
        intent = 'methodology'
    
    # Usar entidades normalizadas si están disponibles, sino usar raw
    indicator_norm = normalized.get('indicator', {})
    period_norm = normalized.get('period', {})
    
    # Extraer indicador (priorizar normalizado)
    if indicator_norm and indicator_norm.get('standard_name'):
        indicator = indicator_norm['standard_name'].lower()
    else:
        indicator = entities.get('indicator', '').lower()
    
    # Extraer período (priorizar normalizado)
    if period_norm:
        period = str(period_norm)  # La normalización retorna un dict estructurado
    else:
        period = entities.get('period', '')
    
    # Mapeo de intenciones JointBERT a query_type
    query_type_map = {
        'value': 'DATA',
        'data': 'DATA',
        'last': 'DATA',
        'table': 'DATA',
        'methodology': 'METHODOLOGICAL',
        'definition': 'METHODOLOGICAL',
    }
    query_type = query_type_map.get(intent.lower(), None)
    
    entities_out = entities
    normalized_out = normalized

    if _looks_like_small_talk(question, indicator):
        logger.info("[JOINTBERT] Greeting-like query detected, forcing METHODOLOGICAL flow")
        query_type = 'METHODOLOGICAL'
        data_domain = None
        default_key = None
        is_generic = True
        entities_out = {}
        normalized_out = {}

    logger.info(
        "[JOINTBERT MAPPED] query_type=%s data_domain=%s is_generic=%s default_key=%s",
        query_type, data_domain, is_generic, default_key
    )
    
    return ClassificationResult(
        query_type=query_type,
        data_domain=data_domain,
        is_generic=is_generic,
        default_key=default_key,
        imacec=imacec_tree,
        pibe=pibe_tree,
        # Info adicional de JointBERT
        intent=intent,
        confidence=result.get('confidence', 0.0),
        entities=entities_out,
        normalized=normalized_out,
    )


def classify_question_with_history(
    question: str, history: Optional[List[Dict[str, str]]]
) -> Tuple[ClassificationResult, str]:
    """Clasificación y construcción de history_text."""
    
    t_start = time.perf_counter()
    logger.info("[CLASSIFICATION] Iniciando clasificación de la consulta | question='%s' | model='%s'", question, MODEL_JOINTBERT_NAME)
    try:
        classification = _classify_with_jointbert(question)
    except Exception as exc:
        t_end = time.perf_counter()
        logger.error("[CLASSIFICATION] ERROR al clasificar | time='%s' | error=%s", t_end - t_start, exc)
        raise
    t_end = time.perf_counter()
    logger.error("[CLASSIFICATION] Clasificación finalizada | time='%s'", t_end - t_start)
    
    # DEBUG
    try:
        raw_payload = classification.__dict__ if hasattr(classification, "__dict__") else classification
        logger.debug("[CLASSIFICATION] classification_payload=%s", raw_payload)
    except Exception:
        logger.exception("[CLASSIFICATION] Failed to log classification payload")
    
    t_end = time.perf_counter()
    summary = (
        classification.query_type,
        classification.data_domain,
        classification.is_generic,
        classification.default_key,
        getattr(classification, "error", None),
    )
    logger.info(
        "[FASE] end: Fase C1: Clasificación de consulta (%.3fs) | query_type=%s | data_domain=%s | "
        "is_generic=%s | default_key=%s | error=%s",
        t_end - t_start,
        *summary,
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
                "[CLASSIFIER_FILE] query_type=%s data_domain=%s is_generic=%s default_key=%s error=%s\n"
                % summary
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
        return {
            "intent": cls.query_type or "unknown",
            "score": 1.0 if cls.query_type else 0.0,
            "entities": {
                "data_domain": cls.data_domain,
                "is_generic": cls.is_generic,
                "default_key": cls.default_key,
                "imacec": cls.imacec.__dict__ if getattr(cls, "imacec", None) else None,
                "pibe": cls.pibe.__dict__ if getattr(cls, "pibe", None) else None,
                "intent_frequency_change": getattr(cls, "intent_frequency_change", None),
            },
            "spans": [],  # placeholder para el modelo BIO
        }
    except Exception:
        logger.exception("build_intent_info failed")
        return None
