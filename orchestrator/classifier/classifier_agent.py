"""Clasificador modular basado en el flujo legacy de original orchestrator"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# JointBERT imports
from orchestrator.classifier.joint_bert_classifier import get_predictor
from orchestrator.classifier.entity_normalizer import normalize_entities

logger = logging.getLogger(__name__)

# MODEL
MODEL_JOINTBERT_NAME = os.getenv("JOINT_BERT_MODEL_NAME", "pibot_model_beto")


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
    
    # Detectar small talk
    _SMALL_TALK_GREETINGS = ("hola", "holi", "buenas", "buenos dias", "buenas tardes", "buenas noches", "saludos", "que tal")
    _SMALL_TALK_DATA_TOKENS = ("imacec", "pib", "inflacion", "dato", "serie", "valor", "porcentaje", "indicador", "consulta")
    
    def _looks_like_small_talk(q: str, ind: str) -> bool:
        ql = q.lower().strip()
        for greeting in _SMALL_TALK_GREETINGS:
            if ql.startswith(greeting):
                has_data_token = any(tok in ql for tok in _SMALL_TALK_DATA_TOKENS)
                return not has_data_token
        return False

    # Override para small talk
    if _looks_like_small_talk(question, indicator):
        logger.info("[JOINTBERT] Greeting-like query detected, forcing 'greeting' intent")
        intent = 'greeting'
        entities = {}
        normalized = {}

    logger.info(
        "[JOINTBERT FINAL] intent=%s confidence=%.3f indicator=%s",
        intent, result.get('confidence', 0.0), indicator
    )
    
    return ClassificationResult(
        intent=intent,
        confidence=result.get('confidence', 0.0),
        entities=entities,
        normalized=normalized,
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
    # Extraer indicador normalizado
    indicator = None
    if classification.normalized and isinstance(classification.normalized, dict):
        indicator_data = classification.normalized.get('indicator', {})
        if isinstance(indicator_data, dict):
            indicator = indicator_data.get('standard_name') or indicator_data.get('normalized')
    
    logger.info(
        "[FASE] end: Fase C1: Clasificación de consulta (%.3fs) | intent=%s | confidence=%.3f | indicator=%s",
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
            "indicator": indicator,
            "spans": [],
        }
    except Exception:
        logger.exception("build_intent_info failed")
        return None
