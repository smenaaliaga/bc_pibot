"""Clasificador modular basado en el flujo legacy de original orchestrator"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.prompts.query_classifier import ClassificationResult, classify_query

logger = logging.getLogger(__name__)

# Flag para usar JointBERT en lugar de LLM
USE_JOINTBERT_CLASSIFIER = os.getenv("USE_JOINTBERT_CLASSIFIER", "false").lower() in ("true", "1", "yes")


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
    Clasificación usando JointBERT en lugar de LLM.
    Mapea los resultados de JointBERT al formato ClassificationResult.
    """
    try:
        from orchestrator import get_predictor
        from orchestrator.prompts.query_classifier import ImacecTree, PibeTree
        
        predictor = get_predictor()
        result = predictor.predict(question)
        
        intent = result.get('intent', 'unknown')
        entities = result.get('entities', {})
        normalized = result.get('normalized', {})
        
        # Mostrar predicción completa
        logger.info(f"[JOINTBERT PREDICTION] question='{question}'")
        logger.info(f"[JOINTBERT PREDICTION] intent={intent}, confidence={result.get('confidence', 0.0):.3f}")
        logger.info(f"[JOINTBERT PREDICTION] Raw entities: {entities}")
        logger.info(f"[JOINTBERT PREDICTION] Normalized entities: {normalized}")
        
        print(f"\n{'='*80}")
        print(f"[JOINTBERT PREDICTION]")
        print(f"Question: {question}")
        print(f"\nIntent: {intent} (confidence: {result.get('confidence', 0.0):.3f})")
        print(f"\nRaw Entities: {entities}")
        print(f"\nNormalized Entities: {normalized}")
        print(f"{'='*80}\n")
        
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
        
        # Determinar data_domain desde el indicador
        data_domain = None
        default_key = None
        imacec_tree = None
        pibe_tree = None
        
        # Si no hay indicador en entidades, buscar en la pregunta original
        if not indicator:
            q_lower = question.lower()
            if 'imacec' in q_lower:
                indicator = 'imacec'
            elif 'pib' in q_lower:
                indicator = 'pib'
        
        if 'imacec' in indicator:
            data_domain = 'IMACEC'
            default_key = 'IMACEC'
            imacec_tree = ImacecTree()
        elif 'pib' in indicator:
            # Detectar si es regional
            region = entities.get('region', '')
            q_lower = question.lower()
            if region or 'regional' in q_lower or 'región' in q_lower:
                data_domain = 'PIB_REGIONAL'
                default_key = 'PIB_REGIONAL'
                pibe_tree = PibeTree(region=region if region else None)
            else:
                data_domain = 'PIB'
                default_key = 'PIB_TOTAL'
                pibe_tree = PibeTree()
        
        # Detectar si es genérico (basado en período)
        is_generic = not period or any(
            word in str(period).lower() 
            for word in ['ultimo', 'último', 'actual', 'reciente']
        )
        
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
            entities=entities,
            normalized=normalized,
        )
        
    except Exception as e:
        logger.warning(f"JointBERT classification failed: {e}, falling back to LLM")
        return classify_query(question)


def classify_question_with_history(
    question: str, history: Optional[List[Dict[str, str]]]
) -> Tuple[ClassificationResult, str]:
    """Replica la fase C1 del original orchestrator: clasifica y construye history_text."""
    t_start = time.perf_counter()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Decidir qué clasificador usar
    if USE_JOINTBERT_CLASSIFIER:
        logger.info("[FASE] start: Fase C1: Clasificación de consulta | classifier='JointBERT'")
        try:
            classification = _classify_with_jointbert(question)
        except Exception as exc:
            t_end = time.perf_counter()
            logger.error(
                "[FASE] error: Fase C1: Clasificación con JointBERT (%.3fs) | question='%s' | error=%s",
                t_end - t_start,
                question,
                exc,
            )
            raise
    else:
        logger.info("[FASE] start: Fase C1: Clasificación de consulta | classifier='LLM' | model='%s'", model)
        try:
            classification = classify_query(question)
        except Exception as exc:
            t_end = time.perf_counter()
            logger.error(
                "[FASE] error: Fase C1: Clasificación de consulta (%.3fs) | question='%s' | error=%s",
                t_end - t_start,
                question,
                exc,
            )
            raise
    
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
    # Forzar log también en root para trazabilidad en el archivo único de sesión
    try:
        import logging as _logging

        _logging.getLogger().info(
            "[CLASSIFIER] query_type=%s data_domain=%s is_generic=%s default_key=%s error=%s",
            *summary,
        )
    except Exception:
        pass
    # Emisión a stdout para depuración rápida en entorno interactivo/Streamlit logs
    try:
        print(
            "[CLASSIFIER_RETURN] query_type=%s data_domain=%s is_generic=%s default_key=%s error=%s"
            % summary
        )
    except Exception:
        pass
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


def classify_question(question: str) -> ClassificationResult:
    """Convenience wrapper cuando no se necesita history_text."""
    cls, _ = classify_question_with_history(question, history=None)
    return cls


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
