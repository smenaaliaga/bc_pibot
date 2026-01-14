"""
IntentRouter - pibot-intent-router

Clasifica la intención de la consulta del usuario.

Cabezas del modelo:
- intent: "value" | "methodology"
- context_mode: "standalone" | "followup"

Para usar un modelo real entrenado:
1. Colocar el modelo en esta carpeta (e.g., checkpoint/, model.safetensors, config.json)
2. Descomentar y adaptar el método load_model()
3. El modelo debe retornar un dict con keys: "intent" y "context_mode"
"""

from typing import Optional, Union, Dict, Any
import os
import re
import json
import joblib
from ..base import BaseClassifier, IntentRouterOutput, LabeledScore


class IntentRouter(BaseClassifier):
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Ruta al modelo entrenado. Si es None, usa clasificador heurístico.
        """
        self.model = None
        self.model_path = model_path
        
        if model_path:
            self.model = self.load_model(model_path)
    
    def load_model(self, model_path: str) -> Any:
        """
        Carga el modelo real desde disco.
        
        Arquitectura:
        - Embedder: sentence-transformers/all-MiniLM-L6-v2 (384-dim, normalizado L2)
        - Intent classifier: LogisticRegression (scikit-learn)
        - Context classifier: LogisticRegression (scikit-learn)
        
        Args:
            model_path: Ruta al modelo (absoluta o relativa al directorio de este archivo)
        
        Artefactos:
        - intent_clf.joblib: Clasificador de intención (value vs methodology)
        - context_clf.joblib: Clasificador de contexto (standalone vs followup)
        - label_maps.json: Mapeo de índices a etiquetas para ambos clasificadores
        - thresholds.json: (Opcional) Umbrales de confianza
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers no instalado. Ejecuta: "
                "pip install sentence-transformers"
            )
        
        # Resolver ruta del modelo
        from pathlib import Path
        model_dir = Path(model_path)
        if not model_dir.is_absolute():
            # Si es relativa, resolver desde el directorio de este archivo
            base_dir = Path(__file__).parent
            model_dir = base_dir / model_path
        
        # Cargar embedder
        embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Cargar clasificadores
        intent_clf_path = model_dir / "intent_clf.joblib"
        context_clf_path = model_dir / "context_clf.joblib"
        label_maps_path = model_dir / "label_maps.json"
        thresholds_path = model_dir / "thresholds.json"
        
        if not intent_clf_path.exists():
            raise FileNotFoundError(f"No se encontró {intent_clf_path}")
        if not context_clf_path.exists():
            raise FileNotFoundError(f"No se encontró {context_clf_path}")
        if not label_maps_path.exists():
            raise FileNotFoundError(f"No se encontró {label_maps_path}")
        
        intent_clf = joblib.load(str(intent_clf_path))
        context_clf = joblib.load(str(context_clf_path))
        
        with open(label_maps_path, "r", encoding="utf-8") as f:
            label_maps = json.load(f)
        
        # Cargar thresholds si existen
        thresholds = None
        if thresholds_path.exists():
            with open(thresholds_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
        
        return {
            "embedder": embedder,
            "intent_clf": intent_clf,
            "context_clf": context_clf,
            "label_maps": label_maps,
            "thresholds": thresholds
        }
    
    def predict(self, query: str) -> IntentRouterOutput:
        """
        Predice la intención de la consulta.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            IntentRouterOutput con intent y context_mode
        """
        if self.model:
            # Si hay modelo real, usarlo
            return self._predict_with_model(query)
        else:
            # Fallback a clasificador heurístico
            return self._predict_heuristic(query)
    
    def _predict_with_model(self, query: str) -> IntentRouterOutput:
        """
        Predice usando el modelo real.
        
        Proceso:
        1. Generar embedding de la consulta con SentenceTransformer
        2. Predecir intent con LogisticRegression
        3. Predecir context_mode con LogisticRegression
        4. Decodificar labels usando label_maps con validación de índices
        """
        try:
            embedder = self.model["embedder"]
            intent_clf = self.model["intent_clf"]
            context_clf = self.model["context_clf"]
            label_maps = self.model["label_maps"]
            thresholds = self.model.get("thresholds")
            
            # Validar estructura de label_maps
            if not label_maps or "intent" not in label_maps or "context_mode" not in label_maps:
                import logging
                logging.error("[ IntentRouter] label_maps corrupted or missing")
                return self._predict_heuristic(query)
            
            # 1. Generar embedding (384-dim, normalizado L2)
            embedding = embedder.encode([query], normalize_embeddings=True)
            
            if embedding is None or len(embedding) == 0:
                import logging
                logging.error("[IntentRouter] embedding generation failed")
                return self._predict_heuristic(query)
            
            # 2. Predecir intent
            intent_idx = intent_clf.predict(embedding)[0]
            intent_proba = intent_clf.predict_proba(embedding)[0]
            intent_confidence = float(intent_proba.max())
            
            # 3. Predecir context_mode
            context_idx = context_clf.predict(embedding)[0]
            context_proba = context_clf.predict_proba(embedding)[0]
            context_confidence = float(context_proba.max())
            
            # 4. Validar índices antes de acceder (prevenir IndexError)
            intent_idx = int(intent_idx)
            context_idx = int(context_idx)
            
            intent_dict = label_maps["intent"]
            context_dict = label_maps["context_mode"]
            
            intent_str = str(intent_idx)
            context_str = str(context_idx)
            
            # Validar que los índices existan en los mapeos
            if intent_str not in intent_dict:
                import logging
                logging.warning(
                    f"[IntentRouter] Invalid intent index {intent_idx}. "
                    f"Available: {list(intent_dict.keys())}"
                )
                return self._predict_heuristic(query)
            
            if context_str not in context_dict:
                import logging
                logging.warning(
                    f"[IntentRouter] Invalid context index {context_idx}. "
                    f"Available: {list(context_dict.keys())}"
                )
                return self._predict_heuristic(query)
            
            # 4. Decodificar usando label_maps (ahora con validación)
            intent_label = intent_dict[intent_str]
            context_label = context_dict[context_str]
            
            return IntentRouterOutput(
                intent=LabeledScore(label=intent_label, confidence=intent_confidence),
                context_mode=LabeledScore(label=context_label, confidence=context_confidence)
            )
        
        except IndexError as e:
            import logging
            logging.warning(
                f"[IntentRouter] IndexError al predecir: {e}. Usando clasificador heurístico."
            )
            return self._predict_heuristic(query)
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            import logging
            logging.warning(
                f"[IntentRouter] Error al predecir con modelo ({type(e).__name__}: {e}). "
                f"Usando clasificador heurístico."
            )
            return self._predict_heuristic(query)
    
    def _predict_heuristic(self, query: str) -> IntentRouterOutput:
        """
        Clasificador heurístico de respaldo.
        No calcula confidence (devuelve None).
        """
        q = query.lower()
        
        # Detectar metodología
        methodology_tokens = [
            "metodologia", "metodología", "definicion", "definición",
            "como se calcula", "cómo se calcula", "nota metodologica",
            "nota metodológica", "explicar", "qué es", "que es"
        ]
        intent = "methodology" if any(t in q for t in methodology_tokens) else "value"
        
        # Detectar follow-up
        followup_patterns = [
            r"^y ", r"y también", r"además", r"continuando",
            r"como te decía", r"también"
        ]
        context_mode = "followup" if any(re.search(p, q) for p in followup_patterns) else "standalone"
        
        return IntentRouterOutput(
            intent=LabeledScore(label=intent, confidence=None),
            context_mode=LabeledScore(label=context_mode, confidence=None)
        )
