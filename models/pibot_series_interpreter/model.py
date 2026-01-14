"""
SeriesInterpreter - pibot-series-interpreter

Clasifica las características de la serie solicitada.

Cabezas del modelo:
- indicator: "imacec" | "pib"
- metric_type: "index" | "contribution"
- seasonality: "sa" | "nsa"
- activity: "total" | "imc_*" | "pib_*"
- frequency: "m" | "q" | "a"
- calc_mode: "none" | "yoy" | "prev_period"
- req_form: "latest" | "point" | "range"

Para usar un modelo real entrenado:
1. Colocar el modelo en esta carpeta (e.g., checkpoint/, model.safetensors, config.json)
2. Descomentar y adaptar el método load_model()
3. El modelo debe retornar un dict con todas las cabezas listadas arriba
"""

from typing import Optional, Union, Dict, Any, List
import os
import re
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
import torch
from ..base import BaseClassifier, SeriesInterpreterOutput, LabeledScore


class SeriesInterpreter(BaseClassifier):
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Ruta al modelo entrenado. Si es None, usa clasificador heurístico.
                       Si es "enabled", carga el modelo desde pibot-jointbert/.
        """
        self.model = None
        self.model_path = model_path
        self.label_maps = {}
        
        if model_path:
            self.model = self.load_model(model_path)
    
    def load_model(self, model_path: str) -> Any:
        """
        Carga el modelo JointBERT desde la carpeta especificada.
        
        Args:
            model_path: Ruta a la carpeta del modelo (absoluta o relativa).
                       Si es relativa, se resuelve desde el directorio de este archivo.
        
        El modelo tiene 7 clasificadores independientes (no BIO tagging):
        - IndicatorClassifier
        - MetricTypeClassifier  
        - CalcModeClassifier
        - SeasonalClassifier
        - ReqFormClassifier
        - FrequencyClassifier
        - ActivityClassifier
        """
        
        # Resolver ruta del modelo
        model_dir = Path(model_path)
        if not model_dir.is_absolute():
            # Si es relativa, resolver desde el directorio de este archivo
            base_dir = Path(__file__).parent
            model_dir = base_dir / model_path
        
        # Cargar labels primero
        self.label_maps = self._load_label_maps(model_dir)
        
        try:
            # Importar módulos locales sin alterar sys.path de manera permanente
            import importlib.util
            
            # Añadir el directorio del modelo a sys.path temporalmente para que
            # los imports dentro de modeling_jointbert.py (como torchcrf) funcionen
            model_dir_str = str(model_dir)
            sys.path.insert(0, model_dir_str)
            
            try:
                # Cargar module.py primero
                module_spec = importlib.util.spec_from_file_location("module", model_dir / "module.py")
                module_module = importlib.util.module_from_spec(module_spec)
                sys.modules["module"] = module_module
                module_spec.loader.exec_module(module_module)
                
                # Cargar modeling_jointbert.py
                spec = importlib.util.spec_from_file_location("modeling_jointbert", model_dir / "modeling_jointbert.py")
                modeling_module = importlib.util.module_from_spec(spec)
                sys.modules["modeling_jointbert"] = modeling_module
                spec.loader.exec_module(modeling_module)
            finally:
                # Remover el directorio del modelo de sys.path
                if model_dir_str in sys.path:
                    sys.path.remove(model_dir_str)
            JointBERT = modeling_module.JointBERT
            
            # Cargar config
            config = AutoConfig.from_pretrained(str(model_dir))
            
            # Crear args mock (necesario para JointBERT.__init__)
            class Args:
                dropout_rate = 0.1
                slot_loss_coef = 1.0
                ignore_index = -100
            
            args = Args()
            
            # Inicializar modelo
            model = JointBERT(
                config=config,
                args=args,
                indicator_label_lst=self.label_maps.get("indicator", []),
                metric_type_label_lst=self.label_maps.get("metric_type", []),
                calc_mode_label_lst=self.label_maps.get("calc_mode", []),
                seasonal_label_lst=self.label_maps.get("seasonality", []),
                req_form_label_lst=self.label_maps.get("req_form", []),
                frequency_label_lst=self.label_maps.get("frequency", []),
                activity_label_lst=self.label_maps.get("activity", [])
            )
            
            # Cargar pesos desde safetensors
            from safetensors.torch import load_file
            state_dict = load_file(str(model_dir / "model.safetensors"))
            model.load_state_dict(state_dict)
            model.eval()
            
            # Cargar tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            
            return {
                "model": model,
                "tokenizer": tokenizer
            }
        except Exception as e:
            import traceback
            raise RuntimeError(
                f"Error al cargar modelo desde {model_dir}:\n{e}\n\n{traceback.format_exc()}"
            )
        finally:
            # Limpiar módulos del cache
            for mod in ["modeling_jointbert", "module"]:
                if mod in sys.modules:
                    del sys.modules[mod]
    
    def _load_label_maps(self, model_dir: Path) -> Dict[str, List[str]]:
        """Carga los mapeos de labels desde archivos .txt"""
        label_maps = {}
        label_files = {
            "indicator": "indicator_label.txt",
            "metric_type": "metric_type_label.txt",
            "seasonality": "seasonal_label.txt",
            "activity": "activity_label.txt",
            "frequency": "frequency_label.txt",
            "calc_mode": "calc_mode_label.txt",
            "req_form": "req_form_label.txt",
        }
        
        for slot_name, filename in label_files.items():
            filepath = model_dir / filename
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    labels = [line.strip() for line in f.readlines() if line.strip()]
                    label_maps[slot_name] = labels
        
        return label_maps
    
    def predict(self, query: str) -> SeriesInterpreterOutput:
        """
        Clasifica las características de la serie.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            SeriesInterpreterOutput con todas las cabezas clasificadas
        """
        if self.model:
            # Si hay modelo real, usarlo
            return self._predict_with_model(query)
        else:
            # Fallback a clasificador heurístico
            return self._predict_heuristic(query)
    
    def _predict_with_model(self, query: str) -> SeriesInterpreterOutput:
        """
        Predice usando el modelo JointBERT con 7 clasificadores.
        
        El modelo retorna: (loss, logits_tuple, ...)
        donde logits_tuple = (indicator_logits, metric_type_logits, calc_mode_logits,
                               seasonal_logits, req_form_logits, frequency_logits, activity_logits)
        
        Con validación de índices para prevenir IndexError.
        """
        import torch
        import logging
        
        try:
            # Extraer modelo y tokenizer
            model = self.model["model"]
            tokenizer = self.model["tokenizer"]
            
            # Validar que label_maps exista y tenga contenido
            if not self.label_maps:
                logging.error("[SeriesInterpreter] label_maps is empty or missing")
                return self._predict_heuristic(query)
            
            # Tokenizar entrada
            inputs = tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs.get("token_type_ids", None)
                )
            
            # outputs[0] es la pérdida total, outputs[1] es la tupla de 7 logits
            logits_tuple = outputs[1]
            
            (indicator_logits, metric_type_logits, calc_mode_logits,
             seasonal_logits, req_form_logits, frequency_logits,
             activity_logits) = logits_tuple
            
            # Aplicar argmax para obtener índices predichos
            # Aplicar argmax para obtener índices predichos + calcular confidence
            indicator_idx = int(torch.argmax(indicator_logits, dim=-1).item())
            indicator_confidence = float(torch.softmax(indicator_logits, dim=-1).max().item())
            
            metric_type_idx = int(torch.argmax(metric_type_logits, dim=-1).item())
            metric_type_confidence = float(torch.softmax(metric_type_logits, dim=-1).max().item())
            
            calc_mode_idx = int(torch.argmax(calc_mode_logits, dim=-1).item())
            calc_mode_confidence = float(torch.softmax(calc_mode_logits, dim=-1).max().item())
            
            seasonal_idx = int(torch.argmax(seasonal_logits, dim=-1).item())
            seasonal_confidence = float(torch.softmax(seasonal_logits, dim=-1).max().item())
            
            req_form_idx = int(torch.argmax(req_form_logits, dim=-1).item())
            req_form_confidence = float(torch.softmax(req_form_logits, dim=-1).max().item())
            
            frequency_idx = int(torch.argmax(frequency_logits, dim=-1).item())
            frequency_confidence = float(torch.softmax(frequency_logits, dim=-1).max().item())
            
            activity_idx = int(torch.argmax(activity_logits, dim=-1).item())
            activity_confidence = float(torch.softmax(activity_logits, dim=-1).max().item())
            
            # Helper para acceder a labels con validación
            def safe_get_label(label_dict_key, idx):
                label_list = self.label_maps.get(label_dict_key, [])
                if idx < 0 or idx >= len(label_list):
                    logging.warning(
                        f"[SeriesInterpreter] Index {idx} out of range for {label_dict_key}. "
                        f"Length: {len(label_list)}. Using 'unknown'."
                    )
                    return "unknown"
                return label_list[idx]
            
            # Mapear índices a labels usando los label_maps con validación
            indicator = safe_get_label("indicator", indicator_idx)
            metric_type = safe_get_label("metric_type", metric_type_idx)
            calc_mode = safe_get_label("calc_mode", calc_mode_idx)
            seasonality = safe_get_label("seasonality", seasonal_idx)
            req_form = safe_get_label("req_form", req_form_idx)
            frequency = safe_get_label("frequency", frequency_idx)
            activity = safe_get_label("activity", activity_idx)
            
            return SeriesInterpreterOutput(
                indicator=LabeledScore(label=indicator, confidence=indicator_confidence),
                metric_type=LabeledScore(label=metric_type, confidence=metric_type_confidence),
                seasonality=LabeledScore(label=seasonality, confidence=seasonal_confidence),
                activity=LabeledScore(label=activity, confidence=activity_confidence),
                frequency=LabeledScore(label=frequency, confidence=frequency_confidence),
                calc_mode=LabeledScore(label=calc_mode, confidence=calc_mode_confidence),
                req_form=LabeledScore(label=req_form, confidence=req_form_confidence)
            )
        
        except IndexError as e:
            logging.warning(
                f"[SeriesInterpreter] IndexError: {e}. Falling back to heuristic."
            )
            return self._predict_heuristic(query)
        except Exception as e:
            logging.error(
                f"[SeriesInterpreter] Unexpected error in _predict_with_model: "
                f"{type(e).__name__}: {e}. Falling back to heuristic.",
                exc_info=True
            )
            return self._predict_heuristic(query)
    
    def _predict_heuristic(self, query: str) -> SeriesInterpreterOutput:
        """
        Clasificador heurístico de respaldo.
        No calcula confidence (devuelve None).
        """
        q = query.lower()
        
        # Indicator
        indicator = "imacec" if "imacec" in q else "pib"
        
        # metric_type
        metric_type = "contribution" if ("contrib" in q or "contribucion" in q or "contribución" in q) else "index"
        
        # Seasonality
        seasonality_tokens = ["sa", "desestacional", "ajustado estacionalmente", "ajustado", "seasonally", "seasonal"]
        seasonality = "sa" if any(t in q for t in seasonality_tokens) else "nsa"
        # Default para imacec es sa
        if not any(t in q for t in seasonality_tokens + ["sin ajuste", "no ajustado", "nsa"]):
            if "imacec" in q:
                seasonality = "sa"
        
        # Activity
        activity = "total"
        m = re.search(r"(imc|pib)_[a-z0-9_]+", q)
        if m:
            activity = m.group(0)
        
        # Frequency (orden de prioridad: q > a > m)
        freq_map = [
            ("q", ["trimestral", "trimestre", "q1", "q2", "q3", "q4", "t1", "t2", "t3", "t4", 
                   "primer trimestre", "segundo trimestre", "tercer trimestre", "cuarto trimestre"]),
            ("a", ["anual", "año", "aa"]),
            ("m", ["mensual", "mes", "enero", "febrero", "marzo", "abril", "mayo", "junio",
                   "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]),
        ]
        frequency = "m"
        for f, toks in freq_map:
            if f == "a" and "interanual" in q:
                # Skip 'a' si interanual está presente (es calc_mode no frequency)
                continue
            if any(t in q for t in toks):
                frequency = f
                break
        
        # Calc mode
        if any(t in q for t in ["yoy", "interanual", "a/a", "año anterior", "anterior año"]):
            calc_mode = "yoy"
        elif any(t in q for t in ["m/m", "mensual", "variación mensual", "trimestral", "q/q", 
                                   "periodo anterior", "período anterior"]):
            calc_mode = "prev_period"
        else:
            calc_mode = "none"
        
        # Request form
        if any(t in q for t in ["ultima", "última", "ultimo", "último", "mas reciente", 
                                "más reciente", "latest"]):
            req_form = "latest"
        else:
            # Detect range connectors
            is_range = False
            if ("entre" in q and " y " in q) or ("desde" in q and "hasta" in q):
                is_range = True
            elif " a " in q and re.search(r"[q|t]([1-4])", q):
                quarters = re.findall(r"[q|t]([1-4])", q)
                if len(quarters) >= 2:
                    is_range = True
            
            if is_range:
                req_form = "range"
            else:
                # Detect date mentions
                date_tokens = [
                    "enero", "febrero", "marzo", "abril", "mayo", "junio",
                    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
                    "q1", "q2", "q3", "q4", "t1", "t2", "t3", "t4"
                ]
                req_form = "point" if any(t in q for t in date_tokens) else "latest"
        
        return SeriesInterpreterOutput(
            indicator=LabeledScore(label=indicator, confidence=None),
            metric_type=LabeledScore(label=metric_type, confidence=None),
            seasonality=LabeledScore(label=seasonality, confidence=None),
            activity=LabeledScore(label=activity, confidence=None),
            frequency=LabeledScore(label=frequency, confidence=None),
            calc_mode=LabeledScore(label=calc_mode, confidence=None),
            req_form=LabeledScore(label=req_form, confidence=None)
        )
