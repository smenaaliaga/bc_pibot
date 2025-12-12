"""
Módulo de predicción de intención y entidades usando Joint BERT.

Uso:
    # Inicializar
    predictor = PIBotPredictor(model_dir='model_out/pibot_model_beto')
    
    # Predecir
    result = predictor.predict("cual fue el imacec de agosto 2024")
    
    # Resultado:
    # {
    #     'intent': 'value',
    #     'confidence': 0.98,
    #     'entities': {
    #         'indicator': 'imacec',
    #         'period': 'agosto 2024'
    #     }
    # }
"""

import os
import logging
import torch
from datetime import date
from typing import Dict, Optional, Any
from transformers import BertTokenizer

try:
    from orchestrator.utils.period_normalizer import standardize_imacec_time_ref
    from orchestrator.utils.indicator_normalizer import standardize_indicator
    logger = logging.getLogger(__name__)
    logger.info("✓ Normalizadores cargados exitosamente")
except ImportError as e:
    # Normalizadores opcionales
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠ Normalizadores no disponibles: {e}")
    standardize_imacec_time_ref = None
    standardize_indicator = None

# Importar JointBERT model dinámicamente desde model/in
try:
    import sys
    import importlib.util
    from pathlib import Path
    
    # Cargar el módulo dinámicamente
    model_path = Path(__file__).parent.parent.parent / 'model' / 'in' / 'modeling_jointbert.py'
    spec = importlib.util.spec_from_file_location("modeling_jointbert", model_path)
    if spec and spec.loader:
        modeling_module = importlib.util.module_from_spec(spec)
        sys.modules['modeling_jointbert'] = modeling_module
        spec.loader.exec_module(modeling_module)
        JointBERT = modeling_module.JointBERT
        logger.info("✓ JointBERT cargado exitosamente")
    else:
        raise ImportError(f"No se pudo cargar spec desde {model_path}")
except Exception as e:
    # Fallback si no se encuentra el modelo
    JointBERT = None
    logger.warning(f"No se pudo importar JointBERT: {e}")

# Mapeo de tipos de modelo
MODEL_CLASSES = {
    'bert': (None, JointBERT) if JointBERT else (None, None),
    'beto': (None, JointBERT) if JointBERT else (None, None),
}

def get_intent_labels(args):
    """Lee labels de intención desde model_dir o usa fallback"""
    model_dir = getattr(args, 'model_dir', None)
    if model_dir:
        label_file = os.path.join(model_dir, 'intent_label.txt')
        if os.path.exists(label_file):
            return [label.strip() for label in open(label_file, 'r', encoding='utf-8')]
    # Fallback
    return ['value', 'methodology']


def get_slot_labels(args):
    """Lee labels de slots desde model_dir o usa fallback"""
    model_dir = getattr(args, 'model_dir', None)
    if model_dir:
        label_file = os.path.join(model_dir, 'slot_label.txt')
        if os.path.exists(label_file):
            return [label.strip() for label in open(label_file, 'r', encoding='utf-8')]
    # Fallback
    return [
        'O',
        'B-indicator', 'I-indicator',
        'B-frequency', 'I-frequency',
        'B-period', 'I-period',
        'B-component', 'I-component',
        'B-seasonality', 'I-seasonality',
    ]


class PIBotPredictor:
    """
    El predictor:
    1. Carga el modelo entrenado
    2. Procesa el texto de entrada
    3. Extrae intención y entidades
    4. Normaliza entidades (fechas e indicadores)
    5. Retorna un formato estructurado y limpio
    """
    
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        normalize_entities: bool = True,
        max_seq_len: int = 50
    ):
        """
        Inicializa el predictor.
        
        Args:
            model_dir: Directorio con el modelo entrenado
            device: 'cuda', 'cpu', o None (auto-detectar)
            normalize_entities: Si True, normaliza fechas e indicadores
            max_seq_len: Longitud máxima de secuencia
        """
        self.model_dir = model_dir
        self.normalize_entities = normalize_entities
        self.max_seq_len = max_seq_len
        
        # Detectar dispositivo
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Cargar modelo
        self._load_model()
        
        logger.info(f"PIBotPredictor inicializado en {self.device}")
    
    def _load_model(self):
        """Carga el modelo y componentes necesarios."""
        # Cargar args
        args_path = os.path.join(self.model_dir, 'training_args.bin')
        if not os.path.exists(args_path):
            raise FileNotFoundError(f"No se encuentra {args_path}")
        
        self.args = torch.load(args_path, weights_only=False)
        self.args.max_seq_len = self.max_seq_len  # Override si es necesario
        # Agregar model_dir a args para que get_intent_labels/get_slot_labels lo usen
        self.args.model_dir = self.model_dir
        
        # Cargar labels
        self.intent_label_lst = get_intent_labels(self.args)
        self.slot_label_lst = get_slot_labels(self.args)
        
        # Cargar tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.args.model_name_or_path)
        
        # Cargar modelo
        model_class = MODEL_CLASSES[self.args.model_type][1]
        self.model = model_class.from_pretrained(
            self.model_dir,
            args=self.args,
            intent_label_lst=self.intent_label_lst,
            slot_label_lst=self.slot_label_lst
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Modelo cargado: {len(self.intent_label_lst)} intenciones, "
                   f"{len(self.slot_label_lst)} slots")
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Realiza predicción completa sobre el texto.
        
        Args:
            text: Texto de entrada (consulta del usuario)
        
        Returns:
            Dict con formato:
            {
                'intent': str,              # Intención detectada
                'confidence': float,        # Confianza 0-1
                'entities': {               # Entidades extraídas
                    'indicator': str,       # Opcional
                    'period': str,          # Opcional
                    'measure_type': str,    # Opcional
                    ...
                },
                'normalized': {             # Opcional: entidades normalizadas
                    'indicator': {...},
                    'period': {...}
                }
            }
        """
        if not text or not text.strip():
            return {
                'intent': None,
                'confidence': 0.0,
                'entities': {}
            }
        
        # 1. Predicción básica (intent + slots)
        raw_prediction = self._predict_raw(text)
        
        # 2. Extraer entidades de los slots
        entities = self._extract_entities(
            raw_prediction['words'],
            raw_prediction['slots']
        )
        
        # 3. Normalizar entidades si está habilitado
        normalized = {}
        if self.normalize_entities:
            normalized = self._normalize_entities(entities)
        
        # 4. Construir respuesta
        result = {
            'intent': raw_prediction['intent'],
            'confidence': raw_prediction['intent_confidence'],
            'entities': entities
        }
        
        if normalized:
            result['normalized'] = normalized
        
        return result
    
    def _predict_raw(self, text: str) -> Dict[str, Any]:
        """
        Predicción básica del modelo (sin post-procesamiento).
        
        Returns:
            Dict con intent, confidence, words, slots
        """
        # Tokenizar
        words = text.split()
        tokens = []
        word_ids = []
        
        for i, word in enumerate(words):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.tokenizer.unk_token]
            tokens.extend(word_tokens)
            word_ids.extend([i] * len(word_tokens))
        
        # Truncar si es necesario
        if len(tokens) > self.args.max_seq_len - 2:
            tokens = tokens[:(self.args.max_seq_len - 2)]
            word_ids = word_ids[:(self.args.max_seq_len - 2)]
        
        # Agregar tokens especiales
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        word_ids = [-1] + word_ids + [-1]
        
        # Convertir a IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        
        # Padding
        padding_length = self.args.max_seq_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        
        # Convertir a tensors
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long).to(self.device)
        
        # Predecir
        with torch.no_grad():
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'intent_label_ids': None,
                'slot_labels_ids': None
            }
            outputs = self.model(**inputs)
            
            # El modelo siempre retorna: (total_loss, (intent_logits, slot_logits), ...)
            # Cuando no hay labels, total_loss = 0
            if isinstance(outputs, tuple) and len(outputs) > 0:
                # Caso: (loss, (intent_logits, slot_logits), ...)
                if isinstance(outputs[1], tuple):
                    intent_logits, slot_logits = outputs[1]
                else:
                    # Caso alternativo: (intent_logits, slot_logits)
                    intent_logits, slot_logits = outputs[0], outputs[1]
            else:
                raise ValueError(f"Formato de salida inesperado: {type(outputs)}")
            
            # Intent
            intent_pred = intent_logits.argmax(dim=-1).item()
            intent_label = self.intent_label_lst[intent_pred]
            intent_confidence = torch.softmax(intent_logits, dim=-1)[0][intent_pred].item()
            
            # Slots
            if self.args.use_crf:
                slot_preds = self.model.crf.decode(slot_logits)[0]
            else:
                slot_preds = slot_logits.argmax(dim=-1)[0].cpu().numpy()
            
            # Mapear slots a palabras originales
            slot_labels = []
            for i, word in enumerate(words):
                subtoken_idx = word_ids.index(i) if i in word_ids else -1
                if subtoken_idx >= 0 and subtoken_idx < len(slot_preds):
                    slot_labels.append(self.slot_label_lst[slot_preds[subtoken_idx]])
                else:
                    slot_labels.append("O")
        
        return {
            'text': text,
            'words': words,
            'intent': intent_label,
            'intent_confidence': intent_confidence,
            'slots': slot_labels
        }
    
    def _extract_entities(self, words: list, slots: list) -> Dict[str, str]:
        """
        Extrae entidades de las etiquetas BIO.
        
        Args:
            words: Lista de palabras
            slots: Lista de etiquetas BIO
        
        Returns:
            Dict con entidades extraídas {'indicator': 'imacec', 'period': 'agosto 2024'}
        """
        entities = {}
        current_entity = None
        current_value = []
        
        for word, slot in zip(words, slots):
            if slot.startswith('B-'):
                # Guardar entidad anterior si existe
                if current_entity:
                    entities[current_entity] = ' '.join(current_value)
                # Iniciar nueva entidad
                current_entity = slot[2:]  # Remover 'B-'
                current_value = [word]
            elif slot.startswith('I-') and current_entity:
                # Continuar entidad actual
                current_value.append(word)
            else:
                # Finalizar entidad actual si existe
                if current_entity:
                    entities[current_entity] = ' '.join(current_value)
                    current_entity = None
                    current_value = []
        
        # Guardar última entidad si existe
        if current_entity:
            entities[current_entity] = ' '.join(current_value)
        
        return entities
    
    def _normalize_entities(self, entities: Dict[str, str]) -> Dict[str, Any]:
        """
        Normaliza las entidades extraídas.
        
        Args:
            entities: Dict con entidades sin normalizar
        
        Returns:
            Dict con entidades normalizadas
        """
        normalized = {}
        
        # Normalizar período (solo si la función existe)
        if 'period' in entities:
            if standardize_imacec_time_ref is None:
                logger.warning("standardize_imacec_time_ref no está disponible")
            else:
                try:
                    logger.info(f"Normalizando período: '{entities['period']}'")
                    period_normalized = standardize_imacec_time_ref(
                        entities['period'],
                        date.today()
                    )
                    logger.info(f"Período normalizado: {period_normalized}")
                    if period_normalized:
                        normalized['period'] = period_normalized
                except Exception as e:
                    logger.warning(f"Error normalizando período '{entities['period']}': {e}", exc_info=True)
        
        # Normalizar indicador (solo si la función existe)
        if 'indicator' in entities:
            if standardize_indicator is None:
                logger.warning("standardize_indicator no está disponible")
            else:
                try:
                    logger.info(f"Normalizando indicador: '{entities['indicator']}'")
                    indicator_normalized = standardize_indicator(entities['indicator'])
                    logger.info(f"Indicador normalizado: {indicator_normalized}")
                    if indicator_normalized and indicator_normalized.get('indicator'):
                        normalized['indicator'] = {
                            'standard_name': indicator_normalized['indicator'].upper(),
                            'text_normalized': indicator_normalized['text_norm'],
                            'detected_by': indicator_normalized.get('text_standardized_imacec')
                        }
                except Exception as e:
                    logger.warning(f"Error normalizando indicador '{entities['indicator']}': {e}", exc_info=True)
        
        return normalized
    
    def predict_batch(self, texts: list) -> list:
        """
        Predice múltiples textos de forma secuencial.
        
        Args:
            texts: Lista de textos
        
        Returns:
            Lista de predicciones
        """
        return [self.predict(text) for text in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre el modelo cargado.
        
        Returns:
            Dict con información del modelo
        """
        return {
            'model_dir': self.model_dir,
            'model_type': self.args.model_type,
            'device': self.device,
            'num_intents': len(self.intent_label_lst),
            'intent_labels': self.intent_label_lst,
            'num_slots': len(self.slot_label_lst),
            'slot_labels': self.slot_label_lst,
            'max_seq_len': self.args.max_seq_len,
            'use_crf': self.args.use_crf,
            'normalize_entities': self.normalize_entities
        }


# Función de conveniencia para uso rápido
_global_predictor = None

def get_predictor(model_dir: Optional[str] = None, **kwargs) -> PIBotPredictor:
    """
    Obtiene una instancia global del predictor (patrón singleton con lazy loading).
    
    La primera llamada carga el modelo. Llamadas subsecuentes retornan la instancia cacheada.
    
    Args:
        model_dir: Directorio del modelo. Si None, usa JOINT_BERT_MODEL_DIR env var 
                   o 'model/out/pibot_model_beto' por defecto
        **kwargs: Argumentos adicionales para PIBotPredictor
    
    Returns:
        Instancia de PIBotPredictor
    """
    global _global_predictor
    
    # TEMPORAL: Forzar recarga si los normalizadores no estaban disponibles antes
    force_reload = os.getenv('FORCE_RELOAD_PREDICTOR', 'false').lower() == 'true'
    if force_reload and _global_predictor is not None:
        logger.info("Forzando recarga del predictor...")
        _global_predictor = None
    
    if _global_predictor is None:
        if model_dir is None:
            model_dir = os.getenv('JOINT_BERT_MODEL_DIR', 'model/out/pibot_model_beto')
        logger.info(f"Inicializando JointBERT predictor desde {model_dir}")
        _global_predictor = PIBotPredictor(model_dir, **kwargs)
    return _global_predictor


def predict(text: str, model_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para predicción rápida usando el singleton global.
    
    Args:
        text: Texto a predecir
        model_dir: Directorio del modelo (solo se usa en la primera llamada)
    
    Returns:
        Dict con predicción
    """
    predictor = get_predictor(model_dir)
    return predictor.predict(text)

