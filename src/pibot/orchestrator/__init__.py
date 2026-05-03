"""LangGraph orchestrator adapters and helpers."""

import logging

# Reutiliza el logger raíz configurado en main/Streamlit (no crear archivos nuevos aquí)
logger = logging.getLogger(__name__)

from .memory.memory_adapter import MemoryAdapter
from .llm.llm_adapter import LLMAdapter
from .rag.rag_factory import create_retriever
from .classifier.intent_memory import IntentMemory
from .classifier.intent_store import (
    IntentStoreBase,
    InMemoryIntentStore,
    PostgresIntentStore,
    RedisIntentStore,
    create_intent_store,
)

# Exponer JointBERT predictor globalmente
# try:
#     from .classifier.joint_bert_classifier import get_predictor, predict, PIBotPredictor
#     _JOINT_BERT_AVAILABLE = True
# except ImportError as e:
#     # logger.warning(f"No se pudo importar JointBERT predictor: {e}")
#     # _JOINT_BERT_AVAILABLE = False
#     # get_predictor = None  # type: ignore
#     # predict = None  # type: ignore
#     # PIBotPredictor = None  # type: ignore

__all__ = [
    "MemoryAdapter",
    "LLMAdapter",
    "create_retriever",
    "IntentMemory",
    "IntentStoreBase",
    "InMemoryIntentStore",
    "PostgresIntentStore",
    "RedisIntentStore",
    "create_intent_store",
    "logger",
]

# # Agregar JointBERT si está disponible
# if _JOINT_BERT_AVAILABLE:
#     __all__.extend(["get_predictor", "predict", "PIBotPredictor"])
