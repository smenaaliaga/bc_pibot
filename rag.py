"""
rag.py
------
Módulo para futuras funciones RAG (Retrieval-Augmented Generation).

- Encapsula la generación de respuestas metodológicas para poder desconectar
  el uso directo de OpenAI y conectar a un pipeline RAG cuando esté listo.
"""

from typing import Optional, Dict


def generate_methodological_response(question: str, classification: Optional[Dict], history_text: str) -> str:
    """
    Placeholder de respuesta metodológica vía RAG.

    Implementar aquí conexión a tu índice/documentos (FAISS/Azure Cognitive Search/PGVector)
    y construir la respuesta usando un LLM con contexto recuperado.

    Por ahora, retorna un mensaje informativo.
    """
    return (
        "[RAG] Esta respuesta metodológica puede ser alimentada por un pipeline de recuperación "
        "(RAG). Configura 'rag.generate_methodological_response' para activar esa vía."
    )
