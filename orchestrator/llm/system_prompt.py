"""System prompt builder."""

from __future__ import annotations
from typing import Literal

GuardrailMode = Literal["rag", "fallback"]


def build_system_message(include_guards: bool = True, mode: GuardrailMode = "rag") -> str:
    """Construye el prompt del sistema."""
    base = """Eres el asistente económico del Banco Central de Chile (PIBot).
Respondes SIEMPRE en español.

Ayudas con consultas sobre indicadores económicos chilenos (IMACEC, PIB).
- Responde de forma clara y concisa
- Usa los datos proporcionados cuando estén disponibles
- Si no tienes información, indícalo claramente"""
    
    if include_guards:
        base += "\n- No inventes datos numéricos"
    
    if mode == "rag":
        return base + "\n\nMODO RAG: Usa el contexto de documentos recuperados."
    return base + "\n\nMODO FALLBACK: Responde con conocimiento general."


__all__ = ["build_system_message"]
