"""
Definición de dataclasses
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ClassificationResult:
    # Campos adicionales de JointBERT
    intent: str  # Intención ('value', 'methodology', 'ambiguous')
    confidence: float  # Confianza del modelo
    entities: dict  # Entidades raw 
    normalized: Optional[dict] = None  # Entidades normalizadas