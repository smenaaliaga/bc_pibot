"""
Normalizador de estacionalidad para entidades tipo 'seasonality'.
Convierte diferentes formas de mención a una forma estándar.
"""

from difflib import get_close_matches

def normalize_seasonality(value: str) -> str:
    """
    Normaliza la entidad de estacionalidad a valores estándar, tolerando faltas de ortografía.
    Ejemplo: "desestacionalizado", "no desestacionalizado", "original", etc.
    """
    if not value:
        return ""
    v = value.strip().lower()
    # Listas de variantes aceptadas
    desestac = [
        "desestacionalizado", "desestacionalizada", "desestacionalizados", "desestacionalizadas",
        "ajustado por estacionalidad", "ajustada por estacionalidad", "ajustados por estacionalidad", "ajustadas por estacionalidad"
    ]
    original = [
        "no desestacionalizado", "no desestacionalizada", "original", "sin ajuste estacional", "sin ajustar"
    ]
    # Fuzzy match
    match = get_close_matches(v, desestac, n=1, cutoff=0.75)
    if match:
        return "desestacionalizado"
    match = get_close_matches(v, original, n=1, cutoff=0.75)
    if match:
        return "original"
    # Fallback: devolver el valor tal cual
    return v
