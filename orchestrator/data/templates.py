"""
Plantillas de respuesta organizadas por categoría.
Sistema de decisión basado en contexto (intent, entidades disponibles, flags).
"""
from typing import Any, Callable, Dict, List


# Plantillas para consultas de datos (value, last, table)
DATA_TEMPLATES: List[Dict[str, Any]] = [
    {
        "match": lambda ctx: (
            ctx.get("has_indicator") and ctx.get("has_value") and ctx.get("has_period")
        ),
        "text": "Acorde al {indicator} disponible en la BDE, la variación {var_label} para {period_label} fue de {var_value:.1f}%",
    },
    {
        "match": lambda ctx: (
            ctx.get("has_indicator") and ctx.get("has_value")
        ),
        "text": "Acorde al {indicator} disponible en la BDE, la última variación {var_label} fue de {var_value:.1f}% (período {period_label}).",
    },
    {
        "match": lambda ctx: (
            ctx.get("has_indicator") and ctx.get("no_data")
        ),
        "text": "No encontré datos recientes para {indicator} en el rango consultado.",
    },
]

# Plantillas para saludos y general
GREETING_TEMPLATES: List[Dict[str, Any]] = [
    {
        "match": lambda ctx: ctx.get("intent") == "greeting",
        "text": "Hola, ¿en qué indicador o período te ayudo?",
    },
]


# Plantilla fallback cuando ninguna regla aplica
FALLBACK_TEMPLATE: str = "No pude generar una respuesta con los datos disponibles."


def get_all_templates() -> List[Dict[str, Any]]:
    """
    Retorna todas las plantillas en orden de prioridad.
    DATA_TEMPLATES tienen mayor especificidad, luego saludos.
    """
    return DATA_TEMPLATES + GREETING_TEMPLATES


def select_template(context: Dict[str, Any]) -> str:
    """
    Evalúa el contexto contra todas las plantillas y retorna la primera que aplique.
    
    Args:
        context: Dict con flags y valores (intent, has_indicator, has_value, etc.)
    
    Returns:
        Texto de la plantilla seleccionada o FALLBACK_TEMPLATE.
    """
    for entry in get_all_templates():
        try:
            match_fn: Callable[[Dict[str, Any]], bool] = entry["match"]
            if match_fn(context):
                return entry["text"]
        except Exception:
            continue
    return FALLBACK_TEMPLATE


def render_template(text: str, payload: Dict[str, Any]) -> str:
    """
    Formatea una plantilla con valores seguros (defaults para campos faltantes).
    
    Args:
        text: Template string con placeholders {indicator}, {var_value}, etc.
        payload: Dict con valores para reemplazar.
    
    Returns:
        Texto formateado con fallbacks seguros.
    """
    safe = {
        "indicator": payload.get("indicator") or "el indicador",
        "var_label": payload.get("var_label") or "variación anual",
        "var_value": payload.get("var_value") if payload.get("var_value") is not None else 0.0,
        "period_label": payload.get("period_label") or "el período consultado",
        "rag_summary": payload.get("rag_summary") or "",
        "definition_snippet": payload.get("definition_snippet") or "",
    }
    try:
        return text.format(**safe)
    except Exception:
        return text
