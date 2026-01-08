"""
Plantillas de respuesta organizadas por categoría.
Sistema de decisión basado en contexto (intent, entidades disponibles, flags).
"""
from typing import Any, Callable, Dict, List


# Plantillas para consultas de datos (value, last, table)
DATA_TEMPLATES = [
    # Sin variación calculable (primer año o sin datos previos)
    {
        "match": lambda ctx: (
            ctx.get("has_indicator")
            and ctx.get("has_value")
            and not ctx.get("has_var_value")
            and ctx.get("has_period")
        ),
        "text": (
            "Acorde a la Base de Datos Estadísticos (BDE), en {period_label} el {indicator} "
            "registró un valor de {value} *, "
            "sin variación {var_label} disponible por no existir un período comparable para su cálculo."
        ),
    },
    # Consultas fuera de rango: posterior al último dato disponible
    {
        "match": lambda ctx: (
            ctx.get("has_indicator")
            and ctx.get("has_var_value")
            and ctx.get("has_period")
            and ctx.get("lastdate_position") == "gt_latest"
        ),
        "text": (
            "El período consultado está fuera del rango disponible "
            "(último dato disponible: {last_available_label}). "
            "Acorde a la Base de Datos Estadísticos (BDE), el último período disponible del "
            "{indicator} registró una variación {var_label} de {var_value:.1f}%."
        ),
    },
    # Consultas fuera de rango: anterior al inicio de la serie
    {
        "match": lambda ctx: (
            ctx.get("has_indicator")
            and ctx.get("has_var_value")
            and ctx.get("has_period")
            and ctx.get("lastdate_position") == "lt_first"
        ),
        "text": (
            "El período consultado está fuera del rango disponible "
            "(primer dato disponible: {first_available_label}). "
            "Acorde a la Base de Datos Estadísticos (BDE), el primer período disponible del "
            "{indicator} registró una variación {var_label} de {var_value:.1f}%."
        ),
    },
    # Consultas exactas al último dato disponible
    {
        "match": lambda ctx: (
            ctx.get("has_indicator")
            and ctx.get("has_var_value")
            and ctx.get("has_period")
            and ctx.get("lastdate_position") == "eq_latest"
        ),
        "text": (
            "Acorde a la Base de Datos Estadísticos (BDE), " #  del Banco Central de Chile
            "en {period_label} el {indicator} registró una variación {var_label} de "
            "{var_value:.1f}%."
        ),
    },
    # Consulta dentro de rango con período determinado
    {
        "match": lambda ctx: (
            ctx.get("has_indicator") and ctx.get("has_var_value") and ctx.get("has_period")
        ),
        "text": (
            "Acorde a la Base de Datos Estadísticos (BDE), " #  del Banco Central de Chile
            "en {period_label} el {indicator} registró una variación {var_label} de "
            "{var_value:.1f}%."
        ),
    },
    # Consulta general sin período determinado
    {
        "match": lambda ctx: (ctx.get("has_indicator") and ctx.get("has_var_value")),
        "text": (
            "Acorde a la Base de Datos Estadísticos (BDE), " # del Banco Central de Chile
            "la última variación {var_label} del {indicator} fue {var_value:.1f}% "
            "({period_label})."
        ),
    },
    # Sin datos
    {
        "match": lambda ctx: (ctx.get("has_indicator") and ctx.get("no_data")),
        "text": (
            "Según la Base de Datos Estadísticos (BDE) del Banco Central de Chile, "
            "no se encontraron registros del {indicator} para el período o rango consultado."
        ),
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
        "value": payload.get("value") if payload.get("value") is not None else "--",
        "var_label": payload.get("var_label") or "variación anual",
        "var_value": payload.get("var_value") if payload.get("var_value") is not None else 0.0,
        "period_label": payload.get("period_label") or "el período consultado",
        "last_available_label": payload.get("last_available_label") or "el último período disponible",
        "first_available_label": payload.get("first_available_label") or "el primer período disponible",
        "rag_summary": payload.get("rag_summary") or "",
        "definition_snippet": payload.get("definition_snippet") or "",
    }
    try:
        return text.format(**safe)
    except Exception:
        return text
