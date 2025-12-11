from typing import Dict, Any, List
import re


def build_followups(context: Dict[str, Any]) -> List[str]:
    """
    Genera follow-ups deterministas basados en el contexto actual.
    context puede incluir: domain, freq, metric, year, chart_last, csv_last, facts (dict), has_table, intent.
    """
    domain = (context.get("domain") or "").upper()
    freq = (context.get("freq") or "").upper()
    metric = (context.get("metric") or "").lower()
    facts = context.get("facts") or {}
    has_table = bool(context.get("has_table"))
    intent = (context.get("intent") or "").upper()

    # Evitar follow-ups si no hay dominio claro o no hay tabla mostrada
    if not domain or not has_table:
        return []

    items: List[str] = []

    # Si la consulta fue metodológica, orientar a pedir datos
    if intent == "METHODOLOGICAL":
        return [f"¿Quieres que busque los datos más recientes de {domain}?"]

    chart_last = (facts.get("chart_last") or "").upper()
    if domain == "IMACEC":
        if metric != "monthly":
            items.append("¿Quieres ver la variación mensual (mes a mes) del IMACEC?")
        if metric != "annual":
            items.append("¿Prefieres ver la variación anual del IMACEC?")
        if chart_last != "IMACEC":
            items.append("¿Necesitas un gráfico de la serie IMACEC?")
        items.append("¿Quieres consultar otro rango de meses del IMACEC?")

    elif domain == "PIB":
        if freq != "T":
            items.append("¿Quieres ver el PIB en frecuencia trimestral?")
        if freq != "A":
            items.append("¿Quieres ver el PIB en frecuencia anual?")
        if chart_last != "PIB":
            items.append("¿Necesitas un gráfico con la variación anual del PIB?")
        items.append("¿Quieres comparar con el año anterior?")

    # Usa facts para no repetir
    seen = set()
    out: List[str] = []
    for it in items:
        key = re.sub(r"[^a-z0-9]+", "", it.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    # Limitar a 3 follow-ups
    return out[:3]
