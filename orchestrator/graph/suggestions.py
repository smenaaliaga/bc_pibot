"""Follow-up suggestion helpers for PIBot graph."""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any, Dict, List, Optional

from orchestrator.classifier.intent_store import IntentStoreBase

from .state import AgentState

logger = logging.getLogger(__name__)

_CHART_BLOCK_PATTERN = re.compile(r"##CHART_START(?P<body>.*?)##CHART_END", re.DOTALL)


def _coerce_indicator_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("standard_name", "normalized", "original", "text_normalized", "label"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _coerce_intent_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text.lower() or None
    if isinstance(value, dict):
        label = value.get("label")
        if isinstance(label, str):
            text = label.strip()
            return text.lower() or None
    return None


def _extract_indicator_context_from_entities(entities: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(entities, dict):
        return None
    jointbert = entities.get("jointbert")
    if not isinstance(jointbert, dict):
        return None
    normalized = jointbert.get("normalized")
    if not isinstance(normalized, dict):
        return None
    indicator = _coerce_indicator_value(normalized.get("indicator"))
    sector = _coerce_indicator_value(normalized.get("sector"))
    component = _coerce_indicator_value(normalized.get("component"))
    if not sector:
        sector = component
    if indicator or sector:
        context: Dict[str, str] = {}
        if indicator:
            context["indicator"] = indicator
        if sector:
            context["sector"] = sector
            context.setdefault("component", sector)
        elif component:
            context["component"] = component
        return context
    raw_entities = jointbert.get("entities")
    if isinstance(raw_entities, dict):
        indicator = _coerce_indicator_value(raw_entities.get("indicator"))
        if indicator:
            return {"indicator": indicator}
    return None


def _get_last_indicator_context(intent_store: Optional[IntentStoreBase], session_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not session_id or not intent_store or not hasattr(intent_store, "history"):
        return None
    try:
        records = intent_store.history(session_id, k=25)
    except Exception:
        logger.debug("[GRAPH] Unable to read intent history for indicator context", exc_info=True)
        return None
    for record in reversed(records or []):
        context = _extract_indicator_context_from_entities(getattr(record, "entities", None))
        if context:
            try:
                logger.debug("[GRAPH] indicator_context_resolved session_id=%s context=%s", session_id, context)
            except Exception:
                logger.exception("[GRAPH] Failed to log indicator_context")
            return context
    return None


def extract_chart_metadata_from_output(output: str) -> Optional[Dict[str, str]]:
    if not output:
        return None
    match = _CHART_BLOCK_PATTERN.search(output)
    if not match:
        return None
    body = match.group("body") or ""
    domain_match = re.search(r"domain\s*=\s*([A-Za-z0-9_\- ]+)", body, flags=re.IGNORECASE)
    if not domain_match:
        return None
    domain = domain_match.group(1).strip().upper()
    if not domain:
        return None
    return {
        "chart_domain": domain,
        "chart_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def generate_suggested_questions(state: AgentState, intent_store: Optional[IntentStoreBase]) -> List[str]:
    suggestions: List[str] = []
    session_id = state.get("session_id")
    entities = state.get("entities") or []
    primary_entity = next((ent for ent in entities if isinstance(ent, dict)), None)
    indicator = None
    component = None
    seasonality = None
    period: Optional[str] = None
    if primary_entity:
        indicator = primary_entity.get("indicator") or primary_entity.get("indicador")
        component = primary_entity.get("activity") or primary_entity.get("component")
        seasonality = primary_entity.get("seasonality")
    classification = state.get("classification")
    intent = _coerce_intent_label(getattr(classification, "intent", None) if classification else None)

    if not indicator:
        last_ctx = _get_last_indicator_context(intent_store, session_id)
        if last_ctx:
            indicator = last_ctx.get("indicator") or indicator
            component = component or last_ctx.get("component") or last_ctx.get("sector")

    if not indicator:
        suggestions.extend(
            [
                "¿Quieres que busque los datos más recientes?",
                "¿Te muestro un gráfico con la última variación?",
                "¿Prefieres consultar IMACEC o PIB?",
            ]
        )
    else:
        indicator_lower = indicator.lower()
        if seasonality:
            if "desestacionalizado" in seasonality.lower():
                suggestions.append(f"¿Cuál es el {indicator} sin desestacionalizar?")
            else:
                suggestions.append(f"¿Cuál es el {indicator} desestacionalizado?")
        else:
            suggestions.append(f"¿Cuál es el {indicator} desestacionalizado?")

        if "imacec" in indicator_lower:
            if not component or str(component).lower() == "total":
                suggestions.append("¿Cómo estuvo el IMACEC minero?")
            elif "minero" in str(component).lower():
                suggestions.append("¿Cómo estuvo el IMACEC no minero?")
            else:
                suggestions.append("¿Cómo estuvo el IMACEC total?")

        if "pib" in indicator_lower:
            if not component:
                suggestions.append("¿Cuál es la variación del PIB por sectores?")

        suggestions.append(f"¿Qué mide el {indicator}?")

        if period:
            suggestions.append(f"¿Cómo ha evolucionado el {indicator} en los últimos años?")

        if intent in ("methodology", "definition"):
            suggestions.insert(0, f"¿Cuál es el último valor del {indicator}?")

    seen = set()
    unique_suggestions = []
    for s in suggestions:
        normalized_key = re.sub(r"[^a-z0-9]+", "", s.lower())
        if normalized_key not in seen:
            seen.add(normalized_key)
            unique_suggestions.append(s)
    return unique_suggestions[:3]


__all__ = [
    "extract_chart_metadata_from_output",
    "generate_suggested_questions",
]
