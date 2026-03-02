"""Business rules for response type routing.

Este módulo mantiene solo las reglas de negocio de selección de respuesta.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from rules.business_rule_support import _response_rules_config


def _has_valid_series(series_id: Any) -> bool:
    text = str(series_id or "").strip().lower()
    return bool(text and text not in {"none", "null", "nan"})


def _resolve_req_form(req_form: Any) -> str:
    value = str(req_form or "latest").strip().lower()
    return value or "latest"


def _rule_response_general(*, has_series: bool, rules_config: Dict[str, Any]) -> Optional[str]:
    # ## REGLA 1
    # 1. ##Título: Response general sin serie
    # 2. ##Descripción: Si no existe serie válida, se debe responder en modo general.
    # 3. ##Condiciones: `has_series == False`.
    # 4. ##Resultado: `response_general`.
    rule = rules_config.get("response_general") or {}
    conditions = rule.get("conditions") or {}
    expects_series = bool(conditions.get("has_series", False))
    if expects_series is False and not has_series:
        return str(rule.get("result") or "response_general")
    return None


def _rule_response_specific(
    *,
    has_series: bool,
    req_form_value: str,
    rules_config: Dict[str, Any],
) -> Optional[str]:
    # ## REGLA 2
    # 1. ##Título: Response específica estándar
    # 2. ##Descripción: Para serie válida y consultas latest/range/point se usa salida específica.
    # 3. ##Condiciones: `has_series == True` y `req_form in {latest, range, point}`.
    # 4. ##Resultado: `response_specific`.
    rule = rules_config.get("response_specific") or {}
    conditions = rule.get("conditions") or {}
    forms = {
        str(item).strip().lower()
        for item in (conditions.get("req_form_in") or ["latest", "range", "point"])
    }
    if has_series and req_form_value in forms:
        return str(rule.get("result") or "response_specific")
    return None


def _rule_response_specific_point(
    *,
    has_series: bool,
    req_form_value: str,
    rules_config: Dict[str, Any],
) -> Optional[str]:
    # ## REGLA 3
    # 1. ##Título: Response de punto específico
    # 2. ##Descripción: Para consultas explicitadas como specific_point se usa plantilla dedicada.
    # 3. ##Condiciones: `has_series == True` y `req_form in {specific_point}`.
    # 4. ##Resultado: `response_specific_point`.
    rule = rules_config.get("response_specific_point") or {}
    conditions = rule.get("conditions") or {}
    forms = {
        str(item).strip().lower()
        for item in (conditions.get("req_form_in") or ["specific_point"])
    }
    if has_series and req_form_value in forms:
        return str(rule.get("result") or "response_specific_point")
    return None


def resolve_response_rule(*, req_form: Any, series_id: Any) -> str:
    """Resolve response type rule based on user query type.

    Returns one of: ``response_general``, ``response_specific``,
    ``response_specific_point``.
    """

    req_form_value = _resolve_req_form(req_form)
    has_series = _has_valid_series(series_id)
    rules_config = _response_rules_config()

    # Orden de evaluación de reglas de negocio:
    #   1) Regla 1: general sin serie
    #   2) Regla 3: specific_point
    #   3) Regla 2: specific estándar
    resolved = _rule_response_general(has_series=has_series, rules_config=rules_config)
    if resolved:
        return resolved

    resolved = _rule_response_specific_point(
        has_series=has_series,
        req_form_value=req_form_value,
        rules_config=rules_config,
    )
    if resolved:
        return resolved

    resolved = _rule_response_specific(
        has_series=has_series,
        req_form_value=req_form_value,
        rules_config=rules_config,
    )
    if resolved:
        return resolved

    return "response_specific" if has_series else "response_general"


__all__ = [
    "resolve_response_rule",
]
