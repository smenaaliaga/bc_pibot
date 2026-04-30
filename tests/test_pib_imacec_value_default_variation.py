"""Regresión: consultas genéricas de PIB / IMACEC sin hint léxico de nivel
monetario deben responderse SOLO con variación (yoy_pct), no agregando un
párrafo adicional con el nivel.

Casos del usuario (run del 30/04/2026):
  1. "cual es el pib de mayo del 2026"
  2. "cual es el imacec del ultimo mes"
  3. "cual es el valor del pib del ultimo trimestre"
  4. "cual es el pib del primer trimestre del 2023"
  5. "cual es el valor del pib del 2023"

Para todos:
  - `_is_level_only_query` debe ser False.
  - `_build_level_prefetch_messages` debe retornar None
    (no se inyectan mensajes assistant/tool sintéticos con nivel).

Si en cambio el usuario pide explícitamente nivel monetario ("monto",
"valor en pesos", "miles de millones"), `_is_level_only_query` debe
seguir retornando True.
"""
from __future__ import annotations

import pytest

from orchestrator.data.response import (
    _is_level_only_query,
    _build_level_prefetch_messages,
)


@pytest.mark.parametrize(
    "question,ctx",
    [
        (
            "cual es el pib de mayo del 2026",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "",
                "frequency_ent": "m",
                "req_form_cls": "point",
            },
        ),
        (
            "cual es el imacec del ultimo mes",
            {
                "indicator_ent": "imacec",
                "intent_cls": "value",
                "calc_mode_cls": "",
                "frequency_ent": "m",
                "req_form_cls": "point",
            },
        ),
        (
            "cual es el valor del pib del ultimo trimestre",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "",
                "frequency_ent": "q",
                "req_form_cls": "point",
            },
        ),
        (
            "cual es el valor del pib del ultimo trimestre",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "original",
                "frequency_ent": "q",
                "req_form_cls": "point",
            },
        ),
        (
            "cual es el pib del primer trimestre del 2023",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "",
                "frequency_ent": "q",
                "req_form_cls": "point",
            },
        ),
        (
            "cual es el valor del pib del 2023",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "",
                "frequency_ent": "a",
                "req_form_cls": "point",
            },
        ),
        (
            "cual es el valor del pib del 2023",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "original",
                "frequency_ent": "a",
                "req_form_cls": "point",
            },
        ),
    ],
)
def test_value_query_is_not_level_only(question, ctx):
    assert _is_level_only_query(question, ctx) is False


@pytest.mark.parametrize(
    "question,ctx",
    [
        (
            "cual es el pib de mayo del 2026",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "",
                "frequency_ent": "m",
            },
        ),
        (
            "cual es el imacec del ultimo mes",
            {
                "indicator_ent": "imacec",
                "intent_cls": "value",
                "calc_mode_cls": "",
                "frequency_ent": "m",
            },
        ),
        (
            "cual es el valor del pib del ultimo trimestre",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "original",
                "frequency_ent": "q",
            },
        ),
        (
            "cual es el pib del primer trimestre del 2023",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "",
                "frequency_ent": "q",
            },
        ),
        (
            "cual es el valor del pib del 2023",
            {
                "indicator_ent": "pib",
                "intent_cls": "value",
                "calc_mode_cls": "",
                "frequency_ent": "a",
            },
        ),
    ],
)
def test_no_level_prefetch_injected(question, ctx):
    """Sin level-only ⇒ no se inyectan mensajes sintéticos de nivel."""
    observations = {
        "classification": {"calc_mode": "yoy"},
        "latest_available": {"m": "2026-04-30", "q": "2026-03-31", "a": "2025-12-31"},
        "series": [],
    }
    assert _build_level_prefetch_messages(question, ctx, observations) is None


@pytest.mark.parametrize(
    "question",
    [
        "a cuanto asciende el monto del pib del ultimo trimestre",
        "cual fue el pib en pesos del ultimo trimestre",
        "cuantos miles de millones fue el pib en 2023",
        "valor en pesos del pib en 2023",
    ],
)
def test_explicit_monetary_level_still_level_only(question):
    """Hints léxicos monetarios explícitos siguen activando level-only."""
    ctx = {
        "indicator_ent": "pib",
        "intent_cls": "value",
        "calc_mode_cls": "",
    }
    assert _is_level_only_query(question, ctx) is True


def test_pib_per_capita_still_level_only():
    """PIB per cápita sigue siendo nivel (capa 3 estructural)."""
    ctx = {
        "indicator_ent": "pib_per_capita",
        "intent_cls": "value",
        "calc_mode_cls": "",
    }
    assert _is_level_only_query("cual es el pib per capita del 2023", ctx) is True


def test_pib_nominal_still_level_only():
    """PIB nominal / precios corrientes sigue siendo nivel."""
    ctx = {
        "indicator_ent": "pib",
        "intent_cls": "value",
        "calc_mode_cls": "",
        "price_ent": "co",
    }
    assert _is_level_only_query("cual es el pib nominal del 2023", ctx) is True


def test_explicit_nivel_keyword_still_level_only():
    """'nivel del PIB' sigue gatillando level-only (capa 5 fallback léxico)."""
    ctx = {
        "indicator_ent": "pib",
        "intent_cls": "value",
        "calc_mode_cls": "",
    }
    assert _is_level_only_query("cual es el nivel del pib del 2023", ctx) is True
