"""Tests para historical PIB range: detección y omisión del level prefetch.

Reproducciones del run_detail_20260430:
  - "cual es el valor del pib de 1961 a 1970"
  - "cual es el valor del pib de 1960 en adelante"
  - "cual es el pib de chile desde 1900 en adelante"

Comportamiento esperado:
  - _is_historical_pib_range -> True
  - _build_level_prefetch_messages -> None (no inyecta prefetch sintético)
  - _build_historical_pib_yoy_instruction -> string con regla yoy_pct
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from orchestrator.data.response import (
    _build_historical_pib_yoy_instruction,
    _build_level_prefetch_messages,
    _is_historical_pib_range,
)


HISTORICAL_PIB_CASES = [
    # (question, entities_ctx)
    (
        "cual es el valor del pib de 1961 a 1970",
        {
            "indicator_ent": "pib",
            "frequency_ent": "a",
            "req_form_cls": "range",
            "period_ent": ["1961-01-01", "1970-12-31"],
            "hist": 1,
            "intent_cls": "value",
            "calc_mode_cls": "original",
            "historical_floor_instruction": (
                "REGLA DE DISPONIBILIDAD HISTÓRICA (PIB)..."
            ),
        },
    ),
    (
        "cual es el valor del pib de 1960 en adelante",
        {
            "indicator_ent": "pib",
            "frequency_ent": "a",
            "req_form_cls": "range",
            "period_ent": ["1960-01-01", "1960-12-31"],
            "hist": 1,
            "intent_cls": "value",
            "calc_mode_cls": "original",
            "historical_floor_instruction": (
                "REGLA DE DISPONIBILIDAD HISTÓRICA (PIB)..."
            ),
        },
    ),
    (
        "cual es el pib de chile desde 1900 en adelante",
        {
            "indicator_ent": "pib",
            "frequency_ent": "a",
            "req_form_cls": "range",
            "period_ent": ["1960-01-01", "2025-12-31"],
            "hist": 1,
            "intent_cls": "value",
            "calc_mode_cls": "original",
            "historical_floor_instruction": (
                "REGLA DE DISPONIBILIDAD HISTÓRICA (PIB)..."
            ),
        },
    ),
    # Rango amplio sin hist=1 (post-1996)
    (
        "cual es la evolución del pib entre 2000 y 2024",
        {
            "indicator_ent": "pib",
            "frequency_ent": "a",
            "req_form_cls": "range",
            "period_ent": ["2000-01-01", "2024-12-31"],
            "hist": 0,
            "intent_cls": "value",
            "calc_mode_cls": "original",
        },
    ),
]


@pytest.mark.parametrize("question,entities_ctx", HISTORICAL_PIB_CASES)
def test_is_historical_pib_range_true(question: str, entities_ctx: Dict[str, Any]):
    assert _is_historical_pib_range(question, entities_ctx) is True, question


@pytest.mark.parametrize("question,entities_ctx", HISTORICAL_PIB_CASES)
def test_level_prefetch_skipped_for_historical_pib(
    question: str, entities_ctx: Dict[str, Any]
):
    observations: Dict[str, Any] = {
        "series": [
            {
                "series_id": "F032.PIB.FLU.R.CLP.HIST18.Z.Z.0.A",
                "short_title": "PIB",
                "data": {"A": {"records": [{"period": "1960", "value": 1.0}]}},
                "classification": {"indicator": "pib"},
            }
        ],
        "frequency": "A",
        "latest_available": {"A": "2025"},
        "cuadro_name": "Producto Interno Bruto, información histórica",
    }
    out = _build_level_prefetch_messages(question, entities_ctx, observations)
    assert out is None, f"level prefetch debió omitirse para: {question!r}"


@pytest.mark.parametrize("question,entities_ctx", HISTORICAL_PIB_CASES)
def test_yoy_instruction_emitted(question: str, entities_ctx: Dict[str, Any]):
    instruction = _build_historical_pib_yoy_instruction(question, entities_ctx)
    assert instruction is not None
    low = instruction.lower()
    assert "yoy_pct" in low
    assert "get_series_data" in low
    assert "no dispongo" in low or "prohibido" in low


# ---------------------------------------------------------------------------
# Negativos: NO debe gatillarse para casos no históricos / no-rango / no-pib.
# ---------------------------------------------------------------------------

NEGATIVE_CASES = [
    # PIB pero punto único, no rango → flujo normal.
    (
        "cual es el pib del último trimestre",
        {
            "indicator_ent": "pib",
            "frequency_ent": "q",
            "req_form_cls": "point",
            "period_ent": ["2025-10-01", "2025-12-31"],
            "intent_cls": "value",
        },
    ),
    # IMACEC, no aplica.
    (
        "cual es la evolución del imacec desde 2010",
        {
            "indicator_ent": "imacec",
            "frequency_ent": "m",
            "req_form_cls": "range",
            "period_ent": ["2010-01-01", "2025-12-31"],
            "intent_cls": "value",
        },
    ),
    # PIB anual, mismo año (1 punto), sin keywords históricas → no aplica.
    (
        "cual es el pib en 2024",
        {
            "indicator_ent": "pib",
            "frequency_ent": "a",
            "req_form_cls": "point",
            "period_ent": ["2024-01-01", "2024-12-31"],
            "intent_cls": "value",
        },
    ),
]


@pytest.mark.parametrize("question,entities_ctx", NEGATIVE_CASES)
def test_is_historical_pib_range_false(question: str, entities_ctx: Dict[str, Any]):
    assert _is_historical_pib_range(question, entities_ctx) is False, question


@pytest.mark.parametrize("question,entities_ctx", NEGATIVE_CASES)
def test_yoy_instruction_not_emitted(question: str, entities_ctx: Dict[str, Any]):
    assert _build_historical_pib_yoy_instruction(question, entities_ctx) is None
