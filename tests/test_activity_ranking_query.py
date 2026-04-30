"""Validación del fix: queries con calc_mode=yoy NO entran a level prefetch.

Bug: "¿Qué actividad creció más al último trimestre?" (classifier:
intent=value, calc_mode=yoy, indicator=pib, activity=general) era
clasificado como level-only por capa 4 de `_is_level_only_query`, lo que
inyectaba prefetch sintético con tool_choice='none' y dejaba al LLM sin
posibilidad de invocar `rank_series` → alucinación.

Fix: remover 'yoy' del set de calc_mode neutros en capa 4. El clasificador
ya distingue intención variacional vía calc_mode.
"""
from __future__ import annotations

import pytest

from orchestrator.data.response import (
    _is_level_only_query,
    _build_level_prefetch_messages,
)


@pytest.mark.parametrize(
    "calc_mode,expected_level_only",
    [
        ("", True),
        ("original", True),
        ("yoy", False),
        ("prev_period", False),
        ("contribution", False),
        ("share", False),
    ],
)
def test_level_only_calc_mode_dispatch(calc_mode, expected_level_only):
    """calc_mode del clasificador es la señal canónica de level vs variación."""
    ctx = {
        "indicator_ent": "pib",
        "intent_cls": "value",
        "calc_mode_cls": calc_mode,
    }
    assert _is_level_only_query("cuanto fue el pib", ctx) is expected_level_only


def test_activity_ranking_query_not_level_only():
    """Caso del log run_detail_20260430: ranking sectorial trimestral."""
    ctx = {
        "indicator_ent": "pib",
        "intent_cls": "value",
        "calc_mode_cls": "yoy",
        "activity_ent": "general",
        "req_form_cls": "point",
        "frequency_ent": "q",
    }
    assert _is_level_only_query(
        "¿Qué actividad creció más al último trimestre?", ctx
    ) is False


def test_activity_ranking_no_prefetch_injected():
    """Sin prefetch → el LLM puede invocar rank_series."""
    ctx = {
        "indicator_ent": "pib",
        "intent_cls": "value",
        "calc_mode_cls": "yoy",
        "activity_ent": "general",
        "req_form_cls": "point",
        "frequency_ent": "q",
    }
    observations = {
        "classification": {"calc_mode": "yoy"},
        "latest_available": {"q": "2026-03-31"},
    }
    result = _build_level_prefetch_messages(
        "¿Qué actividad creció más al último trimestre?", ctx, observations
    )
    assert result is None


def test_pib_level_query_still_level_only():
    """Sanidad: 'cuanto fue el pib del último trimestre' (calc_mode='') sigue siendo level."""
    ctx = {
        "indicator_ent": "pib",
        "intent_cls": "value",
        "calc_mode_cls": "",
    }
    assert _is_level_only_query(
        "cuanto fue el pib del último trimestre", ctx
    ) is True


def test_pib_yoy_growth_query_not_level_only():
    """'cuanto creció el pib en 2025' (calc_mode='yoy') debe permitir tool calls."""
    ctx = {
        "indicator_ent": "pib",
        "intent_cls": "value",
        "calc_mode_cls": "yoy",
    }
    # capa 1 (regex) ya intercepta "creció", pero validamos que aunque no
    # matcheara, capa 4 tampoco lo deja escapar a level-only.
    assert _is_level_only_query("cuanto creció el pib en 2025", ctx) is False


def test_imacec_value_query_unaffected():
    """IMACEC con calc_mode neutro sigue cayendo a fallback léxico (no level-only)."""
    ctx = {
        "indicator_ent": "imacec",
        "intent_cls": "value",
        "calc_mode_cls": "",
    }
    # IMACEC sin "nivel del" explícito → False (capa 4 lo excluye, capa 5 no matchea).
    assert _is_level_only_query("cuanto fue el imacec", ctx) is False


def test_imacec_yoy_query_not_level_only():
    """IMACEC con calc_mode=yoy: el LLM debe poder consultar yoy_pct vía tool."""
    ctx = {
        "indicator_ent": "imacec",
        "intent_cls": "value",
        "calc_mode_cls": "yoy",
    }
    assert _is_level_only_query("cuanto creció el imacec", ctx) is False
