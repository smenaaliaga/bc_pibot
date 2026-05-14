"""Tests para el manejo de rango histórico IMACEC mensual.

Análogo al manejo de PIB anual histórico (_is_historical_pib_range).
"""
from orchestrator.data.response import (
    _is_historical_imacec_range,
    _build_historical_imacec_instruction,
)


def _ctx(**overrides):
    base = {
        "indicator_ent": "imacec",
        "frequency_ent": "m",
        "req_form_cls": "range",
        "period_ent": ["1996-01-01", "2026-03-01"],
    }
    base.update(overrides)
    return base


def test_is_historical_imacec_range_desde_en_adelante():
    assert _is_historical_imacec_range(
        "entregame los valores del imacec desde enero de 1996 en adelante",
        _ctx(),
    )


def test_is_historical_imacec_range_a_partir_de():
    assert _is_historical_imacec_range(
        "imacec a partir de 1996",
        _ctx(),
    )


def test_is_historical_imacec_range_two_year_span():
    assert _is_historical_imacec_range(
        "imacec entre 2010 y 2020",
        _ctx(period_ent=["2010-01-01", "2020-12-31"]),
    )


def test_is_historical_imacec_range_rejects_pib():
    assert not _is_historical_imacec_range(
        "pib desde 1960 en adelante",
        _ctx(indicator_ent="pib", frequency_ent="a"),
    )


def test_is_historical_imacec_range_rejects_point_query():
    assert not _is_historical_imacec_range(
        "imacec de marzo de 2024",
        _ctx(req_form_cls="point", period_ent=["2024-03-01"]),
    )


def test_build_historical_imacec_instruction_present_keywords():
    msg = _build_historical_imacec_instruction(
        "imacec desde 1996 en adelante",
        _ctx(),
    )
    assert msg is not None
    low = msg.lower()
    # Debe instruir uso de get_series_data y prohibir 'n.d.'
    assert "get_series_data" in low
    assert "n.d." in low or "n.d" in low
    assert "1996" in msg


def test_build_historical_imacec_instruction_none_for_non_matching():
    msg = _build_historical_imacec_instruction(
        "imacec de marzo de 2024",
        _ctx(req_form_cls="point", period_ent=["2024-03-01"]),
    )
    assert msg is None
