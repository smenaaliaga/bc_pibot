"""Tests para rangos abiertos hacia adelante en el period normalizer.

Casos cubiertos:
- "desde X en adelante" → [X-01-01, hoy]
- "desde X" (sin "hasta") → [X-01-01, hoy]
- "a partir de X" → [X-01-01, hoy]
- "X a Y" (bounded) → no abierto
- "desde X hasta Y" (bounded) → no abierto
"""
from datetime import date
from orchestrator.normalizer._period import resolve_period, reference_now


def _today_year() -> int:
    return reference_now().year


def test_resolve_period_desde_x_en_adelante_open_ended():
    start, end = resolve_period(
        ["desde 1996 en adelante"], None, {}, "range", "m"
    )
    assert start == "1996-01-01"
    assert end.startswith(f"{_today_year():04d}-")


def test_resolve_period_desde_x_sin_hasta_open_ended():
    start, end = resolve_period(["desde 1996"], None, {}, "range", "m")
    assert start == "1996-01-01"
    assert end.startswith(f"{_today_year():04d}-")


def test_resolve_period_a_partir_de_open_ended():
    start, end = resolve_period(["a partir de 2000"], None, {}, "range", "q")
    assert start == "2000-01-01"
    assert end.startswith(f"{_today_year():04d}-")


def test_resolve_period_bounded_range_not_affected():
    # "desde X hasta Y" no debe interpretarse como abierto.
    rng = resolve_period(["desde 1996 hasta 2000"], None, {}, "range", "m")
    assert rng == ["1996-01-01", "2000-12-31"]


def test_resolve_period_two_years_not_affected():
    # "X a Y" tampoco debe extenderse al presente.
    rng = resolve_period(["1996 a 2000"], None, {}, "range", "m")
    assert rng == ["1996-01-01", "2000-12-31"]


def test_resolve_period_single_year_point_unchanged():
    # Consulta puntual de un año no debe extenderse al presente.
    rng = resolve_period(["1961"], None, {}, "point", "a")
    assert rng == ["1961-01-01", "1961-12-31"]
