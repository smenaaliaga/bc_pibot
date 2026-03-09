from datetime import datetime

import orchestrator.normalizer.normalizer as normalizer_mod
from orchestrator.normalizer.normalizer import normalize_entities


def _expected_prev_month_range() -> list[str]:
    now = datetime.now()
    if now.month == 1:
        year = now.year - 1
        month = 12
    else:
        year = now.year
        month = now.month - 1

    start = f"{year:04d}-{month:02d}-01"
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1

    next_month_start = datetime(next_year, next_month, 1)
    end = (next_month_start - datetime.resolution).strftime("%Y-%m-%d")
    return [start, end]


def _expected_current_month_range() -> list[str]:
    now = datetime.now()
    year = now.year
    month = now.month

    start = f"{year:04d}-{month:02d}-01"
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1

    next_month_start = datetime(next_year, next_month, 1)
    end = (next_month_start - datetime.resolution).strftime("%Y-%m-%d")
    return [start, end]


def _expected_prev_quarter_range() -> list[str]:
    now = datetime.now()
    current_quarter_start = ((now.month - 1) // 3) * 3 + 1
    if current_quarter_start == 1:
        year = now.year - 1
        quarter_start = 10
    else:
        year = now.year
        quarter_start = current_quarter_start - 3

    quarter_end_month = quarter_start + 2
    start = f"{year:04d}-{quarter_start:02d}-01"

    if quarter_end_month == 12:
        next_q_year = year + 1
        next_q_month = 1
    else:
        next_q_year = year
        next_q_month = quarter_end_month + 1

    next_q_month_start = datetime(next_q_year, next_q_month, 1)
    end = (next_q_month_start - datetime.resolution).strftime("%Y-%m-%d")
    return [start, end]


def test_point_ultimo_trimestre_uses_previous_quarter():
    entities = {
        "indicator": ["pib"],
        "period": ["ultimo trimestre"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["period"] == _expected_prev_quarter_range()


def test_point_ultimo_mes_uses_previous_month():
    entities = {
        "indicator": ["imacec"],
        "period": ["ultimo mes"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["period"] == _expected_prev_month_range()


def test_point_mes_pasado_uses_previous_month():
    entities = {
        "indicator": ["imacec"],
        "period": ["mes pasado"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["period"] == _expected_prev_month_range()


def test_point_ultimo_dato_disponible_uses_frequency_q():
    entities = {
        "indicator": ["pib"],
        "frequency": ["trimestral"],
        "period": ["ultimo dato disponible"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["period"] == _expected_prev_quarter_range()


def test_point_mas_reciente_uses_frequency_m():
    entities = {
        "indicator": ["imacec"],
        "frequency": ["mensual"],
        "period": ["mas reciente"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["period"] == _expected_prev_month_range()


def test_point_ultimo_mes_infieres_m_frequency_for_pib():
    entities = {
        "indicator": ["pib"],
        "period": ["ultimo mes"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["frequency"] == ["m"]


def test_range_generic_indicator_with_1960_inferrs_pib_annual_and_full_year_period():
    entities = {
        "indicator": ["economia"],
        "period": ["1960"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="range")

    assert normalized["indicator"] == ["pib"]
    assert normalized["frequency"] == ["a"]
    assert normalized["period"] == ["1960-01-01", "1960-12-31"]


def test_range_imacec_with_year_only_keeps_monthly_frequency_without_period_inference():
    entities = {
        "indicator": ["imacec"],
        "period": ["durante el 2020"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="range")

    assert normalized["indicator"] == ["imacec"]
    assert normalized["frequency"] == ["m"]
    assert normalized["period"] == ["2020-01-01", "2020-12-31"]


def test_point_pib_este_mes_inferrs_monthly_frequency_and_current_month_period():
    entities = {
        "indicator": ["pib"],
        "period": ["este mes"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["indicator"] == ["pib"]
    assert normalized["frequency"] == ["m"]
    assert normalized["period"] == _expected_current_month_range()


def test_point_pib_trimestre_en_curso_inferrs_quarterly_frequency_and_current_quarter_period():
    now = datetime.now()
    quarter_start = ((now.month - 1) // 3) * 3 + 1
    quarter_end_month = quarter_start + 2
    if quarter_end_month == 12:
        next_q_month_start = datetime(now.year + 1, 1, 1)
    else:
        next_q_month_start = datetime(now.year, quarter_end_month + 1, 1)
    quarter_end = (next_q_month_start - datetime.resolution).day

    entities = {
        "indicator": ["pib"],
        "period": ["trimestre en curso"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["frequency"] == ["q"]
    assert normalized["period"] == [
        f"{now.year:04d}-{quarter_start:02d}-01",
        f"{now.year:04d}-{quarter_end_month:02d}-{quarter_end:02d}",
    ]


def test_imacec_explicit_annual_frequency_is_preserved():
    entities = {
        "indicator": ["imacec"],
        "frequency": ["anual"],
        "period": ["2025"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["indicator"] == ["imacec"]
    assert normalized["frequency"] == ["a"]


def test_point_imacec_first_quarter_without_year_detects_current_year_range():
    now = datetime.now()
    entities = {
        "indicator": ["imacec"],
        "period": ["1er trimestre"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["frequency"] == ["m"]
    assert normalized["period"] == [f"{now.year:04d}-01-01", f"{now.year:04d}-03-31"]


def test_point_imacec_month_abbreviation_detects_period():
    entities = {
        "indicator": ["imacec"],
        "period": ["sep 2024"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["frequency"] == ["m"]
    assert normalized["period"] == ["2024-09-01", "2024-09-30"]


def test_range_pib_decade_reference_detects_full_decade():
    entities = {
        "indicator": ["pib"],
        "period": ["años 90"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="range")

    assert normalized["frequency"] == ["a"]
    assert normalized["period"] == ["1990-01-01", "1999-12-31"]


def test_point_pib_split_period_tokens_ignores_preposition_token():
    entities = {
        "indicator": ["pib"],
        "period": ["en", "marzo del 2025"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["frequency"] == ["m"]
    assert normalized["period"] == ["2025-03-01", "2025-03-31"]


def test_point_pib_primer_trimestre_del_ano_pasado_uses_q1_previous_year(monkeypatch):
    monkeypatch.setattr(normalizer_mod, "_reference_now", lambda: datetime(2026, 3, 3))
    entities = {
        "indicator": ["pib"],
        "period": ["primer trimestre del año pasado"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["frequency"] == ["q"]
    assert normalized["period"] == ["2025-01-01", "2025-03-31"]


def test_point_imacec_mes_del_ano_antepasado_uses_same_month_two_years_back(monkeypatch):
    monkeypatch.setattr(normalizer_mod, "_reference_now", lambda: datetime(2026, 3, 3))
    entities = {
        "indicator": ["imacec"],
        "period": ["mes del año antepasado"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["frequency"] == ["m"]
    assert normalized["period"] == ["2024-03-01", "2024-03-31"]


def test_point_pib_hace_3_anos_atras_is_annual(monkeypatch):
    monkeypatch.setattr(normalizer_mod, "_reference_now", lambda: datetime(2026, 3, 3))
    entities = {
        "indicator": ["pib"],
        "period": ["hace 3 años atrás"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["frequency"] == ["a"]
    assert normalized["period"] == ["2023-01-01", "2023-12-31"]


def test_point_pib_hace_dos_trimestres_atras_uses_quarter_shift(monkeypatch):
    monkeypatch.setattr(normalizer_mod, "_reference_now", lambda: datetime(2026, 3, 3))
    entities = {
        "indicator": ["pib"],
        "period": ["hace dos trimestres atrás"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["frequency"] == ["q"]
    assert normalized["period"] == ["2025-07-01", "2025-09-30"]


def test_point_generic_activity_mayo_del_ano_pasado_infers_monthly(monkeypatch):
    monkeypatch.setattr(normalizer_mod, "_reference_now", lambda: datetime(2026, 3, 3))
    entities = {
        "indicator": ["actividad economicas"],
        "period": ["mayo del año pasado?"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="point")

    assert normalized["indicator"] == ["imacec"]
    assert normalized["frequency"] == ["m"]
    assert normalized["period"] == ["2025-05-01", "2025-05-31"]
