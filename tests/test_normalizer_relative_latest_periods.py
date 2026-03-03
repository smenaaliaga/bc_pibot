from datetime import datetime

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
