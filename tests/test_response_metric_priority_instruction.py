import orchestrator.data.response as response_module


def test_build_metric_priority_instruction_for_yoy():
    text = response_module._build_metric_priority_instruction("yoy")
    assert text is not None
    assert "'yoy_pct'" in text
    assert "PERIODO analizado" in text
    assert "No comiences con 'value'" in text


def test_build_metric_priority_instruction_for_prev_period():
    text = response_module._build_metric_priority_instruction("prev_period")
    assert text is not None
    assert "'pct'" in text
    assert "PERIODO analizado" in text
    assert "No comiences con 'value'" in text


def test_build_metric_priority_instruction_for_original():
    text = response_module._build_metric_priority_instruction("original")
    assert text is not None
    assert "'value'" in text
    assert "PERIODO analizado" in text
    assert "No comiences con 'yoy_pct'" in text


def test_build_metric_priority_instruction_for_unknown_mode_returns_none():
    assert response_module._build_metric_priority_instruction("share") is None
    assert response_module._build_metric_priority_instruction("") is None


def test_build_incomplete_period_instruction_for_incomplete_quarterly_year():
    text = response_module._build_incomplete_period_instruction(
        question="cuanto crecio el pib de la region metropolitana en 2025",
        entities_ctx={"frequency_ent": "q", "period_ent": ["2025"]},
        observations={"latest_available": {"T": "2025-Q3"}, "classification": {"frequency": "q"}},
    )
    assert text is not None
    assert "NO está cerrado" in text
    assert "Hasta 3er trimestre de 2025" in text or "hasta 3er trimestre de 2025" in text
    assert "No escribas frases como 'En 2025, ...'" in text


def test_build_incomplete_period_instruction_for_incomplete_first_semester():
    text = response_module._build_incomplete_period_instruction(
        question="como estuvo el primer semestre de 2025",
        entities_ctx={"frequency_ent": "q", "period_ent": []},
        observations={"latest_available": {"T": "2025-Q1"}, "classification": {"frequency": "q"}},
    )
    assert text is not None
    assert "semestre solicitado NO está completo" in text
    assert "primer semestre de 2025" in text
    assert "Hasta 1er trimestre de 2025" in text or "hasta 1er trimestre de 2025" in text


def test_build_incomplete_period_instruction_returns_none_for_complete_year():
    text = response_module._build_incomplete_period_instruction(
        question="cuanto crecio el pib en 2024",
        entities_ctx={"frequency_ent": "q", "period_ent": ["2024"]},
        observations={"latest_available": {"T": "2024-Q4"}, "classification": {"frequency": "q"}},
    )
    assert text is None


def test_build_incomplete_period_instruction_uses_frequency_from_entities_ctx():
    text = response_module._build_incomplete_period_instruction(
        question="como estuvo el primer semestre de 2025",
        entities_ctx={"frequency_ent": "q", "period_ent": []},
        observations={"latest_available": {"T": "2025-Q1"}, "classification": {"frequency": "a"}},
    )
    assert text is not None
    assert "semestre solicitado NO está completo" in text


def test_relative_period_fallback_instruction_when_last_year_has_no_data():
    year = response_module.date.today().year
    text = response_module._build_relative_period_fallback_instruction(
        question="cuanto crecio la mineria el año pasado",
        entities_ctx={"frequency_ent": "a"},
        observations={"latest_available": {"A": str(year - 2)}},
    )
    assert text is not None
    assert "sin datos disponibles" in text
    assert str(year - 1) in text
    assert str(year - 2) in text


def test_relative_period_fallback_instruction_none_when_last_year_exists():
    year = response_module.date.today().year
    text = response_module._build_relative_period_fallback_instruction(
        question="cuanto crecio la mineria el año pasado",
        entities_ctx={"frequency_ent": "a"},
        observations={"latest_available": {"A": str(year - 1)}},
    )
    assert text is None


def test_missing_activity_instruction_when_requested_activity_not_available():
    text = response_module._build_missing_activity_instruction(
        entities_ctx={"activity_ent": "mineria", "activity_cls": "specific"},
        observations={
            "series": [
                {"classification_series": {"activity": "industria"}},
                {"classification_series": {"activity": "servicios"}},
                {"classification_series": {"indicator": "pib"}},
            ]
        },
    )
    assert text is not None
    assert "actividad solicitada no existe" in text
    assert "NO reemplaces por la serie agregada" in text


def test_missing_activity_instruction_none_when_activity_exists():
    text = response_module._build_missing_activity_instruction(
        entities_ctx={"activity_ent": "industria"},
        observations={
            "series": [
                {"classification_series": {"activity": "industria"}},
                {"classification_series": {"activity": "servicios"}},
            ]
        },
    )
    assert text is None


def test_missing_activity_instruction_none_for_imacec_indicator_token():
    text = response_module._build_missing_activity_instruction(
        entities_ctx={"activity_ent": "imacec", "activity_cls": "specific"},
        observations={
            "series": [
                {"classification_series": {"activity": "industria"}},
                {"classification_series": {"activity": "servicios"}},
            ]
        },
    )
    assert text is None


def test_missing_activity_instruction_none_for_pib_indicator_token():
    text = response_module._build_missing_activity_instruction(
        entities_ctx={"activity_ent": "pib", "activity_cls": "specific"},
        observations={
            "series": [
                {"classification_series": {"activity": "industria"}},
                {"classification_series": {"activity": "servicios"}},
            ]
        },
    )
    assert text is None


def test_no_explicit_period_instruction_when_question_has_no_date():
    text = response_module._build_no_explicit_period_latest_instruction(
        question="dame la contribucion del sector minero a la economia",
        entities_ctx={"frequency_ent": "m"},
        observations={"latest_available": {"M": "2026-01"}},
    )
    assert text is not None
    assert "no especifica fecha" in text
    assert "ultimo periodo disponible" in text
    assert "NO menciones falta de datos" in text


def test_no_explicit_period_instruction_none_when_relative_date_is_explicit():
    text = response_module._build_no_explicit_period_latest_instruction(
        question="dame la contribucion del sector minero el mes pasado",
        entities_ctx={"frequency_ent": "m"},
        observations={"latest_available": {"M": "2026-01"}},
    )
    assert text is None


def test_no_explicit_period_instruction_when_req_form_latest_even_with_relative_date():
    text = response_module._build_no_explicit_period_latest_instruction(
        question="dame la contribucion del sector minero el mes pasado",
        entities_ctx={"frequency_ent": "m", "req_form_cls": "latest"},
        observations={"latest_available": {"M": "2026-01"}},
    )
    assert text is not None
    assert "req_form_cls='latest'" in text


def test_relative_period_fallback_instruction_none_when_req_form_latest():
    year = response_module.date.today().year
    text = response_module._build_relative_period_fallback_instruction(
        question="dame la contribucion del sector minero el mes pasado",
        entities_ctx={"frequency_ent": "m", "req_form_cls": "latest"},
        observations={"latest_available": {"M": f"{year}-01"}},
    )
    assert text is None
