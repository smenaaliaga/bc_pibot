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
