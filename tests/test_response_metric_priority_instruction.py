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
