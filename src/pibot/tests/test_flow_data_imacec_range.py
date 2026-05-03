from orchestrator.data import flow_data


def test_stream_data_flow_keeps_imacec_annual_single_year_as_range(monkeypatch):
    captured = {}

    def fake_specific_response(**kwargs):
        captured["req_form"] = kwargs.get("req_form")
        yield "ok"

    monkeypatch.setattr(flow_data, "specific_response", fake_specific_response)

    payload = {
        "classification": {
            "indicator_ent": "imacec",
            "seasonality_ent": "nsa",
            "frequency_ent": "a",
            "req_form_cls": "range",
            "calc_mode_cls": "yoy",
            "period_ent": ["2025-01-01", "2025-12-31"],
        },
        "series": "SERIE.IMACEC.TEST",
        "family_name": "IMACEC test family",
        "family_series": [{"id": "SERIE.IMACEC.TEST", "title": "IMACEC"}],
        "family_source_url": "https://example.test/family",
        "result": [{"date": "2025-12-31", "value": 1.36, "yoy": 2.3}],
    }

    chunks = list(flow_data.stream_data_flow(payload, session_id="s1"))

    assert chunks == ["ok"]
    assert captured["req_form"] == "range"


def test_stream_data_flow_converts_non_imacec_annual_single_year_range_to_point(monkeypatch):
    captured = {}

    def fake_specific_response(**kwargs):
        captured["req_form"] = kwargs.get("req_form")
        yield "ok"

    monkeypatch.setattr(flow_data, "specific_response", fake_specific_response)

    payload = {
        "classification": {
            "indicator_ent": "pib",
            "seasonality_ent": "nsa",
            "frequency_ent": "a",
            "req_form_cls": "range",
            "calc_mode_cls": "yoy",
            "period_ent": ["2025-01-01", "2025-12-31"],
        },
        "series": "SERIE.PIB.TEST",
        "family_name": "PIB test family",
        "family_series": [{"id": "SERIE.PIB.TEST", "title": "PIB"}],
        "family_source_url": "https://example.test/family",
        "result": [{"date": "2025-12-31", "value": 110.0, "yoy": 2.0}],
    }

    chunks = list(flow_data.stream_data_flow(payload, session_id="s2"))

    assert chunks == ["ok"]
    assert captured["req_form"] == "point"


def test_stream_data_flow_latest_contribution_uses_reference_period_over_parsed_point(monkeypatch):
    captured = {}

    def fake_specific_response(**kwargs):
        captured["req_form"] = kwargs.get("req_form")
        captured["date_range_label"] = kwargs.get("date_range_label")
        captured["display_period_label"] = kwargs.get("display_period_label")
        yield "ok"

    monkeypatch.setattr(flow_data, "specific_response", fake_specific_response)

    payload = {
        "classification": {
            "indicator_ent": "pib",
            "seasonality_ent": "nsa",
            "frequency_ent": "q",
            "req_form_cls": "latest",
            "calc_mode_cls": "contribution",
            "period_ent": ["2025-12-31"],
        },
        "series": "SERIE.PIB.TEST",
        "family_name": "PIB test family",
        "family_series": [{"id": "SERIE.PIB.TEST", "title": "PIB"}],
        "family_source_url": "https://example.test/family",
        "result": [{"date": "2025-09-30", "value": 1.6}],
        "all_series_data": [
            {"date": "2025-09-30", "value": 1.6},
            {"date": "2025-09-30", "value": 0.5},
        ],
    }

    chunks = list(flow_data.stream_data_flow(payload, session_id="s3"))

    assert chunks == ["ok"]
    assert captured["req_form"] == "latest"
    assert captured["date_range_label"] == "3er trimestre 2025"
    assert captured["display_period_label"] == "3er trimestre 2025"


def test_stream_data_flow_regional_pib_includes_region_in_indicator_name(monkeypatch):
    captured = {}

    def fake_specific_response(**kwargs):
        captured["final_indicator_name"] = kwargs.get("final_indicator_name")
        yield "ok"

    monkeypatch.setattr(flow_data, "specific_response", fake_specific_response)

    payload = {
        "classification": {
            "indicator_ent": "pib",
            "seasonality_ent": "nsa",
            "frequency_ent": "q",
            "req_form_cls": "point",
            "calc_mode_cls": "yoy",
            "region_cls": "specific",
            "region_ent": "magallanes",
            "period_ent": ["2025-07-01", "2025-09-30"],
        },
        "series": "SERIE.PIB.MAGALLANES.TEST",
        "family_name": "PIB test family",
        "family_series": [{"id": "SERIE.PIB.MAGALLANES.TEST", "title": "PIB Magallanes"}],
        "family_source_url": "https://example.test/family",
        "result": [{"date": "2025-09-30", "value": 110.0, "yoy": 4.9}],
    }

    chunks = list(flow_data.stream_data_flow(payload, session_id="s4"))

    assert chunks == ["ok"]
    assert "región de Magallanes" in str(captured.get("final_indicator_name") or "")
