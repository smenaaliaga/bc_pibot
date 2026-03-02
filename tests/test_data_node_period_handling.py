from types import SimpleNamespace

from orchestrator.graph.nodes.data import make_data_node


def test_data_node_handles_none_period_without_crashing(monkeypatch):
    import orchestrator.graph.nodes.data as data_module

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        lambda **kwargs: ([{"date": "2025-01", "value": 123.4}], {"date": "2025-01", "value": 123.4}),
    )

    monkeypatch.setattr(
        data_module.flow_data,
        "stream_data_flow",
        lambda payload, session_id=None: iter(["ok"]),
    )

    import orchestrator.catalog.series_search as series_search

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "IMACEC",
            "source_url": "https://example.com/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [{"id": "SERIE.IMACEC", "title": "Imacec"}],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda rows, eq=None, fallback_to_first=True: {"id": "SERIE.IMACEC", "title": "Imacec"},
    )

    classification = SimpleNamespace(
        predict_raw={},
        calc_mode=None,
        activity=None,
        region=None,
        investment=None,
        req_form=None,
        entities={},
        normalized={
            "indicator": ["imacec"],
            "seasonality": ["nsa"],
            "frequency": ["m"],
            "activity": [],
            "region": [],
            "investment": [],
            "period": None,
        },
    )

    state = {
        "question": "cuanto acelero la economia el ultimo mes",
        "session_id": "test-session",
        "entities": [],
        "classification": classification,
    }

    data_node = make_data_node(memory_adapter=None)
    result = data_node(state)

    assert result["output"] == "ok"
    assert result["parsed_range"] is None


def test_data_node_uses_period_bounds_when_available(monkeypatch):
    import orchestrator.graph.nodes.data as data_module

    captured_payload = {}

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        lambda **kwargs: ([{"date": "2025-02", "value": 120.0}], {"date": "2025-02", "value": 120.0}),
    )

    def _fake_stream_data_flow(payload, session_id=None):
        captured_payload.update(payload)
        return iter(["ok"])

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", _fake_stream_data_flow)

    import orchestrator.catalog.series_search as series_search

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "IMACEC",
            "source_url": "https://example.com/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [{"id": "SERIE.IMACEC", "title": "Imacec"}],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda rows, eq=None, fallback_to_first=True: {"id": "SERIE.IMACEC", "title": "Imacec"},
    )

    classification = SimpleNamespace(
        predict_raw={},
        calc_mode=None,
        activity="specific",
        region="none",
        investment="none",
        req_form="range",
        entities={},
        normalized={
            "indicator": ["imacec"],
            "seasonality": ["nsa"],
            "frequency": ["m"],
            "activity": ["imacec"],
            "region": [],
            "investment": [],
            "period": ["2024-01", "2024-12"],
        },
    )

    state = {
        "question": "cuanto acelero la economia en 2024",
        "session_id": "test-session",
        "entities": [],
        "classification": classification,
    }

    data_node = make_data_node(memory_adapter=None)
    result = data_node(state)

    assert result["output"] == "ok"
    assert result["parsed_range"] == ("2024-01", "2024-12")
    assert captured_payload["parsed_range"] == ("2024-01", "2024-12")
