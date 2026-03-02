import types

import orchestrator.catalog.series_search as series_search
import orchestrator.graph.nodes.data as data_module
from orchestrator.graph.nodes.data import make_data_node


def test_data_node_prefers_followup_normalized_from_predict_raw(monkeypatch):
    captured = {}
    payload_holder = {}

    def fake_find_family_by_classification(*args, **kwargs):
        captured["indicator"] = kwargs.get("indicator")
        captured["frequency"] = kwargs.get("frequency")
        captured["seasonality"] = kwargs.get("seasonality")
        return {
            "family_name": "PIB test family",
            "source_url": "https://example.test/family",
        }

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        fake_find_family_by_classification,
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [{"id": "SERIE.PIB.TEST", "title": "PIB"}],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[0],
    )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        lambda **kwargs: (
            [{"date": "2025-12-31", "value": 123.45}],
            {"date": "2025-12-31", "value": 123.45},
        ),
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        # Stale classifier normalization that should NOT be used for follow-up resolution.
        normalized={
            "indicator": ["imacec"],
            "seasonality": ["nsa"],
            "frequency": ["m"],
            "period": ["2026-02-01", "2026-02-28"],
        },
        # Follow-up adjusted normalization that should win.
        predict_raw={
            "interpretation": {
                "entities_normalized": {
                    "indicator": "pib",
                    "seasonality": ["nsa"],
                    "frequency": ["q"],
                    "period": ["2025-10-01", "2025-12-31"],
                }
            }
        },
        entities={},
        calc_mode="original",
        activity="none",
        region="none",
        investment="none",
        req_form="latest",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "dame su ultimo valor",
            "session_id": "s-followup",
            "classification": classification,
            "entities": [{"indicator": "pib"}],
        }
    )

    assert captured["indicator"] == "pib"
    assert result["data_classification"]["indicator"] == "pib"
    assert payload_holder["payload"]["classification"]["indicator"] == "pib"
    assert result["output"] == "ok"
