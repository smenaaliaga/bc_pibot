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
            None,
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


def test_data_node_pib_aggregate_preserves_requested_seasonality_for_family_search(monkeypatch):
    captured = {}

    def fake_find_family_by_classification(*args, **kwargs):
        captured["seasonality"] = kwargs.get("seasonality")
        captured["frequency"] = kwargs.get("frequency")
        return {
            "family_name": "PIB aggregate test family",
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
            None,
        ),
    )

    monkeypatch.setattr(
        data_module.flow_data,
        "stream_data_flow",
        lambda payload, session_id="": iter(["ok"]),
    )

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["sa"],
            "frequency": ["q"],
            "period": ["2025-10-01", "2025-12-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={},
        calc_mode="prev_period",
        activity="general",
        region="none",
        investment="none",
        req_form="point",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "último trimestre desestacionalizado",
            "session_id": "s-pib-sa",
            "classification": classification,
            "entities": [{"indicator": "pib"}],
        }
    )

    assert captured["seasonality"] == "sa"
    assert captured["frequency"] is None
    assert result["output"] == "ok"


def test_build_target_series_url_uses_none_when_original_has_no_prev_period_values():
    url = data_module._build_target_series_url(
        source_url="https://example.test/series",
        series_id="SERIE.PIB.TEST",
        period=["1960-01-01", "1960-12-31"],
        observations=[
            {
                "date": "1960-12-31",
                "value": 19142,
            }
        ],
        frequency="a",
        calc_mode="original",
    )

    assert isinstance(url, str)
    assert "cbCalculo=NONE" in url


def test_build_target_series_url_uses_pct_when_original_has_prev_period_values():
    url = data_module._build_target_series_url(
        source_url="https://example.test/series",
        series_id="SERIE.PIB.TEST",
        period=["1961-01-01", "1961-12-31"],
        observations=[
            {
                "date": "1961-12-31",
                "value": 20000,
                "prev_period": 1.2,
            }
        ],
        frequency="a",
        calc_mode="original",
    )

    assert isinstance(url, str)
    assert "cbCalculo=PCT" in url


def test_build_target_series_url_uses_none_when_yoy_has_no_values():
    url = data_module._build_target_series_url(
        source_url="https://example.test/series",
        series_id="SERIE.PIB.TEST",
        period=["1961-01-01", "1961-12-31"],
        observations=[
            {
                "date": "1961-12-31",
                "value": 20000,
            }
        ],
        frequency="a",
        calc_mode="yoy",
    )

    assert isinstance(url, str)
    assert "cbCalculo=NONE" in url


def test_data_node_returns_explicit_no_data_message_for_empty_range(monkeypatch):
    stream_called = {"value": False}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB test family",
            "source_url": "https://example.test/family",
        },
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
        lambda **kwargs: ([], None, None),
    )

    def fake_stream_data_flow(payload, session_id=""):
        stream_called["value"] = True
        yield "should-not-run"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["a"],
            "period": ["2026-01-01", "2026-12-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"], "period": ["en 2026"]},
        calc_mode="yoy",
        activity="none",
        region="none",
        investment="none",
        req_form="range",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "PIB en 2026",
            "session_id": "s-empty-range",
            "classification": classification,
            "entities": [{"indicator": "pib", "period": ["en 2026"]}],
        }
    )

    assert "No hay datos disponibles" in result["output"]
    assert result["series"] == "SERIE.PIB.TEST"
    assert stream_called["value"] is False


def test_data_node_annual_range_uses_latest_complete_year_when_requested_year_incomplete(monkeypatch):
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB test family",
            "source_url": "https://example.test/family",
        },
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
            [{"date": "2025-12-31", "value": 110.0}],
            {"date": "2025-12-31", "value": 110.0},
            None,
        ),
    )

    def fake_load_series_observations(**kwargs):
        target_frequency = kwargs.get("target_frequency")
        firstdate = kwargs.get("firstdate")
        lastdate = kwargs.get("lastdate")
        if target_frequency == "Q" and firstdate == "2025-01-01" and lastdate == "2025-12-31":
            return (
                [
                    {"date": "2025-03-31", "value": 10.0},
                    {"date": "2025-06-30", "value": 20.0},
                    {"date": "2025-09-30", "value": 30.0},
                ],
                None,
            )
        if target_frequency == "A" and firstdate is None and lastdate is None:
            return (
                [
                    {"date": "2024-12-31", "value": 100.0},
                    {"date": "2025-12-31", "value": 110.0},
                ],
                None,
            )
        return ([], None)

    monkeypatch.setattr(
        data_module,
        "_load_series_observations",
        fake_load_series_observations,
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["a"],
            "period": ["2025-01-01", "2025-12-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"], "period": ["en 2025"]},
        calc_mode="yoy",
        activity="none",
        region="none",
        investment="none",
        req_form="range",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "PIB en 2025",
            "session_id": "s-incomplete-year",
            "classification": classification,
            "entities": [{"indicator": "pib", "period": ["en 2025"]}],
        }
    )

    assert "Como la información de 2025 aún no está completa" in result["output"]
    assert result["output"].endswith("ok")
    assert payload_holder["payload"]["classification"]["req_form_cls"] == "point"
    assert payload_holder["payload"]["parsed_point"] == "2024-12-31"
    assert payload_holder["payload"]["result"][0]["date"] == "2024-12-31"


def test_data_node_no_series_message_hides_empty_placeholders(monkeypatch):
    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: None,
    )

    classification = types.SimpleNamespace(
        normalized={
            "indicator": [],
            "seasonality": ["nsa"],
            "frequency": [],
            "period": ["2026-02-01", "2026-02-28"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["ipv"], "activity": [[]]},
        calc_mode="original",
        activity="none",
        region="none",
        investment="none",
        req_form="latest",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "dame el ultimo IPV",
            "session_id": "s-no-series-placeholders",
            "classification": classification,
            "entities": [{"indicator": ["ipv"], "activity": [[]]}],
        }
    )

    assert "[]" not in result["output"]
    assert "INDICADOR SOLICITADO" not in result["output"]
    assert "No encontré una serie" in result["output"]
    assert "https://si3.bcentral.cl/siete/" in result["output"]


def test_data_node_monthly_pib_falls_back_to_latest_quarter(monkeypatch):
    captured_fetch = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB test family",
            "source_url": "https://example.test/family",
        },
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

    def fake_fetch_series_by_req_form(**kwargs):
        captured_fetch["frequency"] = kwargs.get("frequency")
        captured_fetch["req_form"] = kwargs.get("req_form")
        return (
            [{"date": "2025-12-31", "value": 123.4, "yoy": 2.1}],
            {"date": "2025-12-31", "value": 123.4, "yoy": 2.1},
            None,
        )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        fake_fetch_series_by_req_form,
    )

    def fake_stream_data_flow(payload, session_id=""):
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["m"],
            "period": ["2026-02-01", "2026-02-28"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"], "period": ["mes pasado"]},
        calc_mode="original",
        activity="none",
        region="none",
        investment="none",
        req_form="point",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "dame el PIB del mes pasado",
            "session_id": "s-monthly-pib-fallback",
            "classification": classification,
            "entities": [{"indicator": ["pib"], "period": ["mes pasado"]}],
        }
    )

    assert captured_fetch["frequency"] == "q"
    assert captured_fetch["req_form"] == "latest"
    assert "El PIB no se calcula de forma mensual" in result["output"]
    assert "último trimestre disponible" in result["output"]
    assert result["output"].endswith("ok")


def test_data_node_range_uses_latest_obs_without_metadata_flag(monkeypatch):
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB test family",
            "source_url": "https://example.test/family",
        },
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
            [],
            {"date": "2025-09-30", "value": 52975, "yoy": 2.6},
            None,
        ),
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["q"],
            "period": ["2026-01-01", "2026-03-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"], "period": ["primer trimestre 2026"]},
        calc_mode="yoy",
        activity="none",
        region="none",
        investment="none",
        req_form="range",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "cuanto crecio la economia durante el primer trimestre del 2026",
            "session_id": "s-range-fallback-no-meta",
            "classification": classification,
            "entities": [{"indicator": ["pib"], "period": ["primer trimestre 2026"]}],
        }
    )

    assert "No hay datos disponibles" not in result["output"]
    assert payload_holder["payload"]["classification"]["req_form_cls"] == "point"
    assert payload_holder["payload"]["parsed_point"] == "2026-03-31"
    assert result["output"].endswith("ok")


def test_data_node_forces_imacec_frequency_to_monthly(monkeypatch):
    captured_fetch = {}
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "IMACEC test family",
            "source_url": "https://example.test/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [{"id": "SERIE.IMACEC.TEST", "title": "IMACEC"}],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[0],
    )

    def fake_fetch_series_by_req_form(**kwargs):
        captured_fetch["req_form"] = kwargs.get("req_form")
        captured_fetch["frequency"] = kwargs.get("frequency")
        return (
            [{"date": "2025-12-31", "value": 1.36, "yoy": 2.3}],
            {"date": "2025-12-31", "value": 1.36, "yoy": 2.3},
            None,
        )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        fake_fetch_series_by_req_form,
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["imacec"],
            "seasonality": ["nsa"],
            "frequency": ["a"],
            "period": ["2025-01-01", "2025-12-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["imacec"], "period": ["2025"], "frequency": ["anual"]},
        calc_mode="yoy",
        activity="specific",
        region="none",
        investment="none",
        req_form="point",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "imacec 2025",
            "session_id": "s-imacec-annual-range",
            "classification": classification,
            "entities": [{"indicator": ["imacec"], "period": ["2025"], "frequency": ["anual"]}],
        }
    )

    assert captured_fetch["req_form"] == "point"
    assert captured_fetch["frequency"] == "m"
    assert payload_holder["payload"]["classification"]["frequency"] == "m"
    assert payload_holder["payload"]["classification"]["req_form_cls"] == "point"
    assert result["output"].endswith("ok")


def test_data_node_contribution_aligns_period_with_target_series_date(monkeypatch):
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB contrib family",
            "source_url": "https://example.test/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [
            {"id": "SERIE.TARGET", "short_title": "Minería", "classification": {"activity": "mineria"}},
            {"id": "SERIE.OTRA", "short_title": "Servicios", "classification": {"activity": "servicios"}},
        ],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[0],
    )

    def fake_fetch_series_by_req_form(**kwargs):
        series_id = kwargs.get("series_id")
        if series_id == "SERIE.TARGET":
            return (
                [{"date": "2025-09-30", "value": 0.5}],
                {"date": "2025-09-30", "value": 0.5},
                None,
            )
        return (
            [
                {"date": "2025-06-30", "value": 0.2},
                {"date": "2025-09-30", "value": 0.3},
            ],
            {"date": "2025-09-30", "value": 0.3},
            None,
        )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        fake_fetch_series_by_req_form,
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["q"],
            "period": ["2025-10-01", "2025-12-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"], "period": ["4to trimestre 2025"]},
        calc_mode="contribution",
        activity="none",
        region="none",
        investment="none",
        req_form="point",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "aporte sectorial PIB 4to trimestre 2025",
            "session_id": "s-contrib-align-period",
            "classification": classification,
            "entities": [{"indicator": ["pib"], "period": ["4to trimestre 2025"]}],
        }
    )

    payload = payload_holder["payload"]
    assert payload["classification"]["req_form_cls"] == "point"
    assert payload["parsed_point"] == "2025-12-31"
    assert payload["parsed_range"] == ("2025-10-01", "2025-12-31")
    assert payload["used_latest_fallback_for_point"] is True
    assert all(str(row.get("date")) == "2025-09-30" for row in payload["result"])
    assert result["output"].endswith("ok")


def test_data_node_monthly_contribution_pib_falls_back_to_latest_quarter_with_note(monkeypatch):
    captured_find_kwargs = {}
    captured_fetch = {}

    def fake_find_family_by_classification(*args, **kwargs):
        captured_find_kwargs.update(kwargs)
        return {
            "family_name": "PIB contrib family",
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
        lambda family: [
            {"id": "SERIE.TOTAL", "short_title": "PIB", "classification": {"activity": "total"}},
            {"id": "SERIE.MINERIA", "short_title": "Minería", "classification": {"activity": "mineria"}},
        ],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[0],
    )

    def fake_fetch_series_by_req_form(**kwargs):
        captured_fetch["req_form"] = kwargs.get("req_form")
        captured_fetch["frequency"] = kwargs.get("frequency")
        return (
            [{"date": "2025-09-30", "value": 2.6}],
            {"date": "2025-09-30", "value": 2.6},
            None,
        )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        fake_fetch_series_by_req_form,
    )

    def fake_stream_data_flow(payload, session_id=""):
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["m"],
            "period": ["2025-03-01", "2025-03-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"], "period": ["marzo 2025"]},
        calc_mode="contribution",
        activity="none",
        region="none",
        investment="none",
        req_form="point",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "que sectores incidieron al pib en marzo 2025",
            "session_id": "s-contrib-monthly-fallback",
            "classification": classification,
            "entities": [{"indicator": ["pib"], "period": ["marzo 2025"]}],
        }
    )

    assert captured_fetch["req_form"] == "latest"
    assert captured_fetch["frequency"] == "q"
    assert captured_find_kwargs.get("seasonality") == "nsa"
    assert captured_find_kwargs.get("activity_value") == "general"
    assert "Las contribuciones al PIB no se calculan de forma mensual" in result["output"]
    assert "última contribución trimestral disponible" in result["output"]
    assert result["output"].endswith("ok")


def test_data_node_latest_contribution_uses_target_series_latest_date(monkeypatch):
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB contrib family",
            "source_url": "https://example.test/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [
            {"id": "SERIE.TOTAL", "short_title": "PIB", "classification": {"activity": "total"}},
            {"id": "SERIE.COMERCIO", "short_title": "Comercio", "classification": {"activity": "comercio"}},
        ],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[0],
    )

    def fake_fetch_series_by_req_form(**kwargs):
        series_id = kwargs.get("series_id")
        if series_id == "SERIE.TOTAL":
            return (
                [{"date": "2025-09-30", "value": 1.6}],
                {"date": "2025-09-30", "value": 1.6},
                None,
            )
        return (
            [
                {"date": "2025-12-31", "value": 0.8},
                {"date": "2025-09-30", "value": 0.5},
            ],
            {"date": "2025-12-31", "value": 0.8},
            None,
        )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        fake_fetch_series_by_req_form,
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["q"],
            "period": ["2025-10-01", "2025-12-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"]},
        calc_mode="contribution",
        activity="none",
        region="none",
        investment="none",
        req_form="latest",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "que sectores incidieron al pib",
            "session_id": "s-contrib-latest-target-date",
            "classification": classification,
            "entities": [{"indicator": ["pib"]}],
        }
    )

    payload = payload_holder["payload"]
    assert payload["classification"]["req_form_cls"] == "latest"
    assert payload["reference_period"] == "2025-09-30"
    assert all(str(row.get("date")) == "2025-09-30" for row in payload["all_series_data"])
    assert str(payload["result"][0].get("date")) == "2025-09-30"
    assert result["output"].endswith("ok")


def test_data_node_contribution_general_prefers_target_series_id_for_aggregate_row(monkeypatch):
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB contrib regional family",
            "source_url": "https://example.test/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [
            {"id": "SERIE.ARICA", "short_title": "Región de Arica y Parinacota", "classification": {"region": "arica_parinacota"}},
            {"id": "SERIE.PIB.TOTAL", "short_title": "Producto Interno Bruto", "classification": {"indicator": "pib"}},
            {"id": "SERIE.TARAPACA", "short_title": "Región de Tarapacá", "classification": {"region": "tarapaca"}},
        ],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[1],
    )

    def fake_fetch_series_by_req_form(**kwargs):
        series_id = kwargs.get("series_id")
        if series_id == "SERIE.PIB.TOTAL":
            return (
                [{"date": "2025-09-30", "value": 1.6}],
                {"date": "2025-09-30", "value": 1.6},
                None,
            )
        return (
            [{"date": "2025-09-30", "value": 0.0}],
            {"date": "2025-09-30", "value": 0.0},
            None,
        )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        fake_fetch_series_by_req_form,
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["q"],
            "period": ["2025-10-01", "2025-12-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"]},
        calc_mode="contribution",
        activity="general",
        region="general",
        investment="none",
        req_form="latest",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "contribución pib por región",
            "session_id": "s-contrib-regional-aggregate",
            "classification": classification,
            "entities": [{"indicator": ["pib"]}],
        }
    )

    payload = payload_holder["payload"]
    assert payload["series"] == "SERIE.PIB.TOTAL"
    assert str(payload["result"][0].get("series_id")) == "SERIE.PIB.TOTAL"
    assert float(payload["result"][0].get("value")) == 1.6
    assert result["output"].endswith("ok")


def test_data_node_contribution_aggregate_forces_target_activity_to_pib(monkeypatch):
    captured_eq = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB contrib family",
            "source_url": "https://example.test/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [
            {"id": "SERIE.PIB.TARGET", "short_title": "PIB", "classification": {"activity": "pib"}},
            {"id": "SERIE.COMERCIO", "short_title": "Comercio", "classification": {"activity": "comercio"}},
        ],
    )

    def fake_select_target_series_by_classification(family_series, eq, fallback_to_first=True):
        captured_eq.update(eq)
        return family_series[0]

    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        fake_select_target_series_by_classification,
    )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        lambda **kwargs: (
            [{"date": "2025-09-30", "value": 1.6}],
            {"date": "2025-09-30", "value": 1.6},
            None,
        ),
    )

    monkeypatch.setattr(
        data_module.flow_data,
        "stream_data_flow",
        lambda payload, session_id="": iter(["ok"]),
    )

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["q"],
            "period": ["2025-07-01", "2025-09-30"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"]},
        calc_mode="contribution",
        activity="none",
        region="none",
        investment="none",
        req_form="latest",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "que sectores incidieron al pib",
            "session_id": "s-contrib-target-activity",
            "classification": classification,
            "entities": [{"indicator": ["pib"]}],
        }
    )

    assert captured_eq.get("activity") == "pib"
    assert result["output"].endswith("ok")


def test_data_node_latest_contribution_aligns_to_latest_common_component_period(monkeypatch):
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB contrib family",
            "source_url": "https://example.test/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [
            {"id": "SERIE.PIB.TARGET", "short_title": "PIB", "classification": {"activity": "pib"}},
            {"id": "SERIE.COMERCIO", "short_title": "Comercio", "classification": {"activity": "comercio"}},
        ],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[0],
    )

    def fake_fetch_series_by_req_form(**kwargs):
        series_id = kwargs.get("series_id")
        if series_id == "SERIE.PIB.TARGET":
            return (
                [
                    {"date": "2025-09-30", "value": 1.3},
                    {"date": "2025-12-31", "value": 1.6},
                ],
                {"date": "2025-12-31", "value": 1.6},
                None,
            )
        return (
            [{"date": "2025-09-30", "value": 0.5}],
            {"date": "2025-09-30", "value": 0.5},
            None,
        )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        fake_fetch_series_by_req_form,
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["q"],
            "period": ["2025-10-01", "2025-12-31"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"]},
        calc_mode="contribution",
        activity="none",
        region="none",
        investment="none",
        req_form="latest",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "que explico el crecimiento del pib",
            "session_id": "s-contrib-latest-no-mix",
            "classification": classification,
            "entities": [{"indicator": ["pib"]}],
        }
    )

    payload = payload_holder["payload"]
    assert payload["classification"]["req_form_cls"] == "latest"
    assert payload["result"][0]["date"] == "2025-09-30"
    assert all(str(row.get("date")) == "2025-09-30" for row in (payload.get("all_series_data") or []))
    assert len(payload.get("all_series_data") or []) == 2
    assert result["output"].endswith("ok")


def test_data_node_original_general_collects_family_series_and_keeps_target(monkeypatch):
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB actividad family",
            "source_url": "https://example.test/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [
            {"id": "SERIE.PIB.TARGET", "short_title": "PIB", "classification": {"activity": "pib"}},
            {"id": "SERIE.COMERCIO", "short_title": "Comercio", "classification": {"activity": "comercio"}},
            {"id": "SERIE.INDUSTRIA", "short_title": "Industria", "classification": {"activity": "industria"}},
        ],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[0],
    )

    def fake_fetch_series_by_req_form(**kwargs):
        series_id = kwargs.get("series_id")
        if series_id == "SERIE.PIB.TARGET":
            return (
                [{"date": "2025-09-30", "value": 1000.0, "yoy": 1.6}],
                {"date": "2025-09-30", "value": 1000.0, "yoy": 1.6},
                None,
            )
        if series_id == "SERIE.COMERCIO":
            return (
                [{"date": "2025-09-30", "value": 250.0, "yoy": 4.2}],
                {"date": "2025-09-30", "value": 250.0, "yoy": 4.2},
                None,
            )
        return (
            [{"date": "2025-09-30", "value": 180.0, "yoy": 2.1}],
            {"date": "2025-09-30", "value": 180.0, "yoy": 2.1},
            None,
        )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        fake_fetch_series_by_req_form,
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["q"],
            "period": ["2025-07-01", "2025-09-30"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"]},
        calc_mode="original",
        activity="general",
        region="none",
        investment="none",
        req_form="latest",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "que actividad crecio mas en el pib",
            "session_id": "s-original-general-breakdown",
            "classification": classification,
            "entities": [{"indicator": ["pib"]}],
        }
    )

    payload = payload_holder["payload"]
    assert payload["classification"]["calc_mode_cls"] == "original"
    assert payload["classification"]["activity_cls"] == "general"
    assert payload["series"] == "SERIE.PIB.TARGET"
    assert payload["reference_period"] == "2025-09-30"
    assert len(payload.get("all_series_data") or []) == 3
    assert str(payload["result"][0].get("series_id")) == "SERIE.PIB.TARGET"
    assert str(payload["result"][0].get("date")) == "2025-09-30"
    assert result["output"].endswith("ok")


def test_data_node_yoy_general_collects_family_series_and_keeps_target(monkeypatch):
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "IMACEC family",
            "source_url": "https://example.test/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [
            {"id": "SERIE.IMACEC.TARGET", "short_title": "Imacec", "classification": {"activity": "imacec"}},
            {"id": "SERIE.COMERCIO", "short_title": "Comercio", "classification": {"activity": "comercio"}},
        ],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[0],
    )

    def fake_fetch_series_by_req_form(**kwargs):
        series_id = kwargs.get("series_id")
        if series_id == "SERIE.IMACEC.TARGET":
            return (
                [{"date": "2025-09-30", "value": 110.0, "yoy": 1.2}],
                {"date": "2025-09-30", "value": 110.0, "yoy": 1.2},
                None,
            )
        return (
            [{"date": "2025-09-30", "value": 98.0, "yoy": 2.8}],
            {"date": "2025-09-30", "value": 98.0, "yoy": 2.8},
            None,
        )

    monkeypatch.setattr(
        data_module,
        "_fetch_series_by_req_form",
        fake_fetch_series_by_req_form,
    )

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["imacec"],
            "seasonality": ["nsa"],
            "frequency": ["m"],
            "period": ["2025-09-01", "2025-09-30"],
            "activity": [],
            "region": [],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["imacec"]},
        calc_mode="yoy",
        activity="general",
        region="none",
        investment="none",
        req_form="latest",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "que actividad del imacec crecio mas",
            "session_id": "s-yoy-general-breakdown",
            "classification": classification,
            "entities": [{"indicator": ["imacec"]}],
        }
    )

    payload = payload_holder["payload"]
    assert payload["classification"]["calc_mode_cls"] == "yoy"
    assert payload["classification"]["activity_cls"] == "general"
    assert payload["series"] == "SERIE.IMACEC.TARGET"
    assert payload["reference_period"] == "2025-09-30"
    assert len(payload.get("all_series_data") or []) == 2
    assert str(payload["result"][0].get("series_id")) == "SERIE.IMACEC.TARGET"
    assert payload["all_series_data"][0].get("yoy") is not None
    assert result["output"].endswith("ok")


def test_data_node_contribution_specific_region_returns_only_target_row(monkeypatch):
    payload_holder = {}

    monkeypatch.setattr(
        series_search,
        "find_family_by_classification",
        lambda *args, **kwargs: {
            "family_name": "PIB contrib regional family",
            "source_url": "https://example.test/family",
        },
    )
    monkeypatch.setattr(
        series_search,
        "family_to_series_rows",
        lambda family: [
            {"id": "SERIE.ARICA", "short_title": "Región de Arica y Parinacota", "classification": {"region": "arica_parinacota"}},
            {"id": "SERIE.NUBLE", "short_title": "Región de Ñuble", "classification": {"region": "nuble"}},
            {"id": "SERIE.PIB", "short_title": "Producto Interno Bruto", "classification": {"indicator": "pib"}},
        ],
    )
    monkeypatch.setattr(
        series_search,
        "select_target_series_by_classification",
        lambda family_series, eq, fallback_to_first=True: family_series[1],
    )

    def fake_fetch_series_by_req_form(**kwargs):
        series_id = kwargs.get("series_id")
        if series_id == "SERIE.NUBLE":
            return (
                [{"date": "2025-09-30", "value": 0.0}],
                {"date": "2025-09-30", "value": 0.0},
                None,
            )
        if series_id == "SERIE.PIB":
            return (
                [{"date": "2025-09-30", "value": 1.6}],
                {"date": "2025-09-30", "value": 1.6},
                None,
            )
        return (
            [{"date": "2025-09-30", "value": 0.3}],
            {"date": "2025-09-30", "value": 0.3},
            None,
        )

    monkeypatch.setattr(data_module, "_fetch_series_by_req_form", fake_fetch_series_by_req_form)

    def fake_stream_data_flow(payload, session_id=""):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_module.flow_data, "stream_data_flow", fake_stream_data_flow)

    classification = types.SimpleNamespace(
        normalized={
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["q"],
            "period": ["2025-10-01", "2025-12-31"],
            "activity": [],
            "region": ["nuble"],
            "investment": [],
        },
        predict_raw={},
        entities={"indicator": ["pib"], "region": ["nuble"]},
        calc_mode="contribution",
        activity="none",
        region="specific",
        investment="none",
        req_form="point",
    )

    node = make_data_node(None)
    result = node(
        {
            "question": "contribución de ñuble al pib trimestre pasado",
            "session_id": "s-contrib-specific-region",
            "classification": classification,
            "entities": [{"indicator": ["pib"], "region": ["nuble"]}],
        }
    )

    payload = payload_holder["payload"]
    assert payload["series"] == "SERIE.NUBLE"
    assert len(payload.get("result") or []) == 1
    assert len(payload.get("all_series_data") or []) == 1
    assert str(payload["result"][0].get("series_id")) == "SERIE.NUBLE"
    assert str(payload["all_series_data"][0].get("series_id")) == "SERIE.NUBLE"
    assert result["output"].endswith("ok")
