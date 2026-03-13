from __future__ import annotations

from types import SimpleNamespace

import orchestrator.graph.nodes.data as data_node_module
from orchestrator.data._business_rules import ResolvedEntities
from rules.business_rule import resolve_data_node_overrides


def test_resolve_data_node_overrides_corrientes_and_share(monkeypatch):
    monkeypatch.setenv("ENABLE_RULE_PIB_CORRIENTES", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_SHARE", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_SHARE_GASTO_GUARDRAIL", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_PER_CAPITA", "1")

    corrientes = resolve_data_node_overrides(
        question="Cual es el valor del PIB a precios corrientes del 2020",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="point",
        frequency_ent="a",
    )
    assert corrientes.get("price") == "co"

    share = resolve_data_node_overrides(
        question="Cuanto pesa mineria en el PIB?",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="latest",
        frequency_ent="q",
    )
    assert share.get("calc_mode_cls") == "share"
    assert share.get("frequency_ent") == "a"
    assert share.get("price") == "co"

    share_gasto = resolve_data_node_overrides(
        question="Que porcentaje del PIB son las exportaciones?",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="latest",
        frequency_ent="q",
    )
    assert "short_circuit_message" in share_gasto
    assert share_gasto.get("feature") == "pib_share_gasto_unavailable"
    msg = str(share_gasto.get("short_circuit_message") or "").lower()
    assert "las puedes revisar en el siguiente cuadro" in msg
    assert "no tiene series habilitadas" not in msg

    per_capita = resolve_data_node_overrides(
        question="Cual es el PIB per capita del 2025",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="point",
        frequency_ent="q",
    )
    assert per_capita.get("activity_ent") == "pib_percapita"
    assert per_capita.get("activity_cls") == "specific"
    assert per_capita.get("activity_cls_resolved") == "specific"
    assert per_capita.get("region_ent") is None
    assert per_capita.get("region_cls") == "none"
    assert per_capita.get("investment_ent") is None
    assert per_capita.get("investment_cls") == "none"
    assert per_capita.get("frequency_ent") == "a"
    assert per_capita.get("price") is None
    assert "short_circuit_message" not in per_capita


def test_resolve_data_node_overrides_defaults_corrientes_only_for_cuanto_es_pib(monkeypatch):
    monkeypatch.setenv("ENABLE_RULE_PIB_CORRIENTES", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_SHARE", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_PER_CAPITA", "1")

    variants = [
        "Cuánto es el PIB de Chile?",
        "Cuanto es el PIB de Chile?",
        "A cuánto vale el PIB chileno?",
        "De cuánto fue el PIB de Chile?",
        "Cuánto es el producto interno bruto de Chile?",
    ]

    for question in variants:
        result = resolve_data_node_overrides(
            question=question,
            indicator_ent="pib",
            calc_mode_cls="original",
            req_form_cls="latest",
            frequency_ent="q",
        )
        assert result.get("price") == "co", question
        assert result.get("feature") == "pib_corrientes_cuanto_es", question


def test_resolve_data_node_overrides_does_not_force_corrientes_for_cual_es_encadenado_or_variation(monkeypatch):
    monkeypatch.setenv("ENABLE_RULE_PIB_CORRIENTES", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_SHARE", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_PER_CAPITA", "1")

    cual_es = resolve_data_node_overrides(
        question="Cual es el PIB de Chile?",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="latest",
        frequency_ent="q",
    )
    assert cual_es.get("price") is None

    cuanto_crecio = resolve_data_node_overrides(
        question="Cuánto creció el PIB de Chile?",
        indicator_ent="pib",
        calc_mode_cls="yoy",
        req_form_cls="latest",
        frequency_ent="q",
    )
    assert cuanto_crecio.get("price") is None

    encadenado = resolve_data_node_overrides(
        question="Cual es el PIB a precios encadenados?",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="latest",
        frequency_ent="q",
    )
    assert encadenado.get("price") is None

    variacion = resolve_data_node_overrides(
        question="Cual es la variación del PIB de Chile?",
        indicator_ent="pib",
        calc_mode_cls="yoy",
        req_form_cls="latest",
        frequency_ent="q",
    )
    assert variacion.get("price") is None


def test_resolve_data_node_overrides_infieres_pib_indicator_when_missing(monkeypatch):
    monkeypatch.setenv("ENABLE_RULE_PIB_CORRIENTES", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_SHARE", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_PER_CAPITA", "1")

    variants = [
        "Cuánto es el PIB de Chile?",
        "Cuanto es el PIB de Chile?",
        "¿Cua\u0301nto es el producto interno bruto de Chile?",
    ]

    for question in variants:
        result = resolve_data_node_overrides(
            question=question,
            indicator_ent=None,
            calc_mode_cls="original",
            req_form_cls="latest",
            frequency_ent="q",
        )
        assert result.get("indicator_ent") == "pib", question
        assert result.get("price") == "co", question


def test_data_node_applies_inferred_indicator_override_before_lookup(monkeypatch):
    monkeypatch.setenv("ENABLE_RULE_PIB_CORRIENTES", "1")

    ent = ResolvedEntities(
        indicator_ent=None,
        frequency_ent="q",
        calc_mode_cls="original",
        seasonality_ent="nsa",
        activity_ent=None,
        activity_cls="none",
        activity_cls_resolved="none",
        region_ent=None,
        region_cls="none",
        investment_ent=None,
        investment_cls="none",
        req_form_cls="latest",
        price=None,
    )

    monkeypatch.setattr(
        data_node_module,
        "_extract_entities_from_state",
        lambda state: (state.get("question", ""), [], ent),
    )
    monkeypatch.setattr(data_node_module, "apply_business_rules", lambda _ent: _ent)

    lookup_capture: dict[str, object] = {}

    def _fake_lookup(_ent):
        lookup_capture["indicator_ent"] = _ent.indicator_ent
        lookup_capture["price"] = _ent.price
        return SimpleNamespace(
            family_name="PIB",
            source_url="https://example.com",
            target_series_id="F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T",
            target_series_title="PIB a precios corrientes",
            family_series=[],
        )

    monkeypatch.setattr(data_node_module, "lookup_series", _fake_lookup)

    monkeypatch.setattr(
        data_node_module,
        "load_observations",
        lambda *args, **kwargs: {
            "F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T": {
                "meta": {"original_frequency": "Q"},
                "observations": [{"date": "2025-09-30", "value": 100.0}],
            }
        },
    )

    monkeypatch.setattr(data_node_module, "_emit_stream_chunk", lambda chunk, writer: None)
    monkeypatch.setattr(data_node_module, "stream_data_response", lambda payload: iter(["ok"]))

    node = data_node_module.make_data_node(memory_adapter=None)
    node({"question": "Cuánto es el PIB de Chile?"}, writer=None)

    assert lookup_capture["indicator_ent"] == "pib"
    assert lookup_capture["price"] == "co"


def test_resolve_data_node_overrides_reads_json_toggles(monkeypatch, tmp_path):
    cfg = tmp_path / "rule_toggles.json"
    cfg.write_text(
        '{"rules": {"pib_corrientes": false, "pib_share": false, "pib_per_capita": false}}',
        encoding="utf-8",
    )

    monkeypatch.setenv("PIBOT_RULE_TOGGLES_PATH", str(cfg))
    monkeypatch.delenv("ENABLE_RULE_PIB_CORRIENTES", raising=False)
    monkeypatch.delenv("ENABLE_RULE_PIB_SHARE", raising=False)
    monkeypatch.delenv("ENABLE_RULE_PIB_SHARE_GASTO_GUARDRAIL", raising=False)
    monkeypatch.delenv("ENABLE_RULE_PIB_PER_CAPITA", raising=False)

    corrientes = resolve_data_node_overrides(
        question="Cual es el valor del PIB a precios corrientes del 2020",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="point",
        frequency_ent="a",
    )
    assert corrientes == {}

    share = resolve_data_node_overrides(
        question="Cuanto pesa mineria en el PIB?",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="latest",
        frequency_ent="q",
    )
    assert share == {}

    per_capita = resolve_data_node_overrides(
        question="Cual es el PIB per capita del 2025",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="point",
        frequency_ent="q",
    )
    assert per_capita == {}

    # La variable de entorno tiene prioridad por sobre el JSON.
    monkeypatch.setenv("ENABLE_RULE_PIB_CORRIENTES", "1")
    corrientes_env = resolve_data_node_overrides(
        question="Cual es el valor del PIB a precios corrientes del 2020",
        indicator_ent="pib",
        calc_mode_cls="original",
        req_form_cls="point",
        frequency_ent="a",
    )
    assert corrientes_env.get("price") == "co"


def test_data_node_applies_pib_per_capita_override_before_lookup(monkeypatch):
    monkeypatch.setenv("ENABLE_RULE_PIB_PER_CAPITA", "1")

    ent = ResolvedEntities(
        indicator_ent="pib",
        frequency_ent="q",
        calc_mode_cls="original",
        seasonality_ent="nsa",
        activity_ent=None,
        activity_cls="none",
        activity_cls_resolved="none",
        region_ent="antofagasta",
        region_cls="specific",
        investment_ent="consumo",
        investment_cls="specific",
        req_form_cls="latest",
        price="enc",
    )

    monkeypatch.setattr(
        data_node_module,
        "_extract_entities_from_state",
        lambda state: (state.get("question", ""), [], ent),
    )
    monkeypatch.setattr(data_node_module, "apply_business_rules", lambda _ent: _ent)

    lookup_capture: dict[str, object] = {}

    def _fake_lookup(_ent):
        lookup_capture["activity_ent"] = _ent.activity_ent
        lookup_capture["activity_cls"] = _ent.activity_cls
        lookup_capture["activity_cls_resolved"] = _ent.activity_cls_resolved
        lookup_capture["frequency_ent"] = _ent.frequency_ent
        lookup_capture["price"] = _ent.price
        lookup_capture["region_ent"] = _ent.region_ent
        lookup_capture["region_cls"] = _ent.region_cls
        lookup_capture["investment_ent"] = _ent.investment_ent
        lookup_capture["investment_cls"] = _ent.investment_cls
        return SimpleNamespace(
            family_name="PIB per capita",
            source_url="https://example.com",
            target_series_id="F032.PIB.PP.Z.USD.2018.Z.Z.0.A",
            target_series_title="PIB per capita",
            family_series=[],
        )

    monkeypatch.setattr(data_node_module, "lookup_series", _fake_lookup)

    monkeypatch.setattr(
        data_node_module,
        "load_observations",
        lambda *args, **kwargs: {
            "F032.PIB.PP.Z.USD.2018.Z.Z.0.A": {
                "meta": {"original_frequency": "A"},
                "observations": [{"date": "2024-12-31", "value": 16586.36}],
            }
        },
    )

    streamed: list[str] = []
    monkeypatch.setattr(data_node_module, "_emit_stream_chunk", lambda chunk, writer: streamed.append(chunk))

    monkeypatch.setattr(data_node_module, "stream_data_response", lambda payload: iter(["ok"]))

    node = data_node_module.make_data_node(memory_adapter=None)
    result = node({"question": "Cual es el PIB per capita"}, writer=None)

    assert lookup_capture["activity_ent"] == "pib_percapita"
    assert lookup_capture["activity_cls"] == "specific"
    assert lookup_capture["activity_cls_resolved"] == "specific"
    assert lookup_capture["frequency_ent"] == "a"
    assert lookup_capture["price"] is None
    assert lookup_capture["region_ent"] is None
    assert lookup_capture["region_cls"] == "none"
    assert lookup_capture["investment_ent"] is None
    assert lookup_capture["investment_cls"] == "none"
    assert result["series"] == "F032.PIB.PP.Z.USD.2018.Z.Z.0.A"
    assert result["data_classification"]["activity_ent"] == "pib_percapita"
    assert streamed and streamed[0] == result["output"]


def test_data_node_applies_corrientes_override_before_lookup(monkeypatch):
    monkeypatch.setenv("ENABLE_RULE_PIB_CORRIENTES", "1")

    ent = ResolvedEntities(
        indicator_ent="pib",
        frequency_ent="a",
        calc_mode_cls="original",
        activity_cls="none",
        activity_cls_resolved="none",
        region_cls="none",
        investment_cls="none",
        req_form_cls="point",
        period_ent=["2020-01-01", "2020-12-31"],
        price="enc",
    )

    monkeypatch.setattr(
        data_node_module,
        "_extract_entities_from_state",
        lambda state: (state.get("question", ""), [], ent),
    )

    def _fake_apply_business_rules(_ent):
        _ent.price = "enc"
        return _ent

    monkeypatch.setattr(data_node_module, "apply_business_rules", _fake_apply_business_rules)

    monkeypatch.setattr(
        data_node_module,
        "lookup_series",
        lambda _ent: SimpleNamespace(
            family_name="PIB corrientes",
            source_url="https://example.com",
            target_series_id="F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T",
            target_series_title="PIB a precios corrientes",
            family_series=[],
        ),
    )

    monkeypatch.setattr(
        data_node_module,
        "load_observations",
        lambda *args, **kwargs: {
            "F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T": {
                "meta": {"original_frequency": "Q"},
                "observations": [],
            }
        },
    )

    payload_holder: dict[str, object] = {}

    def _fake_stream_data_response(payload):
        payload_holder["payload"] = payload
        yield "ok"

    monkeypatch.setattr(data_node_module, "stream_data_response", _fake_stream_data_response)

    node = data_node_module.make_data_node(memory_adapter=None)
    result = node({"question": "Cual es el valor del PIB a precios corrientes del 2020"}, writer=None)

    payload = payload_holder["payload"]
    assert isinstance(payload, dict)
    assert payload["price"] == "co"
    assert result["data_classification"]["price"] == "co"


def test_data_node_keeps_share_gasto_short_circuit_deterministic(monkeypatch):
    monkeypatch.setenv("ENABLE_RULE_PIB_SHARE", "1")
    monkeypatch.setenv("ENABLE_RULE_PIB_SHARE_GASTO_GUARDRAIL", "1")

    ent = ResolvedEntities(
        indicator_ent="pib",
        frequency_ent="q",
        calc_mode_cls="original",
        activity_cls="none",
        activity_cls_resolved="none",
        region_cls="none",
        investment_cls="none",
        req_form_cls="latest",
        period_ent=["2025-01-01", "2025-03-31"],
        price="enc",
    )

    monkeypatch.setattr(
        data_node_module,
        "_extract_entities_from_state",
        lambda state: (state.get("question", ""), [], ent),
    )
    monkeypatch.setattr(data_node_module, "apply_business_rules", lambda _ent: _ent)

    def _never_called_stream_data_response(_payload):
        raise AssertionError("stream_data_response no debe ejecutarse en short-circuit determinista")

    monkeypatch.setattr(data_node_module, "stream_data_response", _never_called_stream_data_response)

    streamed: list[str] = []
    monkeypatch.setattr(data_node_module, "_emit_stream_chunk", lambda chunk, writer: streamed.append(chunk))

    node = data_node_module.make_data_node(memory_adapter=None)
    result = node({"question": "Cuánto pesa el consumo en el PIB?"}, writer=None)

    output = str(result.get("output") or "")
    assert "CCNN_EP18_03_ratio" in output
    assert "las series las puedes revisar en el siguiente cuadro" in output.lower()
    assert result["series"] is None
    assert streamed and streamed[0] == output
