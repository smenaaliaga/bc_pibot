from orchestrator.normalizer.followup_rules import resolve_followup_route


def test_first_turn_explicit_indicator_routes_standalone_data():
    result = resolve_followup_route(
        normalized_intent="value",
        context_label="followup",
        macro_label=1,
        current_turn_id=1,
        payload_root={
            "slot_tags": ["O", "B-indicator"],
            "entities": {"indicator": ["pib"]},
        },
        current_intents={},
        current_norm={"indicator": ["pib"]},
        prev_intent_raw={},
        prev_predict_raw={},
    )

    assert result["decision"] == "data"
    assert result["context_label"] == "standalone"


def test_macro_zero_recovers_previous_method_and_routes_rag():
    result = resolve_followup_route(
        normalized_intent="value",
        context_label="followup",
        macro_label=0,
        current_turn_id=3,
        payload_root={"entities": {}, "slot_tags": ["O", "O"]},
        current_intents={},
        current_norm={"indicator": ["imacec"]},
        prev_intent_raw={
            "routing": {
                "intent": {"label": "methodology"},
                "macro": {"label": 1},
                "context": {"label": "standalone"},
            }
        },
        prev_predict_raw={
            "interpretation": {
                "entities_normalized": {
                    "indicator": ["pib"],
                    "period": "2023",
                    "frequency": "q",
                    "seasonality": "nsa",
                }
            }
        },
    )

    assert result["decision"] == "rag"
    assert result["normalized_intent"] == "method"
    assert result["macro_label"] == 1
    assert result["current_norm"]["indicator"] == "pib"


def test_value_activity_specific_assigns_pib_and_backfills_time_fields():
    result = resolve_followup_route(
        normalized_intent="value",
        context_label="followup",
        macro_label=1,
        current_turn_id=4,
        payload_root={"entities": {}, "slot_tags": ["O", "O"]},
        current_intents={"activity": {"label": "specific"}},
        current_norm={
            "activity": "transporte",
            "indicator": None,
            "period": None,
            "frequency": None,
            "seasonality": None,
        },
        prev_intent_raw={
            "intent": {"label": "value"},
            "macro": {"label": 1},
            "context": {"label": "standalone"},
        },
        prev_predict_raw={
            "entities_normalized": {
                "indicator": "imacec",
                "period": "2023",
                "frequency": "m",
                "seasonality": "sa",
            }
        },
    )

    assert result["decision"] == "data"
    assert result["current_norm"]["indicator"] == "pib"
    assert result["current_norm"]["period"] == "2023"
    assert result["current_norm"]["frequency"] == "m"
    assert result["current_norm"]["seasonality"] == "sa"
