import types

from orchestrator.classifier.intent_memory import IntentRecord
from orchestrator.graph.nodes.ingest import make_intent_node


class StubIntentStore:
    def __init__(self, records):
        self._records = list(records)

    def history(self, session_id, k=10):
        return self._records[-int(k) :]


def _make_classification(*, intent, context, macro, intent_raw, predict_raw):
    return types.SimpleNamespace(
        intent=intent,
        context=context,
        macro=macro,
        intent_raw=intent_raw,
        predict_raw=predict_raw,
        normalized={},
    )


def _build_prev_record(*, intent, macro, indicator="imacec", period="2023", frequency="m", seasonality="sa", turn_id=1):
    prev_intent_raw = {
        "intent": {"label": intent},
        "macro": {"label": macro},
        "context": {"label": "standalone"},
    }
    prev_predict_raw = {
        "entities_normalized": {
            "indicator": indicator,
            "period": period,
            "frequency": frequency,
            "seasonality": seasonality,
        }
    }
    return IntentRecord(
        intent=intent,
        score=0.9,
        intent_raw=prev_intent_raw,
        predict_raw=prev_predict_raw,
        turn_id=turn_id,
    )


def test_followup_first_turn_routes_fallback():
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "value"}},
        predict_raw={"entities_normalized": {"indicator": None}},
    )
    node = make_intent_node(None, StubIntentStore([]))

    result = node(
        {
            "question": "y eso?",
            "session_id": "s1",
            "user_turn_id": 1,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "fallback"


def test_followup_recovers_macro_and_intent_from_previous_turn():
    prev_record = _build_prev_record(intent="method", macro=1, indicator="pib", turn_id=3)
    classification = _make_classification(
        intent="none",
        context="followup",
        macro=0,
        intent_raw={
            "context": {"label": "followup"},
            "intent": {"label": "none"},
            "macro": {"label": 0},
        },
        predict_raw={"entities_normalized": {"indicator": None}},
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "y la metodología?",
            "session_id": "s2",
            "user_turn_id": 4,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "rag"
    assert result["intent"]["intent_cls"] == "method"
    assert result["intent"]["macro_cls"] == 1
    assert result["entities"][0]["indicator"] == "pib"


def test_followup_macro_zero_and_other_recovers_previous_method_intent():
    prev_record = _build_prev_record(intent="method", macro=1, indicator="pib", turn_id=5)
    classification = _make_classification(
        intent="other",
        context="followup",
        macro=0,
        intent_raw={
            "context": {"label": "followup"},
            "intent": {"label": "other"},
            "macro": {"label": 0},
        },
        predict_raw={"entities_normalized": {"indicator": None}},
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "hablame del enfoque anterior",
            "session_id": "s2b",
            "user_turn_id": 6,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "rag"
    assert result["intent"]["intent_cls"] == "method"
    assert result["intent"]["macro_cls"] == 1
    assert result["entities"][0]["indicator"] == "pib"


def test_followup_macro_zero_and_value_recovers_previous_method_intent_when_generic():
    prev_record = _build_prev_record(intent="method", macro=1, indicator="pib", turn_id=7)
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=0,
        intent_raw={
            "context": {"label": "followup"},
            "intent": {"label": "value"},
            "macro": {"label": 0},
        },
        predict_raw={
            "entities": {},
            "slot_tags": ["O", "O", "O"],
            "entities_normalized": {"indicator": ["imacec"]},
        },
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "dame más detalles",
            "session_id": "s2c",
            "user_turn_id": 8,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "rag"
    assert result["intent"]["intent_cls"] == "method"
    assert result["intent"]["macro_cls"] == 1
    assert result["entities"][0]["indicator"] == "pib"


def test_followup_macro_zero_recovers_previous_method_with_nested_routing_payload():
    prev_record = IntentRecord(
        intent="method",
        score=0.9,
        intent_raw={
            "routing": {
                "intent": {"label": "methodology"},
                "macro": {"label": 1},
                "context": {"label": "standalone"},
            }
        },
        predict_raw={
            "interpretation": {
                "entities_normalized": {
                    "indicator": ["pib"],
                    "period": "2023",
                    "frequency": "q",
                    "seasonality": "nsa",
                }
            }
        },
        turn_id=9,
    )
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=0,
        intent_raw={
            "routing": {
                "intent": {"label": "value"},
                "macro": {"label": 0},
                "context": {"label": "followup"},
            }
        },
        predict_raw={
            "interpretation": {
                "entities": {},
                "slot_tags": ["O", "O", "O"],
                "entities_normalized": {"indicator": ["imacec"]},
            }
        },
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "dame más detalles",
            "session_id": "s2d",
            "user_turn_id": 10,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "rag"
    assert result["intent"]["intent_cls"] == "method"
    assert result["intent"]["macro_cls"] == 1
    assert result["entities"][0]["indicator"] == "pib"


def test_followup_value_region_specific_backfills_indicator_and_time_fields():
    prev_record = _build_prev_record(intent="value", macro=1, indicator="imacec")
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "value"}},
        predict_raw={
            "intents": {"region": {"label": "specific"}},
            "entities_normalized": {
                "region": ["metropolitana"],
                "indicator": None,
                "period": None,
                "frequency": None,
                "seasonality": None,
            },
        },
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "y en la RM?",
            "session_id": "s3",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "data"
    assert result["entities"][0]["indicator"] == "imacec"
    assert result["entities"][0]["period"] == "2023"
    assert result["entities"][0]["frequency"] == "m"
    assert result["entities"][0]["seasonality"] == "sa"


def test_followup_value_activity_specific_sets_pib_when_indicator_missing():
    prev_record = _build_prev_record(intent="value", macro=1, indicator="imacec")
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "value"}},
        predict_raw={
            "intents": {"activity": {"label": "specific"}},
            "entities_normalized": {"activity": "transporte", "indicator": None},
        },
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "y transporte?",
            "session_id": "s4",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "data"
    assert result["entities"][0]["indicator"] == "pib"


def test_followup_value_activity_specific_sets_imacec_when_indicator_missing():
    prev_record = _build_prev_record(intent="value", macro=1, indicator="pib")
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "value"}},
        predict_raw={
            "intents": {"activity": {"label": "specific"}},
            "entities_normalized": {"activity": "bienes", "indicator": None},
        },
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "y bienes?",
            "session_id": "s5",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "data"
    assert result["entities"][0]["indicator"] == "imacec"


def test_followup_value_activity_comercio_uses_previous_indicator():
    prev_record = _build_prev_record(intent="value", macro=1, indicator="imacec")
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "value"}},
        predict_raw={
            "intents": {"activity": {"label": "specific"}},
            "entities_normalized": {"activity": "comercio", "indicator": None},
        },
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "y comercio?",
            "session_id": "s6",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "data"
    assert result["entities"][0]["indicator"] == "imacec"


def test_followup_value_investment_specific_uses_previous_indicator():
    prev_record = _build_prev_record(intent="value", macro=1, indicator="pib")
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "value"}},
        predict_raw={
            "intents": {"investment": {"label": "specific"}},
            "entities_normalized": {"investment": ["maquinaria"], "indicator": None},
        },
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "y la inversión en maquinaria?",
            "session_id": "s7",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "data"
    assert result["entities"][0]["indicator"] == "pib"


def test_followup_method_missing_indicator_uses_previous_indicator():
    prev_record = _build_prev_record(intent="method", macro=1, indicator="pib")
    classification = _make_classification(
        intent="method",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "method"}},
        predict_raw={"entities_normalized": {"indicator": None}},
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "y metodología?",
            "session_id": "s8",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "rag"
    assert result["entities"][0]["indicator"] == "pib"


def test_followup_method_without_explicit_indicator_uses_previous_indicator_over_default():
    prev_record = _build_prev_record(intent="method", macro=1, indicator="pib")
    classification = _make_classification(
        intent="method",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "method"}},
        predict_raw={
            "entities": {},
            "slot_tags": ["O", "O", "O"],
            "entities_normalized": {"indicator": ["imacec"]},
        },
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "puedes hablarme de las fuentes de informacion metodologia lo mismo",
            "session_id": "s8c",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "rag"
    assert result["entities"][0]["indicator"] == "pib"


def test_followup_method_missing_indicator_without_previous_routes_fallback():
    prev_record = _build_prev_record(intent="method", macro=1, indicator=None)
    classification = _make_classification(
        intent="method",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "method"}},
        predict_raw={"entities_normalized": {"indicator": None}},
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "y la metodología de eso?",
            "session_id": "s8b",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "fallback"


def test_followup_value_without_matching_rule_routes_fallback():
    prev_record = _build_prev_record(intent="value", macro=1, indicator="imacec")
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=1,
        intent_raw={"context": {"label": "followup"}, "intent": {"label": "value"}},
        predict_raw={
            "intents": {"activity": {"label": "general"}},
            "entities_normalized": {"indicator": None, "activity": "otro"},
        },
    )
    node = make_intent_node(None, StubIntentStore([prev_record]))

    result = node(
        {
            "question": "y eso cómo se ve?",
            "session_id": "s9",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "fallback"


def test_standalone_other_keeps_fallback_even_when_predict_signals_value():
    classification = _make_classification(
        intent="other",
        context="standalone",
        macro=1,
        intent_raw={
            "context": {"label": "standalone"},
            "intent": {"label": "other"},
            "macro": {"label": 1},
        },
        predict_raw={
            "routing": {
                "macro": {"label": 1, "confidence": 0.97},
                "intent": {"label": "value", "confidence": 0.96},
                "context": {"label": "standalone", "confidence": 0.89},
            },
            "intents": {
                "calc_mode": {"label": "prev_period", "confidence": 0.99},
                "req_form": {"label": "latest", "confidence": 0.88},
            },
            "entities_normalized": {"indicator": ["imacec"], "period": "01-02-2026"},
        },
    )
    node = make_intent_node(None, StubIntentStore([]))

    result = node(
        {
            "question": "cuanto aceleró la economía el último mes valor?",
            "session_id": "s10",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "fallback"
    assert result["intent"]["intent_cls"] == "other"
    assert result["entities"][0]["indicator"] == ["imacec"]


def test_standalone_other_keeps_fallback_when_signals_are_weak():
    classification = _make_classification(
        intent="other",
        context="standalone",
        macro=1,
        intent_raw={
            "context": {"label": "standalone"},
            "intent": {"label": "other"},
            "macro": {"label": 1},
        },
        predict_raw={
            "routing": {
                "macro": {"label": 1, "confidence": 0.97},
                "intent": {"label": "value", "confidence": 0.75},
                "context": {"label": "standalone", "confidence": 0.89},
            },
            "intents": {
                "calc_mode": {"label": "none", "confidence": 0.95},
                "req_form": {"label": "none", "confidence": 0.9},
            },
            "entities_normalized": {"indicator": ["imacec"]},
        },
    )
    node = make_intent_node(None, StubIntentStore([]))

    result = node(
        {
            "question": "economía ultimo mes",
            "session_id": "s11",
            "user_turn_id": 2,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "fallback"
    assert result["intent"]["intent_cls"] == "other"
