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


def test_followup_first_turn_fallback():
    intent_raw = {"context": {"label": "followup"}, "intent": {"label": "value"}}
    predict_raw = {"entities_normalized": {"indicator": None}}
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=1,
        intent_raw=intent_raw,
        predict_raw=predict_raw,
    )
    intent_store = StubIntentStore([])
    node = make_intent_node(None, intent_store)

    state = {
        "question": "Graficalo",
        "session_id": "sess-1",
        "user_turn_id": 1,
        "classification": classification,
        "entities": [],
    }

    result = node(state)

    assert result["route_decision"] == "fallback"


def test_followup_value_region_backfills_indicator_and_time_fields():
    prev_intent_raw = {"intent": {"label": "value"}, "macro": {"label": 1}}
    prev_predict_raw = {
        "entities_normalized": {
            "indicator": "imacec",
            "period": "2023",
            "frequency": "m",
            "seasonality": "sa",
        }
    }
    prev_record = IntentRecord(
        intent="value",
        score=0.9,
        intent_raw=prev_intent_raw,
        predict_raw=prev_predict_raw,
        turn_id=1,
    )

    intent_raw = {"context": {"label": "followup"}, "intent": {"label": "value"}}
    predict_raw = {
        "intents": {"region": {"label": "specific"}},
        "entities_normalized": {"region": ["metropolitana"], "indicator": None},
    }
    classification = _make_classification(
        intent="value",
        context="followup",
        macro=1,
        intent_raw=intent_raw,
        predict_raw=predict_raw,
    )
    intent_store = StubIntentStore([prev_record])
    node = make_intent_node(None, intent_store)

    state = {
        "question": "y en la metropolitana?",
        "session_id": "sess-2",
        "user_turn_id": 2,
        "classification": classification,
        "entities": [],
    }

    result = node(state)

    assert result["route_decision"] == "data"
    assert result["entities"][0]["indicator"] == "imacec"
    assert result["entities"][0]["period"] == "2023"
    assert result["entities"][0]["seasonality"] == "sa"
    updated_predict = result["intent_info"]["predict_raw"]["entities_normalized"]
    assert updated_predict["indicator"] == "imacec"
    assert updated_predict["frequency"] == "m"


def test_followup_method_backfills_indicator():
    prev_intent_raw = {"intent": {"label": "method"}, "macro": {"label": 1}}
    prev_predict_raw = {"entities_normalized": {"indicator": "pib"}}
    prev_record = IntentRecord(
        intent="method",
        score=0.9,
        intent_raw=prev_intent_raw,
        predict_raw=prev_predict_raw,
        turn_id=3,
    )

    intent_raw = {"context": {"label": "followup"}, "intent": {"label": "method"}}
    predict_raw = {"entities_normalized": {"indicator": None}}
    classification = _make_classification(
        intent="method",
        context="followup",
        macro=1,
        intent_raw=intent_raw,
        predict_raw=predict_raw,
    )
    intent_store = StubIntentStore([prev_record])
    node = make_intent_node(None, intent_store)

    state = {
        "question": "y la metodologia?",
        "session_id": "sess-3",
        "user_turn_id": 4,
        "classification": classification,
        "entities": [],
    }

    result = node(state)

    assert result["route_decision"] == "rag"
    assert result["entities"][0]["indicator"] == "pib"


def test_followup_other_macro_zero_recovers_previous_method_and_indicator():
    prev_intent_raw = {
        "intent": {"label": "method"},
        "macro": {"label": 1},
        "context": {"label": "standalone"},
    }
    prev_predict_raw = {
        "entities_normalized": {
            "indicator": ["pib"],
            "seasonality": ["nsa"],
            "frequency": ["q"],
            "period": "01-01-2026",
        }
    }
    prev_record = IntentRecord(
        intent="method",
        score=0.9,
        intent_raw=prev_intent_raw,
        predict_raw=prev_predict_raw,
        turn_id=10,
    )

    intent_raw = {
        "context": {"label": "followup"},
        "intent": {"label": "other"},
        "macro": {"label": 0},
    }
    predict_raw = {
        "entities": {},
        "slot_tags": ["O", "O", "O"],
        "entities_normalized": {
            "indicator": ["imacec"],
            "seasonality": ["nsa"],
            "frequency": ["m"],
            "period": "01-02-2026",
        },
    }
    classification = _make_classification(
        intent="other",
        context="followup",
        macro=0,
        intent_raw=intent_raw,
        predict_raw=predict_raw,
    )
    intent_store = StubIntentStore([prev_record])
    node = make_intent_node(None, intent_store)

    state = {
        "question": "detalles fuentes de información anterior",
        "session_id": "sess-4",
        "user_turn_id": 11,
        "classification": classification,
        "entities": [],
    }

    result = node(state)

    assert result["route_decision"] == "rag"
    assert result["intent"]["intent_cls"] == "method"
    assert result["entities"][0]["indicator"] == "pib"
