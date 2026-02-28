import types

from orchestrator.classifier import classifier_agent as ca


def test_predict_raw_uses_interpretation(monkeypatch):
    predict_response = {
        "text": "top-level",
        "interpretation": {
            "text": "interp-text",
            "words": ["cual", "fue", "el", "imacec"],
            "slot_tags": ["O", "O", "O", "B-indicator"],
            "entities": {"indicator": ["imacec"]},
        },
    }
    def fake_post_json(url, payload, timeout=None):
        return predict_response

    monkeypatch.setattr(ca, "PREDICT_URL", "predict-url")
    monkeypatch.setattr(ca, "post_json", fake_post_json)

    result = ca._classify_with_jointbert("q")

    assert result.predict_raw == predict_response
    assert result.predict_raw["interpretation"] == predict_response["interpretation"]
    assert result.words == predict_response["interpretation"]["words"]
    assert result.slot_tags == predict_response["interpretation"]["slot_tags"]
    assert result.entities == {}
    assert result.text == "interp-text"


def test_macro_zero_from_intent_api_is_not_overwritten_by_routing(monkeypatch):
    predict_response = {
        "text": "q",
        "interpretation": {
            "entities_normalized": {"indicator": ["imacec"]},
        },
        "routing": {
            "macro": {"label": 1, "confidence": 0.95},
            "intent": {"label": "value", "confidence": 0.95},
            "context": {"label": "standalone", "confidence": 0.9},
        },
    }
    def fake_post_json(url, payload, timeout=None):
        return predict_response

    monkeypatch.setattr(ca, "PREDICT_URL", "predict-url")
    monkeypatch.setattr(ca, "post_json", fake_post_json)

    result = ca._classify_with_jointbert("q")

    assert result.macro == 1
    assert result.intent == "value"


def test_keeps_other_when_intent_api_says_other_even_with_predict_value(monkeypatch):
    predict_response = {
        "text": "q",
        "routing": {
            "macro": {"label": 1, "confidence": 0.97},
            "intent": {"label": "value", "confidence": 0.97},
            "context": {"label": "standalone", "confidence": 0.89},
        },
        "interpretation": {
            "entities_normalized": {
                "indicator": ["imacec"],
                "seasonality": ["sa"],
                "frequency": ["m"],
            },
            "intents": {
                "calc_mode": {"label": "prev_period", "confidence": 0.99},
                "req_form": {"label": "latest", "confidence": 0.88},
            },
        },
    }
    def fake_post_json(url, payload, timeout=None):
        return predict_response

    monkeypatch.setattr(ca, "PREDICT_URL", "predict-url")
    monkeypatch.setattr(ca, "post_json", fake_post_json)

    result = ca._classify_with_jointbert("q")

    assert result.intent == "value"
    assert result.macro == 1
    assert result.req_form == "latest"
    assert result.calc_mode == "prev_period"