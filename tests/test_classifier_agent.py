import types
import pytest

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

    assert result.predict_raw["text"] == predict_response["text"]
    assert result.predict_raw["interpretation"]["text"] == predict_response["interpretation"]["text"]
    assert result.words == predict_response["interpretation"]["words"]
    assert result.slot_tags == predict_response["interpretation"]["slot_tags"]
    assert result.entities["indicator"] == ["imacec"]
    assert result.predict_raw["interpretation"]["entities_normalized"]["indicator"] == ["imacec"]
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


def test_non_value_intent_keeps_empty_indicator_when_frequency_missing(monkeypatch):
    predict_response = {
        "text": "explica metodologia",
        "routing": {
            "macro": {"label": 2, "confidence": 0.97},
            "intent": {"label": "methodology", "confidence": 0.97},
            "context": {"label": "standalone", "confidence": 0.89},
        },
        "interpretation": {
            "entities": {
                "indicator": [],
                "frequency": [],
            },
            "intents": {
                "calc_mode": {"label": "original", "confidence": 0.99},
                "req_form": {"label": "point", "confidence": 0.88},
            },
        },
    }

    def fake_post_json(url, payload, timeout=None):
        return predict_response

    monkeypatch.setattr(ca, "PREDICT_URL", "predict-url")
    monkeypatch.setattr(ca, "post_json", fake_post_json)

    result = ca._classify_with_jointbert("q")

    assert result.intent == "methodology"
    assert result.entities["indicator"] == []
    assert result.entities["frequency"] == []


def test_req_form_point_is_coerced_to_range_when_period_spans_multiple_quarters(monkeypatch):
    predict_response = {
        "text": "pib trimestral 2020",
        "routing": {
            "macro": {"label": 1, "confidence": 0.95},
            "intent": {"label": "value", "confidence": 0.95},
            "context": {"label": "standalone", "confidence": 0.9},
        },
        "interpretation": {
            "entities": {
                "indicator": ["pib"],
                "frequency": ["trimestral"],
                "period": ["2020"],
            },
            "intents": {
                "calc_mode": {"label": "original", "confidence": 0.99},
                "req_form": {"label": "point", "confidence": 0.88},
            },
        },
    }

    def fake_post_json(url, payload, timeout=None):
        return predict_response

    monkeypatch.setattr(ca, "PREDICT_URL", "predict-url")
    monkeypatch.setattr(ca, "post_json", fake_post_json)

    result = ca._classify_with_jointbert("q")

    assert result.req_form == "range"


def test_predict_retries_once_on_timeout(monkeypatch):
    predict_response = {
        "routing": {
            "macro": {"label": 1, "confidence": 0.95},
            "intent": {"label": "value", "confidence": 0.95},
            "context": {"label": "standalone", "confidence": 0.9},
        },
        "interpretation": {
            "entities": {"indicator": ["imacec"]},
            "intents": {
                "calc_mode": {"label": "yoy", "confidence": 0.99},
                "req_form": {"label": "latest", "confidence": 0.88},
            },
        },
    }

    calls = []

    def fake_post_json(url, payload, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise RuntimeError("Failed to call predict-url: timed out")
        return predict_response

    monkeypatch.setattr(ca, "PREDICT_URL", "predict-url")
    monkeypatch.setattr(ca, "post_json", fake_post_json)
    monkeypatch.setattr(ca, "PREDICT_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(ca, "PREDICT_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(ca, "PREDICT_RETRY_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(ca, "PREDICT_RETRY_BACKOFF_SECONDS", 0.0)

    result = ca._classify_with_jointbert("q")

    assert result.intent == "value"
    assert len(calls) == 2
    assert calls[0] == 1.0
    assert calls[1] == 5.0


def test_predict_does_not_retry_on_non_transient_http_error(monkeypatch):
    calls = []

    def fake_post_json(url, payload, timeout=None):
        calls.append(timeout)
        raise RuntimeError("HTTP 400 calling predict-url: bad request")

    monkeypatch.setattr(ca, "PREDICT_URL", "predict-url")
    monkeypatch.setattr(ca, "post_json", fake_post_json)
    monkeypatch.setattr(ca, "PREDICT_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(ca, "PREDICT_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(ca, "PREDICT_RETRY_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(ca, "PREDICT_RETRY_BACKOFF_SECONDS", 0.0)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        ca._classify_with_jointbert("q")

    assert len(calls) == 1