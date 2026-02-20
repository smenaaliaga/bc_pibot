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
    intent_response = {"intent": "value", "macro": 1, "context": "standalone", "confidence": 0.94}

    def fake_post_json(url, payload, timeout=None):
        if url == "predict-url":
            return predict_response
        return intent_response

    monkeypatch.setattr(ca, "PREDICT_URL", "predict-url")
    monkeypatch.setattr(ca, "INTENT_CLASSIFIER_URL", "intent-url")
    monkeypatch.setattr(ca, "post_json", fake_post_json)

    result = ca._classify_with_jointbert("q")

    assert result.predict_raw == predict_response["interpretation"]
    assert result.words == predict_response["interpretation"]["words"]
    assert result.slot_tags == predict_response["interpretation"]["slot_tags"]
    assert result.entities == predict_response["interpretation"]["entities"]
    assert result.text == "interp-text"