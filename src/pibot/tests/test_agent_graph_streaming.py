import types

from orchestrator.graph import agent_graph as ag
from orchestrator.classifier.intent_memory import IntentRecord


class DummyChunk:
    def __init__(self, text: str):
        delta = types.SimpleNamespace(content=text)
        choice = types.SimpleNamespace(delta=delta)
        self.choices = [choice]


def test_yield_openai_stream_chunks_iterable():
    stream = [DummyChunk("hola "), DummyChunk("mundo")]
    pieces = list(ag._yield_openai_stream_chunks(stream))
    assert pieces == ["hola ", "mundo"]


def test_yield_openai_stream_chunks_non_iterable_with_content():
    message = types.SimpleNamespace(content="completo")
    choice = types.SimpleNamespace(message=message)
    stream = types.SimpleNamespace(choices=[choice])
    assert list(ag._yield_openai_stream_chunks(stream)) == ["completo"]


def test_yield_openai_stream_chunks_non_iterable_no_content():
    class NoIter:
        pass

    assert list(ag._yield_openai_stream_chunks(NoIter())) == []


def test_ingest_node_captures_user_turn_id(monkeypatch):
    class StubMemory:
        def __init__(self):
            self.user_turns = []

        def on_user_turn(self, session_id, message, *, metadata=None):
            self.user_turns.append((session_id, message))
            return 42

        def get_window_for_llm(self, session_id):
            return []

    stub_memory = StubMemory()
    monkeypatch.setattr(ag, "_MEMORY", stub_memory, raising=False)
    state = {"question": "Hola", "history": [], "context": {"session_id": "sess-1"}}
    result = ag.ingest_node(state)
    assert result["user_turn_id"] == 42


def test_classify_node_persists_intent_event(monkeypatch):
    class StubStore:
        def __init__(self):
            self.calls = []

        def record(
            self,
            session_id,
            intent,
            score,
            *,
            spans=None,
            entities=None,
            intent_raw=None,
            predict_raw=None,
            turn_id=0,
            model_version=None,
        ):
            self.calls.append(
                {
                    "session_id": session_id,
                    "intent": intent,
                    "score": score,
                    "spans": spans or [],
                    "entities": entities or {},
                    "intent_raw": intent_raw or {},
                    "predict_raw": predict_raw or {},
                    "turn_id": turn_id,
                }
            )

    stub_store = StubStore()
    monkeypatch.setattr(ag, "_INTENT_STORE", stub_store, raising=False)

    def fake_classify(question, history):
        return types.SimpleNamespace(query_type="DATA"), "hist"

    def fake_build_intent_info(cls):
        return {
            "intent": "ask_data",
            "score": 0.75,
            "spans": [{"text": "foo", "label": "O", "start": 0, "end": 3}],
            "entities": {"domain": "IMACEC"},
            "intent_raw": {"intent": "ask_data", "source": "router"},
            "predict_raw": {"entities": {"domain": ["IMACEC"]}},
        }
    monkeypatch.setattr(ag, "classify_question_with_history", fake_classify)
    monkeypatch.setattr(ag, "build_intent_info", fake_build_intent_info)

    state = {"question": "hola", "history": [], "session_id": "sess-2", "user_turn_id": 99}
    ag.classify_node(state)

    assert len(stub_store.calls) == 1
    call = stub_store.calls[0]
    assert call["session_id"] == "sess-2"
    assert call["intent"] == "ask_data"
    assert call["turn_id"] == 99
    assert call["entities"] == {"domain": "IMACEC"}
    assert call["intent_raw"] == {"intent": "ask_data", "source": "router"}
    assert call["predict_raw"] == {"entities": {"domain": ["IMACEC"]}}


def test_intent_node_uses_store_for_followup_inheritance(monkeypatch):
    class StubStore:
        def history(self, session_id, k=10):
            return [
                IntentRecord(
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
                                "seasonality": ["nsa"],
                                "frequency": ["q"],
                                "period": ["2025-10-01", "2025-12-31"],
                            }
                        }
                    },
                    turn_id=3,
                )
            ]

    monkeypatch.setattr(ag, "_MEMORY", object(), raising=False)
    monkeypatch.setattr(ag, "_INTENT_STORE", StubStore(), raising=False)

    classification = types.SimpleNamespace(
        intent="value",
        context="followup",
        macro=0,
        intent_raw={"routing": {"macro": {"label": 0}, "intent": {"label": "value"}, "context": {"label": "followup"}}},
        predict_raw={
            "interpretation": {
                "entities": {},
                "slot_tags": ["O", "O", "O", "O"],
                "entities_normalized": {
                    "indicator": ["imacec"],
                    "seasonality": ["nsa"],
                    "frequency": ["m"],
                    "activity": [],
                    "region": [],
                    "investment": [],
                    "period": ["2026-01-01", "2026-01-31"],
                },
            }
        },
    )

    result = ag.intent_node(
        {
            "question": "puedes darme mas detalles",
            "session_id": "sess-followup",
            "user_turn_id": 5,
            "classification": classification,
            "entities": [],
        }
    )

    assert result["route_decision"] == "rag"
    assert result["intent"]["intent_cls"] == "method"
    assert result["intent"]["macro_cls"] == 1
    assert result["entities"][0]["indicator"] == "pib"
