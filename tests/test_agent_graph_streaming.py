import types

from orchestrator.graph import agent_graph as ag


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


def test_stream_chunk_filter_skips_exact_duplicates():
    filt = ag._StreamChunkFilter()
    assert filt.allow("hola mundo") is True
    assert filt.allow("hola mundo") is False
    assert filt.allow("hola mundo ") is False
    assert filt.allow("nuevo") is True


def test_stream_chunk_filter_handles_whitespace_gaps():
    filt = ag._StreamChunkFilter()
    assert filt.allow("") is False
    assert filt.allow("   ") is False  # whitespace first -> ignored
    assert filt.allow("origen") is True
    assert filt.allow("\n") is True  # single whitespace after text -> allowed
    assert filt.allow("   \n ") is False  # consecutive whitespace suppressed
    assert filt.allow("nuevo bloque") is True


def test_ingest_node_captures_user_turn_id(monkeypatch):
    class StubMemory:
        def __init__(self):
            self.user_turns = []

        def on_user_turn(self, session_id, message, *, metadata=None):
            self.user_turns.append((session_id, message))
            return 42

        def get_facts(self, session_id):
            return {"foo": "bar"}

        def get_window_for_llm(self, session_id):
            return []

    stub_memory = StubMemory()
    monkeypatch.setattr(ag, "_MEMORY", stub_memory, raising=False)
    state = {"question": "Hola", "history": [], "context": {"session_id": "sess-1"}}
    result = ag.ingest_node(state)
    assert result["user_turn_id"] == 42
    assert result["facts"] == {"foo": "bar"}


def test_classify_node_persists_intent_event(monkeypatch):
    class StubStore:
        def __init__(self):
            self.calls = []

        def record(self, session_id, intent, score, *, spans=None, entities=None, turn_id=0, model_version=None):
            self.calls.append(
                {
                    "session_id": session_id,
                    "intent": intent,
                    "score": score,
                    "spans": spans or [],
                    "entities": entities or {},
                    "turn_id": turn_id,
                }
            )

    stub_store = StubStore()
    monkeypatch.setattr(ag, "_INTENT_STORE", stub_store, raising=False)

    def fake_classify(question, history):
        return types.SimpleNamespace(query_type="DATA"), "hist"

    def fake_build_intent_info(cls):
        return {"intent": "ask_data", "score": 0.75, "spans": [{"text": "foo", "label": "O", "start": 0, "end": 3}], "entities": {"domain": "IMACEC"}}

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
