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
