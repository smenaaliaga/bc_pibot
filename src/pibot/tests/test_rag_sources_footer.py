from orchestrator.graph.nodes.llm import make_rag_node
from orchestrator.llm import llm_adapter as llma
from orchestrator.llm.llm_adapter import LLMAdapter


class _Doc:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class _Retriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, payload):
        return self._docs


def test_llm_adapter_collects_last_rag_sources_from_doc_metadata():
    docs = [
        _Doc("ctx1", {"docname": "Doc A", "link": "https://example.com/a"}),
        _Doc("ctx2", {"docname": "Doc B", "link": "https://example.com/b"}),
    ]
    adapter = LLMAdapter(streaming=False, retriever=_Retriever(docs), mode="rag")

    adapter._build_messages("consulta metodológica", history=[], intent_info={})

    sources = adapter.get_last_rag_sources()
    assert len(sources) == 2
    assert sources[0]["docname"] == "Doc A"
    assert sources[0]["link"] == "https://example.com/a"


def test_rag_node_appends_footer_with_deduplicated_top2_sources():
    class _FakeAdapter:
        def stream(self, question, history=None, intent_info=None):
            yield "Respuesta metodológica base."

        def get_last_rag_sources(self):
            return [
                {"docname": "Doc A", "link": "https://example.com/a"},
                {"docname": "Doc B", "link": "https://example.com/b"},
                {"docname": "Doc A", "link": "https://example.com/a"},
                {"docname": "Doc C", "link": "https://example.com/c"},
                {"docname": "Doc D", "link": "https://example.com/d"},
            ]

    chunks = []

    def _writer(payload):
        value = payload.get("stream_chunks")
        if value:
            chunks.append(value)

    rag_node = make_rag_node(_FakeAdapter())
    result = rag_node(
        {
            "question": "explica metodología del pib",
            "conversation_history": [],
            "intent": {"intent": "method"},
            "route_decision": "rag",
        },
        writer=_writer,
    )

    output = result.get("output", "")
    assert "Respuesta metodológica base." in output
    assert "Para mayor información, puedes consultar" in output
    assert "[Doc A](https://example.com/a)" in output
    assert "[Doc B](https://example.com/b)" in output
    assert "[Doc C](https://example.com/c)" not in output
    assert "[Doc D](https://example.com/d)" not in output
    assert output.count("[Doc A](https://example.com/a)") == 1
    assert chunks[-1].startswith("\n\nPara mayor información")


def test_llm_adapter_marks_insufficient_evidence_when_retrieval_empty():
    adapter = LLMAdapter(streaming=False, retriever=_Retriever([]), mode="rag")

    adapter._build_messages("consulta metodológica", history=[], intent_info={})

    assert adapter.rag_context_is_sufficient() is False


def test_llm_adapter_stream_returns_safe_reply_when_evidence_insufficient(monkeypatch):
    adapter = LLMAdapter(streaming=False, retriever=_Retriever([]), mode="rag")

    monkeypatch.setattr(llma, "LANGCHAIN_AVAILABLE", True)

    class _UnusedLLM:
        def invoke(self, _msgs):
            raise AssertionError("No debería invocar LLM cuando no hay evidencia suficiente")

    monkeypatch.setattr(llma, "init_chat_model", lambda **kwargs: _UnusedLLM())

    output = "".join(adapter.stream("explica metodología", history=[], intent_info={}))

    assert "No cuento con evidencia documental suficiente" in output


def test_rag_node_does_not_append_footer_when_generation_fails():
    class _FailingAdapter:
        def stream(self, question, history=None, intent_info=None):
            raise RuntimeError("upstream llm error")

        def get_last_rag_sources(self):
            return [
                {"docname": "Doc A", "link": "https://example.com/a"},
                {"docname": "Doc B", "link": "https://example.com/b"},
            ]

    rag_node = make_rag_node(_FailingAdapter())
    result = rag_node(
        {
            "question": "explica metodología del pib",
            "conversation_history": [],
            "intent": {"intent": "method"},
            "route_decision": "rag",
        }
    )

    output = result.get("output", "")
    assert output.startswith("Tuve un problema generando la respuesta.")
    assert "Para mayor información, puedes consultar" not in output


def test_rag_node_does_not_duplicate_existing_footer_block():
    class _FooterInModelAdapter:
        def stream(self, question, history=None, intent_info=None):
            yield (
                "Respuesta metodológica base."
                "\n\nPara mayor información, puedes consultar los documentos disponibles en la web oficial del Banco Central de Chile:\n"
                "- [Doc A](https://example.com/a)\n"
                "- [Doc B](https://example.com/b)"
            )

        def get_last_rag_sources(self):
            return [
                {"docname": "Doc A", "link": "https://example.com/a"},
                {"docname": "Doc B", "link": "https://example.com/b"},
            ]

    rag_node = make_rag_node(_FooterInModelAdapter())
    result = rag_node(
        {
            "question": "explica metodología del pib",
            "conversation_history": [],
            "intent": {"intent": "method"},
            "route_decision": "rag",
        }
    )

    output = result.get("output", "")
    assert output.count("Para mayor información, puedes consultar") == 1
    assert output.count("[Doc A](https://example.com/a)") == 1
    assert output.count("[Doc B](https://example.com/b)") == 1


