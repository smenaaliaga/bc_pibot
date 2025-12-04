import types
import sys
import os

from orchestrator.langchain_memory import LangChainMemoryAdapter


class FakeEmbeddings:
    def __init__(self, model: str = None):
        self.model = model

    def embed_query(self, text: str):
        # Simple deterministic vector based on length
        l = max(1, len(text))
        return [1.0, float(l % 5), 0.5]


class FakeDiversityManager:
    def __init__(self, embeddings=None, llm=None, similarity_threshold=0.9, max_history=50):
        self.embeddings = embeddings
        self.registered = []
        self.similarity_threshold = similarity_threshold

    def check_redundancy(self, candidate: str):
        # Flag redundancy when candidate contains the token "repeated"
        if "repeated" in candidate.lower():
            return True, self.similarity_threshold + 0.05
        return False, 0.0

    def register_response(self, text: str):
        self.registered.append(text)


def test_diversity_with_embeddings(monkeypatch):
    # Force thresholds low for test
    monkeypatch.setenv("DIVERSITY_MIN_TURNS", "1")
    monkeypatch.setenv("DIVERSITY_MIN_LEN", "1")
    monkeypatch.setenv("DIVERSITY_SIM_THRESHOLD", "0.6")
    monkeypatch.setenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

    # Patch langchain_openai import to use FakeEmbeddings
    fake_module = types.SimpleNamespace(OpenAIEmbeddings=FakeEmbeddings)
    sys.modules["langchain_openai"] = fake_module

    # Patch ResponseDiversityManager to a fake deterministic one
    from orchestrator import langchain_memory as lm

    monkeypatch.setattr(lm, "ResponseDiversityManager", FakeDiversityManager, raising=False)

    adapter = LangChainMemoryAdapter(pg_dsn=None)
    # Forzar manager fake en caso de que no haya inicializado
    if not adapter._div_mgr:
        adapter._div_mgr = FakeDiversityManager(similarity_threshold=0.6)

    # Seed one assistant turn so min_turns is satisfied
    adapter.on_assistant_turn("sess", "primer mensaje")

    is_red, sim = adapter.diversity_check("sess", "mensaje repeated de prueba")
    assert is_red is True
    assert sim > 0.6

    # Register and ensure manager stored it
    adapter.diversity_register("sess", "mensaje repeated de prueba")
    mgr = adapter._div_mgr
    assert isinstance(mgr, FakeDiversityManager)
    assert mgr.registered, "La respuesta debería registrarse en el gestor de diversidad"
