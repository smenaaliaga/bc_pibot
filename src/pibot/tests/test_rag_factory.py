import types

import orchestrator.rag.rag_factory as rf


def test_create_retriever_prefers_pgvector_by_default(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("OPENAI_MODEL", "dummy")
    monkeypatch.delenv("RAG_BACKEND", raising=False)

    calls = []

    def fake_embeddings():
        return "emb"

    def fake_pg(emb):
        calls.append("pg")
        return "pg-ret"

    def fake_faiss(emb):
        calls.append("faiss")
        return None

    def fake_chroma(emb):
        calls.append("chroma")
        return None

    monkeypatch.setattr(rf, "_embeddings", fake_embeddings)
    monkeypatch.setattr(rf, "_pgvector_retriever", fake_pg)
    monkeypatch.setattr(rf, "_faiss_retriever", fake_faiss)
    monkeypatch.setattr(rf, "_chroma_retriever", fake_chroma)

    retriever = rf.create_retriever()
    assert retriever == "pg-ret"
    assert calls == ["pg"]


def test_create_retriever_respects_backend_order(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("OPENAI_MODEL", "dummy")
    monkeypatch.setenv("RAG_BACKEND", "faiss")

    calls = []

    def fake_embeddings():
        return "emb"

    def fake_pg(emb):
        calls.append("pg")
        return "pg-ret"

    def fake_faiss(emb):
        calls.append("faiss")
        return None

    def fake_chroma(emb):
        calls.append("chroma")
        return None

    monkeypatch.setattr(rf, "_embeddings", fake_embeddings)
    monkeypatch.setattr(rf, "_pgvector_retriever", fake_pg)
    monkeypatch.setattr(rf, "_faiss_retriever", fake_faiss)
    monkeypatch.setattr(rf, "_chroma_retriever", fake_chroma)

    retriever = rf.create_retriever()
    assert retriever == "pg-ret"
    # Order should start with faiss when RAG_BACKEND=faiss
    assert calls[:2] == ["faiss", "pg"]
