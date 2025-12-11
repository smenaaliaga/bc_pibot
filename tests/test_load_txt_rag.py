import json
from pathlib import Path

import pytest

from docker.postgres import load_txt_rag


@pytest.fixture()
def tmp_manifest(tmp_path: Path) -> Path:
    manifest = {
        "defaults": {
            "topic": "metodologia",
            "version": "v-test",
            "chunk_size": 100,
            "chunk_overlap": 20,
        },
        "documents": [
            {"id": "doc1", "path": "doc1.txt", "topic": "pib"},
            {"id": "doc2", "path": "doc2.txt"},
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_load_manifest(tmp_manifest: Path):
    defaults, docs = load_txt_rag.load_manifest(tmp_manifest)
    assert defaults.topic == "metodologia"
    assert defaults.chunk_size == 100
    assert len(docs) == 2
    assert docs[0].doc_id == "doc1"
    assert docs[0].topic == "pib"
    assert docs[1].topic == "metodologia"


def test_chunker_fallback():
    chunker = load_txt_rag.build_chunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker("12345678901234567890")
    assert chunks  # fallback keeps producing chunks
    assert all(len(c) <= 10 for c in chunks)


def test_should_accept_chunk_filters_language(monkeypatch):
    monkeypatch.setattr(load_txt_rag, "_langdetect", lambda text: "en")
    ok, reason = load_txt_rag.should_accept_chunk("hello world" * 20, "hash", set(), 5, "es")
    assert not ok and reason.startswith("lang_")


def test_generate_chunk_topics_handles_allowed_topics():
    text = "Metodología de cálculo y encadenamiento del PIB con deflactores"
    topics = load_txt_rag.generate_chunk_topics(text, "pib")
    assert set(topics) >= {"pib", "metodologia", "encadenamiento", "deflactores"}
