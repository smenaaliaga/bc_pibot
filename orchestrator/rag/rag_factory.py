"""Factory to build a LangChain retriever for methodological RAG context."""
from __future__ import annotations

import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception:
    OpenAIEmbeddings = None  # type: ignore

# Prefer langchain_postgres PGVector (new schema); fall back only to local FAISS/Chroma if missing.
try:
    from langchain_postgres import PGVector as NewPGVector  # type: ignore
except Exception:
    NewPGVector = None  # type: ignore
try:
    from langchain_community.vectorstores import FAISS, Chroma  # type: ignore
except Exception:
    FAISS = None  # type: ignore
    Chroma = None  # type: ignore


def _embeddings() -> Optional[Any]:
    if not OpenAIEmbeddings:
        return None
    model = os.getenv("OPENAI_EMBEDDINGS_MODEL") or os.getenv("OPENAI_MODEL")
    if not model:
        return None
    try:
        return OpenAIEmbeddings(model=model)
    except Exception:
        logger.exception("Failed to init OpenAIEmbeddings for RAG")
    return None


def _pgvector_retriever(emb: Any):
    conn = os.getenv("RAG_PGVECTOR_URL") or os.getenv("PG_DSN")
    collection = os.getenv("RAG_PGVECTOR_COLLECTION", "methodology")
    if not conn:
        return None
    create_ext = os.getenv("RAG_PGVECTOR_CREATE_EXTENSION", "false").lower() in {"1", "true", "yes"}
    use_jsonb = os.getenv("RAG_PGVECTOR_USE_JSONB", "true").lower() in {"1", "true", "yes"}
    if not NewPGVector:
        logger.warning("langchain_postgres PGVector not available; skipping PG backend")
        return None
    try:
        vs = NewPGVector(
            embeddings=emb,
            collection_name=collection,
            connection=conn,
            use_jsonb=use_jsonb,
            create_extension=create_ext,
        )
        k = int(os.getenv("RAG_TOP_K", "20"))
        return vs.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        if "extension \"vector\" is not available" in str(e).lower():
            logger.warning("PGVector extension not available; falling back to other RAG backends")
            return None
        logger.exception("Failed to build PGVector retriever")
        return None


def _faiss_retriever(emb: Any):
    if not FAISS:
        return None
    path = os.getenv("RAG_FAISS_PATH")
    if not path:
        return None
    try:
        vs = FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
        k = int(os.getenv("RAG_TOP_K", "20"))
        return vs.as_retriever(search_kwargs={"k": k})
    except Exception:
        logger.exception("Failed to build FAISS retriever")
        return None


def _chroma_retriever(emb: Any):
    if not Chroma:
        return None
    path = os.getenv("RAG_CHROMA_PATH")
    if not path:
        return None
    try:
        vs = Chroma(persist_directory=path, embedding_function=emb)
        k = int(os.getenv("RAG_TOP_K", "20"))
        return vs.as_retriever(search_kwargs={"k": k})
    except Exception:
        logger.exception("Failed to build Chroma retriever")
        return None


def create_retriever() -> Optional[Any]:
    """
    Build a retriever for RAG if enabled and configured. Tries PGVector, then FAISS, then Chroma.
    Returns None if disabled or misconfigured.
    """
    enabled = os.getenv("RAG_ENABLED", "true").lower() not in {"0", "false", "no"}
    if not enabled:
        return None
    emb = _embeddings()
    if not emb:
        logger.warning("RAG enabled but no embeddings available; skipping retriever")
        return None
    backend = os.getenv("RAG_BACKEND", "").lower()
    k = int(os.getenv("RAG_TOP_K", "20"))
    order = []
    if backend == "pgvector":
        order = [_pgvector_retriever, _faiss_retriever, _chroma_retriever]
    elif backend == "faiss":
        order = [_faiss_retriever, _pgvector_retriever, _chroma_retriever]
    elif backend == "chroma":
        order = [_chroma_retriever, _pgvector_retriever, _faiss_retriever]
    else:
        order = [_pgvector_retriever, _faiss_retriever, _chroma_retriever]

    retriever = None
    for fn in order:
        retriever = fn(emb)
        if retriever:
            try:
                retriever.search_kwargs = getattr(retriever, "search_kwargs", {})
                retriever.search_kwargs.update({"k": k})
            except Exception:
                pass
            break
    if not retriever:
        logger.warning("RAG enabled but no retriever could be created; check configuration")
    return retriever
import warnings
