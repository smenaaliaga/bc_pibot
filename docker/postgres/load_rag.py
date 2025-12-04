"""
Load cleaned methodology docs into PGVector inside the Postgres container.

Assumes:
- PGVector extension is available.
- OPENAI_API_KEY present in env.
- Docs copied to /rag/docs during image build.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from langchain_openai import OpenAIEmbeddings  # type: ignore
from langchain_postgres import PGVector as PGVectorCls  # type: ignore
import psycopg  # type: ignore


def _default_dsn() -> str:
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    db = os.getenv("POSTGRES_DB", "pibot")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def chunk_text(text: str, max_chars: int = 400) -> List[str]:
    chunks: List[str] = []
    for i in range(0, len(text), max_chars):
        chunk = text[i : i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def purge_collection(dsn: str, collection: str) -> None:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT uuid FROM langchain_pg_collection WHERE name = %s",
                (collection,),
            )
            rows = cur.fetchall()
            if rows:
                cur.execute(
                    "DELETE FROM langchain_pg_embedding WHERE collection_id = ANY(%s)",
                    ([r[0] for r in rows],),
                )
                cur.execute(
                    "DELETE FROM langchain_pg_collection WHERE name = %s",
                    (collection,),
                )


def _infer_topic(path: Path) -> str:
    name = path.name.lower()
    if "imacec" in name:
        return "imacec"
    if "see100" in name or "season" in name:
        return "seasonality"
    return "pib"


def main() -> None:
    docs_dir = Path("/rag/docs")
    if not docs_dir.exists():
        print("Docs directory not found, skipping RAG load.")
        return

    dsn = os.getenv("PG_DSN", _default_dsn())
    collection = os.getenv("RAG_PGVECTOR_COLLECTION", "methodology")
    model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
    clear = os.getenv("RAG_CLEAR_COLLECTION", "true").lower() in {"1", "true", "yes"}

    if clear:
        try:
            purge_collection(dsn, collection)
            print(f"Purged collection '{collection}'")
        except Exception as exc:  # pragma: no cover
            print(f"Warning: could not purge collection '{collection}': {exc}")

    emb = OpenAIEmbeddings(model=model)
    vs = PGVectorCls(
        embeddings=emb,
        collection_name=collection,
        connection=dsn,
        use_jsonb=True,
        create_extension=False,
    )

    files = sorted(docs_dir.glob("*.txt"))
    total = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        topic = _infer_topic(path)
        metadatas: List[Dict[str, str]] = [
            {"source": str(path), "topic": topic, "section": f"{topic}-{idx}"}
            for idx, _ in enumerate(chunks)
        ]
        vs.add_texts(chunks, metadatas=metadatas)
        total += len(chunks)
        print(f"Loaded {len(chunks)} chunks from {path.name} (topic={topic})")

    print(f"Done. Total chunks loaded: {total} into collection '{collection}'.")


if __name__ == "__main__":
    main()
