"""
Utility to load cleaned .txt documents into the RAG vector database (PGVector).

Usage:
    python tools/load_txt_rag.py

Config via env:
    PG_DSN                    (e.g., postgresql://postgres:postgres@localhost:5432/pibot)
    RAG_PGVECTOR_COLLECTION   (default: methodology)
    OPENAI_EMBEDDINGS_MODEL   (default: text-embedding-3-large)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # Si no está python-dotenv, simplemente seguimos; se espera que las vars estén en el entorno
    pass

try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"langchain_openai is required: {exc}")  # pragma: no cover

try:
    from langchain_postgres import PGVector as PGVectorCls  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"langchain_postgres PGVector is required: {exc}")  # pragma: no cover

try:
    import psycopg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"psycopg is required to purge existing collections: {exc}")  # pragma: no cover


# Files to load and a short topic tag for metadata
DOCS: Dict[str, str] = {
    r"E:\bc_pibot\docker\postgres\docs\ccnn_limpio.txt": "pib",
    r"E:\bc_pibot\docker\postgres\docs\Imacec_all_20251203_193937_limpio.txt": "imacec",
    r"E:\bc_pibot\docker\postgres\docs\see100_all_20251203_194549_limpio.txt": "seasonality",
}


def chunk_text(text: str, max_chars: int = 400) -> List[str]:
    chunks: List[str] = []
    for i in range(0, len(text), max_chars):
        chunk = text[i : i + max_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def purge_collection(dsn: str, collection: str) -> None:
    """Delete existing entries for the given collection in langchain_postgres schema."""
    try:
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
    except Exception as exc:
        print(f"Warning: could not purge collection '{collection}': {exc}")


def main() -> None:
    dsn = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot")
    collection = os.getenv("RAG_PGVECTOR_COLLECTION", "methodology")
    model = "text-embedding-3-large"
    clear = os.getenv("RAG_CLEAR_COLLECTION", "true").lower() in {"1", "true", "yes"}

    docs_dir = os.getenv("RAG_DOCS_DIR")
    dynamic_docs: Dict[str, str] = {}
    if docs_dir:
        base = Path(docs_dir)
        for p in sorted(base.glob("*.txt")):
            topic = "pib"
            name = p.name.lower()
            if "imacec" in name:
                topic = "imacec"
            elif "see100" in name or "season" in name:
                topic = "seasonality"
            dynamic_docs[str(p)] = topic
    doc_map = dynamic_docs or DOCS

    if clear:
        purge_collection(dsn, collection)

    emb = OpenAIEmbeddings(model=model)
    vs = PGVectorCls(
        embeddings=emb,
        collection_name=collection,
        connection=dsn,
        use_jsonb=True,
        create_extension=False,
    )

    total = 0
    for path_str, topic in doc_map.items():
        path = Path(path_str)
        if not path.exists():
            print(f"SKIP: {path} not found")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        metadatas = [
            {"source": str(path), "topic": topic, "section": f"{topic}-{idx}"}
            for idx, _ in enumerate(chunks)
        ]
        vs.add_texts(chunks, metadatas=metadatas)
        total += len(chunks)
        print(f"Loaded {len(chunks)} chunks from {path.name} (topic={topic})")

    print(f"Done. Total chunks loaded: {total} into collection '{collection}'.")


if __name__ == "__main__":
    main()
