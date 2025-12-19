"""Load question/fact/topic rows from Excel into a PGVector collection."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv  # type: ignore

load_dotenv()

import pandas as pd  # type: ignore

try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
    from langchain_postgres import PGVector  # type: ignore
    import psycopg  # type: ignore
    from psycopg import errors as psycopg_errors  # type: ignore
except Exception as exc:
    raise SystemExit(f"Missing dependency: {exc}") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load an Excel file with question/fact/topic columns into PGVector",
    )
    p.add_argument("--excel", type=Path, required=True, help="Path to the Excel (.xlsx) file")
    p.add_argument("--sheet", default=None, help="Sheet name/index (defaults to first sheet)")
    p.add_argument("--collection", default=os.getenv("RAG_PGVECTOR_COLLECTION", "faq"), help="PGVector collection name")
    p.add_argument("--doc-id", default=None, help="Document identifier stored in metadata (defaults to Excel filename)")
    p.add_argument("--version", default="v1", help="Document version stored in metadata")
    p.add_argument("--tags", default="", help="Comma-separated tags stored in metadata")
    p.add_argument(
        "--language",
        default=os.getenv("RAG_LANGUAGE", "es"),
        help="Language code stored in metadata (defaults to RAG_LANGUAGE or 'es')",
    )
    p.add_argument(
        "--extra",
        action="append",
        help="Extra metadata key=value pairs; can be passed multiple times",
        default=[],
    )
    p.add_argument("--question-col", default="question", help="Column containing the user question")
    p.add_argument("--fact-col", default="fact", help="Column containing the answer/fact")
    p.add_argument("--topic-col", default="topic", help="Optional topic column (fallback to --default-topic)")
    p.add_argument("--default-topic", default="general", help="Topic used when the topic column is missing/blank")
    p.add_argument("--purge", action="store_true", help="Drop the target collection before loading")
    p.add_argument("--dry-run", action="store_true", help="Parse file and log stats without writing to PG")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


def _parse_tags(raw: str) -> List[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_extra(raw_items: List[str]) -> Dict[str, str]:
    extra: Dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            logging.warning("Ignoring extra metadata entry without '=': %s", item)
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        extra[k] = v
    return extra


def purge_collection(dsn: str, collection: str) -> None:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT uuid FROM langchain_pg_collection WHERE name = %s", (collection,))
            except psycopg_errors.UndefinedTable:
                logging.warning("LangChain tables not found; nothing to purge for %s", collection)
                return
            ids = [r[0] for r in cur.fetchall() or []]
            if not ids:
                return
            cur.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = ANY(%s)", (ids,))
            cur.execute("DELETE FROM langchain_pg_collection WHERE name = %s", (collection,))
        conn.commit()
    logging.info("Purged collection %s", collection)


def load_rows(
    path: Path,
    sheet: str | None,
    question_col: str,
    fact_col: str,
    topic_col: str,
    default_topic: str,
) -> List[Dict[str, str]]:
    sheet_to_use = 0 if sheet is None else sheet  # pandas returns a dict when sheet_name=None
    df = pd.read_excel(path, sheet_name=sheet_to_use)

    def _get_cell(row, col_spec: str) -> str:
        # Allow numeric index (0-based) or column name
        try:
            idx = int(col_spec)
            return row.iloc[idx]
        except (ValueError, IndexError):
            pass
        if col_spec in row:
            return row[col_spec]
        raise KeyError(col_spec)

    # Validate required columns exist either by name or index
    for required in (question_col, fact_col):
        try:
            _ = _get_cell(df.iloc[0], required)
        except Exception:
            raise ValueError(f"Column '{required}' not found in {path.name}")

    rows: List[Dict[str, str]] = []
    for idx, row in df.iterrows():
        try:
            q_raw = _get_cell(row, question_col)
        except Exception:
            q_raw = ""
        try:
            f_raw = _get_cell(row, fact_col)
        except Exception:
            f_raw = ""
        try:
            t_raw = _get_cell(row, topic_col)
        except Exception:
            t_raw = default_topic
        q = str(q_raw or "").strip()
        f = str(f_raw or "").strip()
        topic_val = str(t_raw or default_topic).strip()
        if not q and not f:
            continue
        rows.append(
            {
                "question": q,
                "fact": f,
                "topic": topic_val or default_topic,
                "row": int(idx) + 1,
            }
        )
    return rows


def build_vector_store(dsn: str, collection: str, model: str):
    embeddings = OpenAIEmbeddings(model=model)
    return PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=dsn,
        use_jsonb=True,
        create_extension=False,
    )


def _format_text(rec: Dict[str, str]) -> str:
    return f"Pregunta: {rec['question']}\nHecho: {rec['fact']}"


def main() -> None:
    args = parse_args()
    setup_logging()

    dsn = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot")
    model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
    doc_id = args.doc_id or args.excel.stem
    tags = _parse_tags(args.tags)
    extra = _parse_extra(args.extra)
    rows = load_rows(
        args.excel, args.sheet, args.question_col, args.fact_col, args.topic_col, args.default_topic
    )
    logging.info("Loaded %s rows from %s", len(rows), args.excel)

    if args.dry_run:
        return

    if args.purge:
        purge_collection(dsn, args.collection)

    vector_store = build_vector_store(dsn, args.collection, model)
    texts = []
    metas = []
    for rec in rows:
        text = _format_text(rec)
        if not text.strip():
            continue
        texts.append(text)
        metas.append(
            {
                "doc_id": doc_id,
                "doc_version": args.version,
                "doc_path": str(args.excel),
                "tags": tags,
                "language": args.language,
                "question": rec["question"],
                "fact": rec["fact"],
                "topic": rec["topic"],
                "row": rec["row"],
                **extra,
            }
        )

    if not texts:
        logging.warning("No non-empty rows to ingest.")
        return

    vector_store.add_texts(texts, metadatas=metas)
    logging.info("Inserted %s records into collection %s", len(texts), args.collection)


if __name__ == "__main__":  # pragma: no cover
    main()
