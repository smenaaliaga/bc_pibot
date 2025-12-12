"""Enhanced loader for RAG documents into PGVector with manifest + evaluation hooks."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv  # type: ignore

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langdetect import detect as _langdetect  # type: ignore
from langchain_openai import OpenAIEmbeddings  # type: ignore
from langchain_postgres import PGVector as PGVectorCls  # type: ignore
import psycopg  # type: ignore
from psycopg import errors as psycopg_errors  # type: ignore
from psycopg import sql  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "docs" / "manifest.json"
DEFAULT_LOG_DIR = REPO_ROOT / "logs" / "exports"
ALLOWED_TOPICS = {
    "pib",
    "imacec",
    "seasonality",
    "metodologia",
    "deflactores",
    "encadenamiento",
    "fuentes",
}
LEGACY_DOCS = {
    r"E:\\bc_pibot\\docker\\postgres\\docs\\Cuentas_Nacionales_metodos_fuentes_ref18_all_20251203_200609_limpio.txt": "pib",
    r"E:\\bc_pibot\\docker\\postgres\\docs\\Imacec_all_20251203_193937_limpio.txt": "imacec",
    r"E:\\bc_pibot\\docker\\postgres\\docs\\see100_all_20251203_194549_limpio.txt": "seasonality",
}


@dataclass
class ManifestDefaults:
    """Default values applied to manifest entries."""

    topic: str = "metodologia"
    version: str = "v1"
    tags: List[str] = field(default_factory=list)
    language: str = "es"
    chunk_size: int = 900
    chunk_overlap: int = 120


@dataclass
class DocumentSpec:
    """Represents a single document entry from the manifest."""

    doc_id: str
    path: Path
    topic: str
    version: str
    tags: List[str] = field(default_factory=list)
    language: str = "es"
    extra: Dict[str, Any] = field(default_factory=dict)

    def exists(self) -> bool:
        return self.path.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload cleaned TXT sources into PGVector with provenance metadata.",
    )
    parser.add_argument("--manifest", type=Path, help="Path to JSON manifest", default=None)
    parser.add_argument("--chunk-size", type=int, help="Override chunk size", default=None)
    parser.add_argument("--chunk-overlap", type=int, help="Override chunk overlap", default=None)
    parser.add_argument(
        "--purge-mode",
        choices=["drop", "delta", "skip"],
        default=os.getenv("RAG_PURGE_MODE", "drop"),
        help="drop: delete collection, delta: per-doc diff, skip: append",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip DB writes")
    parser.add_argument("--log-file", type=Path, help="Optional log output path")
    parser.add_argument("--reject-log", type=Path, help="JSONL path for rejected chunks")
    parser.add_argument("--min-chars", type=int, default=120, help="Minimum chars per chunk")
    parser.add_argument(
        "--eval-report",
        type=Path,
        help="If provided, emit chunk dataset for downstream Ragas eval",
    )
    parser.add_argument("--staging-table", help="Optional staging table name", default=None)
    parser.add_argument(
        "--language",
        default=os.getenv("RAG_LANGUAGE", "es"),
        help="Target ISO language for chunks (via langdetect if installed)",
    )
    parser.add_argument(
        "--emit-manifest-summary",
        action="store_true",
        help="Print manifest digest and exit",
    )
    return parser.parse_args()


def setup_logging(log_file: Optional[Path]) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        handlers=handlers,
    )


def _resolve_path(base: Path, target: str) -> Path:
    path = Path(target)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_manifest(path: Path) -> Tuple[ManifestDefaults, List[DocumentSpec]]:
    if not path.exists():
        logging.warning("Manifest %s not found; falling back to legacy mapping", path)
        docs = [
            DocumentSpec(
                doc_id=f"legacy-{idx}",
                path=Path(doc_path),
                topic=topic,
                version="legacy",
            )
            for idx, (doc_path, topic) in enumerate(LEGACY_DOCS.items())
        ]
        return ManifestDefaults(), docs

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - validated in tests
        raise ValueError(f"Manifest {path} is not valid JSON: {exc}") from exc

    defaults = ManifestDefaults(**raw.get("defaults", {}))
    docs: List[DocumentSpec] = []
    manifest_dir = path.parent
    for entry in raw.get("documents", []):
        if "id" not in entry or "path" not in entry:
            raise ValueError("Manifest entries must include 'id' and 'path'")
        doc_path = _resolve_path(manifest_dir, entry["path"])
        extra_keys = {k: v for k, v in entry.items() if k not in {"id", "path", "topic", "version", "tags", "language"}}
        docs.append(
            DocumentSpec(
                doc_id=entry["id"],
                path=doc_path,
                topic=entry.get("topic", defaults.topic),
                version=entry.get("version", defaults.version),
                tags=entry.get("tags", defaults.tags.copy()),
                language=entry.get("language", defaults.language),
                extra=extra_keys,
            )
        )
    return defaults, docs


def load_docs_from_dir(dir_path: Optional[str], defaults: ManifestDefaults) -> List[DocumentSpec]:
    if not dir_path:
        return []
    base = Path(dir_path)
    docs: List[DocumentSpec] = []
    if not base.exists():
        logging.warning("RAG_DOCS_DIR %s not found", base)
        return docs
    for idx, path in enumerate(sorted(base.glob("*.txt"))):
        topic = defaults.topic
        name = path.name.lower()
        if "imacec" in name:
            topic = "imacec"
        elif "see100" in name or "season" in name:
            topic = "seasonality"
        elif "pib" in name or "cuentas" in name:
            topic = "pib"
        docs.append(
            DocumentSpec(
                doc_id=f"dir-{idx}-{path.stem}",
                path=path,
                topic=topic,
                version=defaults.version,
                language=defaults.language,
            )
        )
    return docs


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def build_chunker(chunk_size: int, chunk_overlap: int):
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

        def _split(text: str) -> List[str]:
            return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

        return _split

    step = max(1, chunk_size - max(0, chunk_overlap))

    def _fallback(text: str) -> List[str]:
        chunks: List[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    return _fallback


def generate_chunk_topics(text: str, topic_hint: str) -> List[str]:
    topics = {topic_hint}
    lower = text.lower()
    if "deflactor" in lower or "precio constante" in lower:
        topics.add("deflactores")
    if "encadenad" in lower:
        topics.add("encadenamiento")
    if "metodolog" in lower:
        topics.add("metodologia")
    if "fuente" in lower or "data source" in lower:
        topics.add("fuentes")
    return sorted(t for t in topics if t in ALLOWED_TOPICS)


def detect_language_code(text: str) -> Optional[str]:
    if not _langdetect:
        return None
    try:
        snippet = text[:4000]
        return _langdetect(snippet)
    except Exception:
        return None


def should_accept_chunk(
    text: str,
    chunk_hash: str,
    seen_hashes: set,
    min_chars: int,
    target_language: Optional[str],
) -> Tuple[bool, Optional[str]]:
    if len(text) < min_chars:
        return False, "too_short"
    if chunk_hash in seen_hashes:
        return False, "duplicate"
    detected = detect_language_code(text)
    if target_language and detected:
        detected_simple = detected.lower().split("-")[0]
        if detected_simple != target_language.lower():
            return False, f"lang_{detected_simple}"
    return True, None


def log_rejection(
    writer,
    spec: DocumentSpec,
    reason: str,
    chunk_hash: str,
    text: str,
    chunk_index: int,
) -> None:
    payload = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "doc_id": spec.doc_id,
        "chunk_index": chunk_index,
        "reason": reason,
        "chunk_hash": chunk_hash,
        "char_count": len(text),
    }
    writer.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_psycopg_available() -> None:
    if psycopg is None or sql is None:  # pragma: no cover - sanity
        raise SystemExit("psycopg (with SQL helpers) is required; install psycopg[binary]")


def purge_collection(dsn: str, collection: str) -> None:
    ensure_psycopg_available()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT uuid FROM langchain_pg_collection WHERE name = %s",
                    (collection,),
                )
            except psycopg_errors.UndefinedTable:
                logging.warning(
                    "LangChain tables missing; skipping purge for %s (run 'python -m langchain_postgres create_tables' once)",
                    collection,
                )
                return

            rows = cur.fetchall()
            if not rows:
                return
            cur.execute(
                "DELETE FROM langchain_pg_embedding WHERE collection_id = ANY(%s)",
                ([r[0] for r in rows],),
            )
            cur.execute(
                "DELETE FROM langchain_pg_collection WHERE name = %s",
                (collection,),
            )


def fetch_existing_doc_hashes(dsn: str, collection: str) -> Dict[str, str]:
    ensure_psycopg_available()
    sql_query = """
        SELECT e.metadata->>'doc_id' AS doc_id, e.metadata->>'doc_hash' AS doc_hash
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql_query, (collection,))
            except psycopg_errors.UndefinedTable:
                logging.warning(
                    "LangChain tables missing; cannot diff against existing docs for %s", collection
                )
                return {}
            return {row[0]: row[1] for row in cur.fetchall() if row[0] and row[1]}


def delete_doc_chunks(dsn: str, collection: str, doc_id: str) -> None:
    ensure_psycopg_available()
    sql_delete = """
        DELETE FROM langchain_pg_embedding e
        USING langchain_pg_collection c
        WHERE e.collection_id = c.uuid
          AND c.name = %s
          AND e.metadata->>'doc_id' = %s
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql_delete, (collection, doc_id))
            except psycopg_errors.UndefinedTable:
                logging.warning(
                    "LangChain tables missing; cannot delete chunks for doc %s in %s", doc_id, collection
                )


def stage_chunks(dsn: str, table_name: str, doc_id: str, records: Sequence[Dict[str, Any]]) -> None:
    ensure_psycopg_available()
    if not records:
        return
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        doc_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                ).format(table_name=sql.Identifier(table_name))
            )
            cur.execute(
                sql.SQL("DELETE FROM {table_name} WHERE doc_id = %s").format(
                    table_name=sql.Identifier(table_name)
                ),
                (doc_id,),
            )
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {table_name} (doc_id, chunk_id, content, metadata) VALUES (%s, %s, %s, %s)"
                ).format(table_name=sql.Identifier(table_name)),
                [
                    (
                        doc_id,
                        rec["metadata"]["chunk_id"],
                        rec["text"],
                        json.dumps(rec["metadata"], ensure_ascii=False),
                    )
                    for rec in records
                ],
            )
        conn.commit()


def emit_eval_dataset(path: Path, records: Sequence[Dict[str, Any]], collection: str) -> None:
    if not records:
        logging.warning("No records to emit for evaluation dataset")
        return
    payload = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "collection": collection,
        "document_ids": sorted({rec["metadata"]["doc_id"] for rec in records}),
        "chunk_count": len(records),
        "records": [
            {
                "doc_id": rec["metadata"]["doc_id"],
                "chunk_id": rec["metadata"]["chunk_id"],
                "text": rec["text"],
                "metadata": rec["metadata"],
            }
            for rec in records
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Eval dataset written to %s (%s chunks)", path, len(records))


def ensure_embeddings_available() -> None:
    if OpenAIEmbeddings is None:  # pragma: no cover - runtime guard
        raise SystemExit("langchain_openai is required to build embeddings")
    if PGVectorCls is None:
        raise SystemExit("langchain_postgres is required for the PGVector store")


def build_vector_store(dsn: str, collection: str, model: str):
    ensure_embeddings_available()
    embeddings = OpenAIEmbeddings(model=model)
    return PGVectorCls(
        embeddings=embeddings,
        collection_name=collection,
        connection=dsn,
        use_jsonb=True,
        create_extension=False,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    dsn = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot")
    collection = os.getenv("RAG_PGVECTOR_COLLECTION", "methodology")
    model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")

    manifest_path = args.manifest or DEFAULT_MANIFEST
    defaults, manifest_docs = load_manifest(manifest_path)
    docs = manifest_docs or load_docs_from_dir(os.getenv("RAG_DOCS_DIR"), defaults)
    if not docs:
        logging.error("No documents found via manifest or RAG_DOCS_DIR; aborting")
        return

    if args.emit_manifest_summary:
        for spec in docs:
            logging.info("%s -> %s (topic=%s version=%s)", spec.doc_id, spec.path, spec.topic, spec.version)
        return

    chunk_size = args.chunk_size or defaults.chunk_size
    chunk_overlap = args.chunk_overlap or defaults.chunk_overlap
    chunker = build_chunker(chunk_size, chunk_overlap)
    reject_log_path = args.reject_log or (DEFAULT_LOG_DIR / f"rag_rejects_{dt.datetime.utcnow():%Y%m%d_%H%M%S}.jsonl")
    reject_log_path.parent.mkdir(parents=True, exist_ok=True)
    reject_writer = reject_log_path.open("a", encoding="utf-8")

    eval_records: List[Dict[str, Any]] = []
    seen_hashes: set = set()
    total_chunks = 0
    total_rejected = 0

    vector_store = None

    if args.purge_mode == "drop":
        if args.dry_run:
            logging.info("[DRY-RUN] Would purge collection %s", collection)
        else:
            logging.info("Purging collection %s", collection)
            purge_collection(dsn, collection)
    existing_hashes: Dict[str, str] = {}
    if args.purge_mode == "delta" and not args.dry_run:
        existing_hashes = fetch_existing_doc_hashes(dsn, collection)

    if not args.dry_run:
        vector_store = build_vector_store(dsn, collection, model)

    try:
        for spec in docs:
            if not spec.exists():
                logging.warning("Skipping %s; file missing at %s", spec.doc_id, spec.path)
                continue
            text = spec.path.read_text(encoding="utf-8", errors="ignore")
            doc_hash = compute_sha256(text)

            if args.purge_mode == "delta" and not args.dry_run:
                if existing_hashes.get(spec.doc_id) == doc_hash:
                    logging.info("Skipping %s; hash unchanged", spec.doc_id)
                    continue
                delete_doc_chunks(dsn, collection, spec.doc_id)

            chunk_payloads: List[Dict[str, Any]] = []
            for idx, chunk_text in enumerate(chunker(text)):
                chunk_hash = compute_sha256(f"{spec.doc_id}-{idx}-{chunk_text}")
                accept, reason = should_accept_chunk(
                    chunk_text,
                    chunk_hash,
                    seen_hashes,
                    args.min_chars,
                    args.language or spec.language,
                )
                if not accept:
                    log_rejection(reject_writer, spec, reason or "unknown", chunk_hash, chunk_text, idx)
                    total_rejected += 1
                    continue
                seen_hashes.add(chunk_hash)
                chunk_id = f"{spec.doc_id}-{len(chunk_payloads):04d}"
                metadata = {
                    "doc_id": spec.doc_id,
                    "doc_version": spec.version,
                    "doc_hash": doc_hash,
                    "chunk_id": chunk_id,
                    "topic": spec.topic,
                    "chunk_topics": generate_chunk_topics(chunk_text, spec.topic),
                    "tags": spec.tags,
                    "language": spec.language,
                    "source": str(spec.path),
                    "char_count": len(chunk_text),
                    "checksum": chunk_hash,
                    **spec.extra,
                }
                chunk_payloads.append({"text": chunk_text, "metadata": metadata})
                if args.eval_report:
                    eval_records.append({"text": chunk_text, "metadata": metadata})

            if not chunk_payloads:
                logging.warning("No valid chunks for %s", spec.doc_id)
                continue
            if args.dry_run:
                logging.info("[DRY-RUN] Would ingest %s chunks for %s", len(chunk_payloads), spec.doc_id)
                continue

            if args.staging_table:
                stage_chunks(dsn, args.staging_table, spec.doc_id, chunk_payloads)

            vector_store.add_texts(
                [payload["text"] for payload in chunk_payloads],
                metadatas=[payload["metadata"] for payload in chunk_payloads],
            )
            total_chunks += len(chunk_payloads)
            logging.info("Loaded %s chunks from %s", len(chunk_payloads), spec.doc_id)
    finally:
        reject_writer.close()

    logging.info(
        "Completed ingestion: %s accepted chunks, %s rejected chunks, mode=%s dry_run=%s",
        total_chunks,
        total_rejected,
        args.purge_mode,
        args.dry_run,
    )

    if args.eval_report:
        emit_eval_dataset(args.eval_report, eval_records, collection)


if __name__ == "__main__":  # pragma: no cover
    main()
