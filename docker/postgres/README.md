# PGVector RAG Loader

This folder hosts the tooling required to refresh the LangChain/PGVector collection used by PIBot.

## Prerequisites
- `.env` or shell exports for:
  - `PG_DSN` (e.g. `postgresql://postgres:postgres@localhost:5432/pibot`)
  - `RAG_PGVECTOR_COLLECTION` (defaults to `methodology`).
  - `OPENAI_API_KEY` (for embeddings).
  - Optional: `RAG_LANGUAGE`, `RAG_DOCS_DIR`, `RAG_PURGE_MODE`.
- Dependencies installed in the repo virtual environment: `uv pip sync` (or `pip install -r requirements.txt`).
- PostgreSQL instance with the LangChain tables initialized (script can create missing collections).

## Document Manifest
`docs/manifest.json` enumerates every `.txt` file that should be loaded. Each entry declares:
- `id`: Stable document identifier.
- `path`: Relative path under `docker/postgres/docs/`.
- `topic`, `version`, `tags`: Metadata baked into every chunk.

Add a new document by appending an entry to `manifest.json` (paths outside the repo root are allowed but must be absolute).

## Common Commands
All commands assume the repo root (`E:\bc_pibot`).

### Dry-run (no DB writes)
```powershell
uv run python docker/postgres/load_txt_rag.py --dry-run --manifest docker/postgres/docs/manifest.json
```
Validates chunking, logs rejected chunks, and prints per-doc counts without touching PGVector.

### Full drop + reload
```powershell
uv run python docker/postgres/load_txt_rag.py `
  --manifest docker/postgres/docs/manifest.json `
  --purge-mode drop `
  --log-file logs/exports/rag_reload.log `
  --reject-log logs/exports/rag_rejects.jsonl
```
Steps performed:
1. Delete the entire `RAG_PGVECTOR_COLLECTION` (via `purge_collection`).
2. Read every manifest entry, chunk text (default 900 chars w/120 overlap), run validation filters, and insert embeddings.
3. Record detailed progress in `rag_reload.log` and rejected chunks in `rag_rejects.jsonl`.

### Delta reload (only changed sources)
```powershell
uv run python docker/postgres/load_txt_rag.py --purge-mode delta
```
Detects per-document hashes, deletes stale embeddings, and reloads only documents whose source text changed.

### Generate evaluation dataset
```powershell
uv run python docker/postgres/load_txt_rag.py --eval-report logs/exports/rag_eval_latest.json
```
Produces a JSON artifact listing every chunk inserted during the run. This is intended for downstream Ragas quality evaluations.

### Excel loader (question/fact/topic)
Ingest an Excel file with columns for question, fact (answer), and topic using `load_excel_facts.py`.

Append to an existing collection (no drop):
```powershell
uv run python docker/postgres/load_excel_facts.py `
  --excel docker/postgres/docs/doc_base.xlsx `
  --collection methodology `
  --doc-id faq_excel `
  --version v1 `
  --tags "faq,excel" `
  --language es `
  --question-col 0 `
  --fact-col 1 `
  --topic-col 2
```

Drop and reload that collection:
```powershell
uv run python docker/postgres/load_excel_facts.py `
  --excel docker/postgres/docs/doc_base.xlsx `
  --collection methodology `
  --doc-id faq_excel `
  --version v1 `
  --tags "faq,excel" `
  --language es `
  --question-col 0 `
  --fact-col 1 `
  --topic-col 2 `
  --purge
```
Notes:
- `--question-col/--fact-col/--topic-col` accept column names or 0-based indexes.
- `doc_id`, `version`, `tags`, `language`, and any `--extra key=value` are stored in metadata.
- `--dry-run` parses the file without writing to PG.

## Useful Flags
- `--chunk-size / --chunk-overlap`: Override manifest defaults for experimentation.
- `--staging-table staging.rag_chunks`: Copy chunks into a Postgres table before vector insertion.
- `--emit-manifest-summary`: Print the resolved doc list and exit (handy for debugging path issues).
- `--reject-log <path>`: Choose where to store chunk rejections (default: `logs/exports/rag_rejects_<timestamp>.jsonl`).

## Logs & Troubleshooting
- `logs/exports/rag_reload.log`: High-level timing and success/failure per document.
- `logs/exports/rag_rejects*.jsonl`: One JSON line per filtered chunk with `doc_id`, reason, and checksum.
- If you see `ValueError: Collection not found`, ensure you’re not instantiating the vector store before `--purge-mode drop` completes (latest script handles this automatically).
- Use `--dry-run` to diagnose missing files (`SKIP` warnings) without contacting OpenAI.

## Tests
Run unit tests for manifest parsing & chunking:
```powershell
uv run python -m pytest tests/test_load_txt_rag.py
```
Add coverage whenever the loader logic changes.
