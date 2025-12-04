# Docker stack (Postgres + optional RAG)

## Run the database
From `docker/`:
```bash
docker compose build postgres
docker compose up -d postgres
```

## Create extension vector

```bash
docker compose exec postgres psql -U postgres -d pibot -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## RAG loading (manual)
- The image already contains `load_txt_rag.py` and the cleaned docs at `/rag/docs`.
- Requires `OPENAI_API_KEY` in the environment.

Example to load vectors after the container is up:
```bash
  .venv/Scripts/activate
  python docker/postgres/load_txt_rag.py
```
## Recreate the database
From `docker/`:
```bash
docker compose down
docker compose build postgres   # si cambiaste la imagen
docker compose up -d postgres
```

You can also point to a different doc path using `RAG_DOCS_DIR`.

## Files
- `docker/postgres/Dockerfile`: builds Postgres with pgvector.
- `docker/postgres/init.sql`: creates core tables (checkpoints, intents, etc.).
- `docker/postgres/load_txt_rag.py`: loader script (imports from `docs` by default if `RAG_DOCS_DIR` not set).

No automatic RAG loading occurs at container start; trigger it manually as above.
