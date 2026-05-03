# Docker stack (Postgres + optional RAG)

## Run Postgres
From `docker/`:
```bash
# build solo si cambiaste la imagen
cd docker
set REDIS_PASS=owMaOQsIAGK80NYwlsvv93a4ZOxX73Um
docker compose build postgres
docker compose --env-file ../.env up -d postgres
```
### Create vector extension
From `docker/`:
```bash
docker compose exec postgres psql -U postgres -d pibot -c "CREATE EXTENSION IF NOT EXISTS vector;"
```
## Run Redis (persistence + auth)
From `docker/`:
```bash
docker compose --env-file ../.env up -d redis
```
Requiere que `REDIS_PASS` esté disponible (el `docker-compose.yml` ya toma `env_file: ../.env`), y persiste datos en `../data/redis`.

## Run Intent API (dummy)
From `docker/`:
```bash
docker compose up -d intent-api
```
Endpoint disponible en `http://localhost:8100/intent`.



## RAG loading (manual)
- The image already contains `load_txt_rag.py` and the cleaned docs at `/rag/docs`.
- Requires `OPENAI_API_KEY` in the environment.

Example to load vectors after the container is up:
```bash
  uv run python docker/postgres/load_txt_rag.py
```


### Excel loader (question/fact/topic)
Ingest an Excel file with columns for question, fact (answer), and topic using `load_excel_facts.py`.

```
uv run python docker/postgres/load_excel_facts.py --excel docker/postgres/docs/doc_base.xlsx --collection methodology --doc-id faq_excel --version v1 --tags "faq,excel" --language es --question-col 0 --fact-col 1 --topic-col 2
```

## Recreate the database
From `docker/`:
```bash
cd docker
docker compose down
docker compose build postgres   # opcional: si cambiaste la imagen
docker compose --env-file ../.env up -d postgres redis
```

You can also point to a different doc path using `RAG_DOCS_DIR`.

## Files
- `docker/postgres/Dockerfile`: builds Postgres with pgvector.
- `docker/postgres/init.sql`: creates core tables (checkpoints, intents, etc.).
- `docker/postgres/load_txt_rag.py`: loader script (imports from `docs` by default if `RAG_DOCS_DIR` not set).

No automatic RAG loading occurs at container start; trigger it manually as above.
