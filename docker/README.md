# Docker stack (Postgres + optional RAG)

## Run Postgres
From `docker/`:
```bash
# build solo si cambiaste la imagen
cd docker
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



## RAG loading (manual)
- The image already contains `load_txt_rag.py` and the cleaned docs at `/rag/docs`.
- Requires `OPENAI_API_KEY` in the environment.

Example to load vectors after the container is up:
```bash
  uv run python docker/postgres/load_txt_rag.py
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
