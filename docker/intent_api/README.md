# Dummy Intent API

Servicio mínimo para responder `macro`, `intent` y `context` en `POST /intent`.

## Build + Run
Desde `docker/`:
```bash
cd docker
set REDIS_PASS=owMaOQsIAGK80NYwlsvv93a4ZOxX73Um
docker compose up -d intent-api
```

## Example
```bash
curl -s -X POST http://localhost:8000/intent -H "Content-Type: application/json" -d '{"text": "cual fue el ultimo imacec"}'
```
