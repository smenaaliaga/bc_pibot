# Dummy Intent API

Servicio mínimo para responder `macro`, `intent` y `context` en `POST /intent`.

> Nota: en `docker-compose.yml` este servicio expone `8100:8000`, por lo que desde host se consume en `http://localhost:8100/intent`.

## Build + Run
Desde `docker/`:
```bash
cd docker
set REDIS_PASS=owMaOQsIAGK80NYwlsvv93a4ZOxX73Um
docker compose up -d intent-api
```

## Example
```bash
curl -s -X POST http://localhost:8100/intent -H "Content-Type: application/json" -d '{"text": "cual fue el ultimo imacec"}'
```

Si quieres usar este servicio desde la app principal, define en `.env`:

```bash
INTENT_API_BASE_URL=http://localhost:8100
```
