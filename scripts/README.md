# Scripts PIBot API

## Ejecutar API con script

Este script levanta la API local en foreground, sin Docker.

### 1) Dar permisos de ejecución (solo la primera vez)

```bash
cd /Users/hernanfernandez/Documents/09\ -\ workspace/01_pibert/bc_pibot
chmod +x scripts/start_api.sh
```

### 2) Ejecutar

```bash
./scripts/start_api.sh
```

### Qué hace el script

- Entra a `docker/intent_api/` del proyecto actual.
- Si falta `fastapi`/`uvicorn`, instala dependencias desde `requirements.txt`.
- Levanta la API local con `uvicorn` en `http://localhost:8000`.
- Queda ejecutándose en la terminal hasta que presiones `Ctrl+C`.

### Probar endpoint `/predict`

```bash
curl -X POST http://localhost:8000/predict \
	-H "Content-Type: application/json" \
	-d '{"text": "cual fue la ultima cifra del imacec"}'
```

### Detener

- En la misma terminal: `Ctrl+C`

Si necesitas ejecutar en background o guardar logs, avísame y lo agrego.
