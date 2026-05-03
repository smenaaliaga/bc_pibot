# Scripts PIBot API

## Ejecutar API con script

Este script levanta la API y hace una prueba `curl`.

### 1) Dar permisos de ejecución (solo la primera vez)

```bash
chmod +x "/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/scripts/start_api.sh"
```

### 2) Ejecutar

```bash
"/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/scripts/start_api.sh"
```

### Qué hace el script

- Entra a `docker/` del proyecto actual.
- Levanta `intent-api` con `docker compose`.
- Ejecuta pruebas `curl` a `http://localhost:8100/health` y `http://localhost:8100/intent`.

### Detener

El servicio queda corriendo en Docker. Para detenerlo:

- `cd /Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/docker`
- `docker compose stop intent-api`

Si necesitas ejecutar en background o guardar logs, avísame y lo agrego.
