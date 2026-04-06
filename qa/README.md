# QA Trace Scripts

Este directorio incluye scripts para ejecutar el grafo y registrar la traza.

## Requisitos

- Activar el entorno virtual de `pibot`.

```bash
source "/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/.venv/bin/activate"
```

## Ejecutar `qa.py` (una pregunta)

```bash
cd "/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/qa"
"/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/.venv/bin/python" qa.py "cual es el valor del imacec"
```

## Ejecutar `qa_interactive.py` (modo interactivo)

```bash
cd "/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/qa"
"/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/.venv/bin/python" qa_interactive.py
```

## Ejecutar `qa_batch.py` (todas las preguntas a `response.txt`)

Lee preguntas desde `questions.txt` (o `question.txt`) y ejecuta cada una por el mismo flujo del chatbot,
guardando solo pares pregunta-respuesta en `response.txt`.

```bash
cd "/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/qa"
"/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/.venv/bin/python" qa_batch.py
```

Opcionalmente puedes definir input/output:

```bash
"/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/.venv/bin/python" qa_batch.py --input questions.txt --output response.txt
```

## Solución permanente conflicto Postgres local vs Docker (puertos separados)

Si `qa.py` falla con `database "pibot" does not exist`, normalmente `Postgres.app` local está tomando `localhost:5432`.
En este repo, Docker queda publicado en `localhost:5433` para evitar choque de puertos.

### 1) Confirmar que Docker usa 5433 (contenedor 5432)

En este proyecto, `docker/docker-compose.yml` ya publica:

- `5433:5432` para `pibot-postgres2`

Levantar servicios:

```bash
cd "/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot"
docker compose -f docker/docker-compose.yml up -d
```

### 2) Deshabilitar Postgres.app permanentemente al iniciar sesión

```bash
uid="$(id -u)"
launchctl bootout "gui/$uid/com.postgresapp.Postgres2LoginHelper" 2>/dev/null || true
launchctl disable "gui/$uid/com.postgresapp.Postgres2LoginHelper" 2>/dev/null || true
pkill -f '/Applications/Postgres.app' || true
```

### 3) Verificar que `localhost:5433` apunta a Docker

```bash
lsof -nP -iTCP:5433 -sTCP:LISTEN
PGPASSWORD=postgres psql -P pager=off -h localhost -p 5433 -U postgres -d pibot -c 'select current_database(), inet_server_addr();'
```

Si devuelve `pibot`, ya no estás conectando al Postgres local.

### Salir del modo interactivo

- Escribe `salir`, `exit` o `quit`.
- También puedes usar `Ctrl+C`.

## Salida de traza

El archivo de traza se escribe en:

- `qa/qa_trace.log`
