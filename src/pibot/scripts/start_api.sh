#!/usr/bin/env bash
set -euo pipefail

PIBOT_API_DIR="/Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot_api"

cd "$PIBOT_API_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  echo "No se encontró .venv en $PIBOT_API_DIR" >&2
  exit 1
fi

uvicorn app.main:app --reload &
UVICORN_PID=$!

cleanup() {
  if ps -p "$UVICORN_PID" >/dev/null 2>&1; then
    kill "$UVICORN_PID"
  fi
}
trap cleanup EXIT

# Espera breve para que el servidor levante
sleep 5

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "cual es el valor del imacec"}'

# Mantener el servidor activo si el usuario lo desea
wait "$UVICORN_PID"
