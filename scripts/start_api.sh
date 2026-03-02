#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INTENT_API_DIR="$PROJECT_ROOT/docker/intent_api"

if [[ ! -d "$INTENT_API_DIR" ]]; then
  echo "No se encontró directorio intent_api en $INTENT_API_DIR" >&2
  exit 1
fi

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No se encontró Python en PATH." >&2
  exit 1
fi

cd "$INTENT_API_DIR"

if ! "$PYTHON_BIN" -c "import fastapi, uvicorn, email_validator" >/dev/null 2>&1; then
  echo "Instalando dependencias locales de API..."
  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m ensurepip --upgrade
  fi
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

APP_MODULE="intent_api.app:app"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Levantando API local en http://localhost:${PORT} (sin Docker)"
exec "$PYTHON_BIN" -m uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" --reload
