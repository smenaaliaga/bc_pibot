"""Carga el .env del proyecto y resuelve el modelo OpenAI productivo.

El automator debe usar EXACTAMENTE el mismo modelo que el chatbot en
producción (OPENAI_MODEL del .env). Sin defaults hardcodeados: si no está
definido, fallamos ruidosamente.
"""
from __future__ import annotations
import os
from pathlib import Path

_DOTENV_FLAG = "_QA_AUTOMATOR_DOTENV_LOADED"


def _load_dotenv_once() -> None:
    if os.getenv(_DOTENV_FLAG) == "1":
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and os.getenv(key) is None:
                os.environ[key] = value
    os.environ[_DOTENV_FLAG] = "1"


def get_openai_model() -> str:
    _load_dotenv_once()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "OPENAI_MODEL no está definido. Asegúrate de que el .env del "
            "proyecto contenga OPENAI_MODEL=... (mismo modelo del chatbot)."
        )
    return model


def get_openai_api_key() -> str | None:
    _load_dotenv_once()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key or None
