"""Registro global de modelos PIBot.

Carga perezosa con caché (singleton) para el IntentRouter y el SeriesInterpreter:
- Se resuelven rutas por argumento o variables de entorno.
- Se guardan en memoria (lru_cache) para evitar recargas en cada rerun de Streamlit.
- Un lock protege la inicialización en entornos multihilo.

Uso:
    from registry import get_intent_router, get_series_interpreter, warmup_models
    router = get_intent_router()
    interpreter = get_series_interpreter()
    warmup_models()  # opcional, precarga ambos

Env vars soportadas:
    INTENT_ROUTER_PATH     (default: "pibot-intent-router")
    SERIES_INTERPRETER_PATH (default: "pibot-jointbert")
"""

from __future__ import annotations

import os
import threading
import logging
from functools import lru_cache
from typing import Optional, Any

logger = logging.getLogger(__name__)

try:
    from models.pibot_intent_router import IntentRouter  # type: ignore
except ModuleNotFoundError as exc:
    IntentRouter = None  # type: ignore
    if exc.name == "models":
        logger.debug("IntentRouter import skipped (optional legacy dependency missing): %s", exc)
    else:
        logger.warning("IntentRouter import skipped: %s", exc)
except Exception as exc:
    IntentRouter = None  # type: ignore
    logger.warning("IntentRouter import skipped: %s", exc)

try:
    from models.pibot_series_interpreter import SeriesInterpreter  # type: ignore
except ModuleNotFoundError as exc:
    SeriesInterpreter = None  # type: ignore
    if exc.name == "models":
        logger.debug("SeriesInterpreter import skipped (optional legacy dependency missing): %s", exc)
    else:
        logger.warning("SeriesInterpreter import skipped: %s", exc)
except Exception as exc:
    SeriesInterpreter = None  # type: ignore
    logger.warning("SeriesInterpreter import skipped: %s", exc)

_init_lock = threading.Lock()


def _resolve_path(env_name: str, override: Optional[str], default: str) -> str:
    if override:
        return override
    env_val = os.getenv(env_name)
    if env_val and env_val.strip():
        return env_val.strip()
    return default


def _load_intent_router(path: str) -> Optional[Any]:
    with _init_lock:
        if IntentRouter is None:
            return None
        try:
            return IntentRouter(model_path=path)
        except Exception as e:
            logger.warning(
                "IntentRouter failed to load from '%s': %s. Falling back to heuristic-only router.",
                path,
                e,
                exc_info=True,
            )
            return IntentRouter(model_path=None)


def _load_series_interpreter(path: str) -> Optional[Any]:
    with _init_lock:
        if SeriesInterpreter is None:
            return None
        try:
            return SeriesInterpreter(model_path=path)
        except Exception as e:
            logger.warning(
                "SeriesInterpreter failed to load from '%s': %s. Falling back to heuristic-only interpreter.",
                path,
                e,
                exc_info=True,
            )
            return SeriesInterpreter(model_path=None)


@lru_cache(maxsize=8)
def get_intent_router(path: Optional[str] = None) -> Optional[Any]:
    """Devuelve un IntentRouter cacheado (clave = ruta resuelta)."""
    resolved = _resolve_path("INTENT_ROUTER_PATH", path, "pibot-intent-router")
    return _load_intent_router(resolved)


@lru_cache(maxsize=8)
def get_series_interpreter(path: Optional[str] = None) -> Optional[Any]:
    """Devuelve un SeriesInterpreter cacheado (clave = ruta resuelta)."""
    resolved = _resolve_path("SERIES_INTERPRETER_PATH", path, "pibot-jointbert")
    return _load_series_interpreter(resolved)


def warmup_models(
    *,
    intent_router_path: Optional[str] = None,
    series_interpreter_path: Optional[str] = None,
) -> None:
    """Precarga los singletons de modelos una vez; llamar múltiples veces es seguro."""
    get_intent_router(intent_router_path)
    get_series_interpreter(series_interpreter_path)


__all__ = [
    "get_intent_router",
    "get_series_interpreter",
    "warmup_models",
]

