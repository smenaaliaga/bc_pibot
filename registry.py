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

try:
    from models.pibot_intent_router import IntentRouter  # type: ignore
except Exception as exc:
    IntentRouter = None  # type: ignore
    logger = logging.getLogger(__name__)
    logger.warning("IntentRouter import skipped: %s", exc)

try:
    from models.pibot_series_interpreter import SeriesInterpreter  # type: ignore
except Exception as exc:
    SeriesInterpreter = None  # type: ignore
    logger = logging.getLogger(__name__)
    logger.warning("SeriesInterpreter import skipped: %s", exc)
from catalog.catalog_service import CatalogService
from api.bde_client import BDEClient

logger = logging.getLogger(__name__)

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
            logger.warning("IntentRouter unavailable; returning None")
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
            # Fallback: return instance without a model (heuristics only)
            return IntentRouter(model_path=None)


def _load_series_interpreter(path: str) -> Optional[Any]:
    with _init_lock:
        if SeriesInterpreter is None:
            logger.warning("SeriesInterpreter unavailable; returning None")
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
            # Fallback: return instance without a model (heuristics only)
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


@lru_cache(maxsize=1)
def get_catalog_service(catalog_path: str = "catalog/series_catalog.json") -> CatalogService:
    """Devuelve una CatalogService cacheada (singleton por ruta)."""
    try:
        service = CatalogService(catalog_path)
        logger.info(f"CatalogService inicializado desde '{catalog_path}'")
        return service
    except Exception as e:
        logger.error(f"Error cargando CatalogService desde '{catalog_path}': {e}", exc_info=True)
        raise


@lru_cache(maxsize=1)
def get_bde_client() -> BDEClient:
    """Devuelve un BDEClient cacheado (singleton)."""
    try:
        client = BDEClient()
        logger.info("BDEClient inicializado")
        return client
    except Exception as e:
        logger.error(f"Error inicializando BDEClient: {e}", exc_info=True)
        raise


def preload_catalog_in_bde(catalog_path: str = "catalog/series_catalog.json") -> None:
    """Precarga el catálogo en el almacén local del BDEClient."""
    try:
        logger.info("Precargando catálogo en almacén local...")
        bde = get_bde_client()
        bde.preload_from_catalog(catalog_path)
        logger.info("Catálogo precargado exitosamente")
    except Exception as exc:
        logger.warning(f"Catalog preload skipped due to error: {exc}", exc_info=True)


def warmup_models(
    *,
    intent_router_path: Optional[str] = None,
    series_interpreter_path: Optional[str] = None,
    catalog_path: str = "catalog/series_catalog.json",
    preload_catalog: bool = False,
) -> None:
    """Precarga todos los singletons una vez; llamar múltiples veces es seguro.
    
    Args:
        intent_router_path: Ruta opcional al modelo IntentRouter
        series_interpreter_path: Ruta opcional al modelo SeriesInterpreter
        catalog_path: Ruta al archivo del catálogo
        preload_catalog: Si True, precarga el catálogo en el almacén del BDEClient
    """
    router = get_intent_router(intent_router_path)
    interpreter = get_series_interpreter(series_interpreter_path)
    catalog = get_catalog_service(catalog_path)
    bde = get_bde_client()
    try:
        logger.info(
            "Singletons cargados: router=%s | interpreter=%s | catalog=%s | bde=%s",
            getattr(router, "model_path", "<desconocido>") if router else "<none>",
            getattr(interpreter, "model_path", "<desconocido>") if interpreter else "<none>",
            catalog_path,
            "BDEClient",
        )
    except Exception:
        # Evitar que un fallo de logging bloquee el warmup
        pass
    
    if preload_catalog:
        preload_catalog_in_bde(catalog_path)


__all__ = [
    "get_intent_router",
    "get_series_interpreter",
    "get_catalog_service",
    "get_bde_client",
    "warmup_models",
    "preload_catalog_in_bde",
]
