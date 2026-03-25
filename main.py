"""
main.py
-------
Punto de entrada del sistema.

- Carga configuración (config.get_settings)
- Inicializa el orquestador LangChain (orchestrator)
- Llama a la app de Streamlit (app.run_app)

Ejecutar con:
    streamlit run main.py
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit, urlunsplit

import streamlit as st
from dotenv import load_dotenv

from config import (
    INGEST_ON_START,
    LANGGRAPH_CHECKPOINT_NS,
    PREDICT_HEALTHCHECK_ON_START,
    PREDICT_HEALTH_TIMEOUT_SECONDS,
    PREDICT_URL,
    STREAM_CHUNK_LOGS,
    get_settings,
)
import app
from orchestrator.utils.http_client import get_json


# ────────────────────────────────────────────────────────────
# Logging helpers
# ────────────────────────────────────────────────────────────

_NOISY_LOGGERS = (
    "httpcore", "httpx", "openai", "urllib3",
    "huggingface_hub", "transformers", "hf_transfer",
    "datasets", "sentence_transformers",
)


class _SessionFilter(logging.Filter):
    """Garantiza que cada record tenga ``session_id``."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "session_id"):
            record.session_id = "main"
        return True


def _setup_logging() -> logging.LoggerAdapter:
    """Configura handlers, filtros y retorna un LoggerAdapter listo."""
    logs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    fixed_log = os.getenv("RUN_MAIN_LOG", "").strip()
    if fixed_log:
        log_path = os.path.join(logs_dir, fixed_log)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(logs_dir, f"run_main_{ts}.log")

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | session=%(session_id)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    session_filter = _SessionFilter()
    for h in logging.getLogger().handlers:
        h.addFilter(session_filter)

    return logging.LoggerAdapter(
        logging.getLogger(__name__), extra={"session_id": "main"}
    )


# ────────────────────────────────────────────────────────────
# Health-check del modelo remoto
# ────────────────────────────────────────────────────────────

def _health_url_from_predict_url(predict_url: str) -> str:
    raw = str(predict_url or "").strip() or "http://localhost:8000/predict"
    split = urlsplit(raw)
    path = split.path or ""
    if path.endswith("/predict"):
        path = path[: -len("/predict")] + "/health"
    else:
        path = "/health"
    return urlunsplit((split.scheme, split.netloc, path, "", ""))


def _log_remote_model_health(logger: logging.LoggerAdapter) -> None:
    if not PREDICT_HEALTHCHECK_ON_START:
        return
    health_url = _health_url_from_predict_url(PREDICT_URL)
    try:
        payload = get_json(health_url, timeout=PREDICT_HEALTH_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.warning("[MODEL_HEALTH] GET %s failed: %s", health_url, exc)
        return
    if not isinstance(payload, dict):
        logger.warning("[MODEL_HEALTH] Unexpected payload type from %s: %s", health_url, type(payload).__name__)
        return
    logger.info(
        "[MODEL_HEALTH] status=%s model_loaded=%s router_loaded=%s device=%s source=%s",
        payload.get("status"), payload.get("model_loaded"),
        payload.get("router_loaded"), payload.get("device"),
        payload.get("model_source"),
    )
    logger.info(
        "[MODEL_HEALTH] model repo=%s revision=%s commit=%s",
        payload.get("model_hf_repo_id"), payload.get("model_hf_revision"),
        payload.get("model_hf_commit"),
    )
    logger.info(
        "[MODEL_HEALTH] router repo=%s revision=%s commit=%s",
        payload.get("router_hf_repo_id"), payload.get("router_hf_revision"),
        payload.get("router_hf_commit"),
    )


# ────────────────────────────────────────────────────────────
# Ingest de series
# ────────────────────────────────────────────────────────────

def _run_ingest_if_enabled(logger: logging.LoggerAdapter) -> None:
    if not INGEST_ON_START:
        return
    try:
        from orchestrator.catalog.ingest_series import run_ingest

        base = os.path.dirname(__file__)
        catalog_path = os.path.join(base, "orchestrator", "catalog", "catalog.json")
        output_dir = os.path.join(base, "orchestrator", "memory", "data_store")
        n = run_ingest(catalog_path=catalog_path, output_dir=output_dir)
        logger.info("Ingest completado | cuadros_procesados=%s", n)
    except Exception as e:
        logger.warning("Ingest de series falló: %s", e)


# ────────────────────────────────────────────────────────────
# Stream / Invoke — funciones de orquestación
# ────────────────────────────────────────────────────────────

def _extract_field(value: Any, field: str):
    """Recorre recursivamente *value* buscando claves == *field*."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == field:
                yield nested
            else:
                yield from _extract_field(nested, field)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _extract_field(item, field)


def _iter_strings(payload: Any):
    if payload is None:
        return
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            yield from _iter_strings(item)


def _split_event(raw_event: Any):
    mode = None
    payload = raw_event
    if isinstance(raw_event, tuple):
        if len(raw_event) == 2 and isinstance(raw_event[0], str):
            mode, payload = raw_event
        elif len(raw_event) == 3 and isinstance(raw_event[1], str):
            mode, payload = raw_event[1], raw_event[2]
    return mode, payload


def make_stream_fn(
    graph: Any,
    logger: logging.LoggerAdapter,
):
    """Devuelve un generador que hace streaming del grafo LangGraph."""

    def stream_fn(
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        stream_chunks: Optional[bool] = None,
    ) -> Iterable[str]:
        if graph is None:
            logger.error("graph.stream no disponible: LangGraph no se inicializó correctamente")
            yield "No pude generar una respuesta (graph no inicializado)."
            return

        thread_id = session_id or f"graph-session-{uuid.uuid4().hex}"
        logger.info("[STREAM] session=%s question=%s", thread_id, question[:80])

        state: Dict[str, Any] = {
            "question": question,
            "history": history or [],
            "context": {"session_id": thread_id},
        }
        graph_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": LANGGRAPH_CHECKPOINT_NS,
            },
        }

        got_any = False
        final_output_text = ""
        seen_stream_chunk_count = 0

        try:
            for event in graph.stream(
                state, config=graph_config, stream_mode=["updates", "custom"],
            ):
                mode, payload = _split_event(event)

                for chunk_payload in _extract_field(payload, "stream_chunks"):
                    payload_items = chunk_payload
                    if isinstance(chunk_payload, (list, tuple)):
                        start_idx = max(0, min(seen_stream_chunk_count, len(chunk_payload)))
                        payload_items = chunk_payload[start_idx:]
                        seen_stream_chunk_count = len(chunk_payload)
                    for chunk_text in _iter_strings(payload_items):
                        if not chunk_text:
                            continue
                        final_output_text += chunk_text
                        if STREAM_CHUNK_LOGS:
                            logger.debug("[STREAM_OUT] chunk=%s", chunk_text[:120])
                        got_any = True
                        yield chunk_text

                if mode in (None, "updates", "values"):
                    for out_payload in _extract_field(payload, "output"):
                        for out_text in _iter_strings(out_payload):
                            if isinstance(out_text, str):
                                final_output_text = out_text

            if not got_any:
                if final_output_text:
                    yield final_output_text
                else:
                    logger.warning("[STREAM_EMPTY] no chunks produced for session=%s", session_id)
        except Exception as e:
            logger.exception("stream_fn graph stream failed: %s", e)
            yield "No pude generar una respuesta."

    return stream_fn


def make_invoke_fn(
    stream_fn,
    logger: logging.LoggerAdapter,
):
    """Devuelve una función síncrona que consume ``stream_fn``."""

    def invoke_fn(
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        logger.info("[INVOKE] question=%s", question[:80])
        return "".join(stream_fn(question, history=history, session_id=None))

    return invoke_fn


# ────────────────────────────────────────────────────────────
# Punto de entrada
# ────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    # Fast-path para reruns de Streamlit (ya inicializado)
    if "app_initialized" in st.session_state:
        settings = st.session_state.get("settings")
        sfn = st.session_state.get("stream_fn")
        ifn = st.session_state.get("invoke_fn")
        if all([settings, sfn, ifn]):
            app.run_app(settings=settings, stream_fn=sfn, invoke_fn=ifn)
        return

    logger = _setup_logging()
    logger.info("Inicializando aplicación Streamlit")

    _log_remote_model_health(logger)
    settings = get_settings()
    _run_ingest_if_enabled(logger)

    # Construir grafo LangGraph
    try:
        from orchestrator.graph.agent_graph import build_graph
        graph = build_graph()
        logger.info("Agente basado en LangGraph inicializado")
    except Exception as e:
        logger.error("No se pudo inicializar LangGraph: %s", e)
        raise

    stream_fn = make_stream_fn(graph, logger)
    invoke_fn = make_invoke_fn(stream_fn, logger)

    # Persistir en session_state para reruns
    st.session_state.settings = settings
    st.session_state.stream_fn = stream_fn
    st.session_state.invoke_fn = invoke_fn
    st.session_state.app_initialized = True

    logger.info("Lanzando UI Streamlit")
    app.run_app(settings=settings, stream_fn=stream_fn, invoke_fn=invoke_fn)


if __name__ == "__main__":
    main()
