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

from typing import List, Dict, Optional, Any
import uuid
import os
import logging
import datetime
import json

import streamlit as st
from dotenv import load_dotenv

from config import get_settings
# from orchestrator.classifier.joint_bert_classifier import get_predictor
from registry import warmup_models
import app


def main() -> None:
    """Punto de entrada principal."""
    # Cargar variables de entorno desde .env
    load_dotenv()

    # Mantener paridad con qa/qa.py (flujo validado): usar clasificación remota
    # salvo que se fuerce explícitamente desde entorno para Streamlit.
    os.environ["USE_JOINTBERT_CLASSIFIER"] = os.getenv("USE_JOINTBERT_CLASSIFIER_STREAMLIT", "false")

    try:
        from orchestrator.graph.agent_graph import build_graph  # type: ignore
    except Exception:
        build_graph = None  # type: ignore
    
    # Usar st.session_state para rastrear si ya se inicializó
    if "app_initialized" in st.session_state:
        # Ya se inicializó, solo ejecutar la app
        settings = st.session_state.get("settings")
        stream_fn = st.session_state.get("stream_fn")
        invoke_fn = st.session_state.get("invoke_fn")
        if all([settings, stream_fn, invoke_fn]):
            app.run_app(
                settings=settings,
                stream_fn=stream_fn,
                invoke_fn=invoke_fn,
            )
        return
    
    # Configurar logger local por ejecución
    logs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Permite forzar un único archivo de log entre reruns (útil en Streamlit)
    fixed_log = os.getenv("RUN_MAIN_LOG", "").strip()
    if fixed_log:
        log_path = os.path.join(logs_dir, fixed_log)
    else:
        log_path = os.path.join(logs_dir, f"run_main_{ts}.log")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | session=%(session_id)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),  # no truncar log en reruns/preguntas
            logging.StreamHandler(),
        ],
        force=True,  # asegura un único set de handlers por ejecución/rerun
    )
    # Reducir ruido de clientes HTTP/HF (urllib3, huggingface_hub, transformers, etc.)
    for noisy in [
        "httpcore",
        "httpx",
        "openai",
        "urllib3",
        "huggingface_hub",
        "transformers",
        "hf_transfer",
        "datasets",
        "sentence_transformers",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    class SessionFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "session_id"):
                record.session_id = "main"
            return True

    for h in logging.getLogger().handlers:
        h.addFilter(SessionFilter())

    base_logger = logging.getLogger(__name__)
    logger = logging.LoggerAdapter(base_logger, extra={"session_id": "main"})
    logger.info("Inicializando aplicación Streamlit")

    settings = get_settings()

    # # Pre-cargar modelo JointBERT (singleton global dentro de get_predictor)
    # logger.info("Inicializando predictor")
    # try:
    #     get_predictor()  # fuerza la inicialización una sola vez
    #     # logger.info("Predictor inicializado")
    # except Exception as e:
    #     logger.warning(f"Predictor no disponible: {e}")

    # Warm-up de modelos (router + interpreter) para evitar recargas en reruns
    # Leer PRELOAD_CATALOG de variables de entorno (default: False)
    preload_catalog = os.getenv("PRELOAD_CATALOG", "0").lower() in {"1", "true", "yes", "on"}
    try:
        warmup_models(preload_catalog=preload_catalog)
    except Exception as e:
        logger.warning(f"Warmup de modelos adicionales falló: {e}")

    orch = None
    graph = None
    if build_graph:
        try:
            graph = build_graph()
            logger.info("Agente basado en LangGraph inicializado")
        except Exception as e:
            logger.error("No se pudo inicializar LangGraph: %s", e)
            raise
    else:
        logger.error(
            "No se pudo importar orchestrator.graph.agent_graph.build_graph; revisa dependencias y errores previos."
        )

    def stream_fn(
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        stream_chunks: Optional[bool] = None,
    ):
        if graph is None:
            logger.error("graph.stream no disponible: LangGraph no se inicializó correctamente")
            yield "No pude generar una respuesta (graph no inicializado)."
            return
        thread_id = session_id or f"graph-session-{uuid.uuid4().hex}"
        checkpoint_ns = os.getenv("LANGGRAPH_CHECKPOINT_NS", "memory")
        logger.info(f"[STREAM] session={thread_id} question={question[:80]}")
        state = {
            "question": question,
            "history": history or [],
            "context": {"session_id": thread_id},
        }
        graph_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            },
        }
        current_state: Dict[str, Any] = dict(state)

        def _safe_json(data: Any) -> str:
            try:
                return json.dumps(data, ensure_ascii=False, default=str)
            except Exception:
                return str(data)

        def _log_qa_trace(update_payload: Dict[str, Any]) -> None:
            if not isinstance(update_payload, dict):
                return
            for node_name, delta in update_payload.items():
                input_snapshot = dict(current_state)
                logger.info("[QA_TRACE] NODO=%s", node_name)
                logger.info("[QA_TRACE] INPUT=%s", _safe_json(input_snapshot))
                logger.info("[QA_TRACE] OUTPUT=%s", _safe_json(delta))

                if node_name in {"intent", "router"}:
                    decision = None
                    if isinstance(delta, dict):
                        decision = delta.get("route_decision")
                    if decision is None:
                        decision = current_state.get("route_decision")
                    logger.info("[QA_TRACE] ROUTE_DECISION (%s)=%s", node_name, decision)

                if node_name == "memory":
                    final_output = None
                    if isinstance(delta, dict):
                        final_output = delta.get("output")
                    if final_output is None:
                        final_output = current_state.get("output")
                    logger.info("[QA_TRACE] RESPUESTA_FINAL=%s", _safe_json(final_output))

                if isinstance(delta, dict):
                    current_state.update(delta)

        # Usar graph.stream para propagar actualizaciones incrementales
        got_any = False

        def _extract_field(value, field):
            if isinstance(value, dict):
                for key, nested in value.items():
                    if key == field:
                        yield nested
                    else:
                        yield from _extract_field(nested, field)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from _extract_field(item, field)

        def _iter_strings(payload):
            if payload is None:
                return
            if isinstance(payload, str):
                yield payload
            elif isinstance(payload, (list, tuple)):
                for item in payload:
                    yield from _iter_strings(item)

        final_output_text = ""
        seen_stream_chunk_count = 0

        def _split_event(raw_event):
            mode = None
            payload = raw_event
            if isinstance(raw_event, tuple):
                if len(raw_event) == 2 and isinstance(raw_event[0], str):
                    mode, payload = raw_event
                elif len(raw_event) == 3 and isinstance(raw_event[1], str):
                    mode, payload = raw_event[1], raw_event[2]
            return mode, payload

        try:
            for event in graph.stream(
                state,
                config=graph_config,
                stream_mode=["updates", "custom"],
            ):
                mode, payload = _split_event(event)

                if mode in (None, "updates", "values") and isinstance(payload, dict):
                    _log_qa_trace(payload)

                for chunk_payload in _extract_field(payload, "stream_chunks"):
                    payload_items = chunk_payload
                    if isinstance(chunk_payload, (list, tuple)):
                        start_idx = seen_stream_chunk_count
                        if start_idx < 0 or start_idx > len(chunk_payload):
                            start_idx = len(chunk_payload)
                        payload_items = chunk_payload[start_idx:]
                        seen_stream_chunk_count = len(chunk_payload)
                    for chunk_text in _iter_strings(payload_items):
                        if not chunk_text:
                            continue
                        final_output_text += chunk_text
                        try:
                            # Gate per-chunk logs with env flag
                            if os.getenv("STREAM_CHUNK_LOGS", "0").lower() in {"1", "true", "yes", "on"}:
                                logger.debug("[STREAM_OUT] chunk=%s", chunk_text[:120])
                        except Exception:
                            pass
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

    def invoke_fn(question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        logger.info(f"[INVOKE] question={question[:80]}")
        return "".join(stream_fn(question, history=history, session_id=None))

    # Guardar en session_state para evitar reinicialización en reruns
    st.session_state.settings = settings
    st.session_state.stream_fn = stream_fn
    st.session_state.invoke_fn = invoke_fn
    st.session_state.app_initialized = True
    
    # Delegar la construcción de la UI a app.py
    logger.info("Lanzando UI Streamlit")
    app.run_app(
        settings=settings,
        stream_fn=stream_fn,
        invoke_fn=invoke_fn,
    )


if __name__ == "__main__":
    # En ejecución normal (python main.py) simplemente llamamos main().
    # Para Streamlit, el comando recomendado sigue siendo:
    #   streamlit run main.py
    main()
