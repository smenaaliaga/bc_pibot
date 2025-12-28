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

from typing import List, Dict, Optional
import uuid

from config import get_settings
try:
    from orchestrator.graph.agent_graph import build_graph  # type: ignore
except Exception:
    build_graph = None  # type: ignore
from orchestrator.classifier.joint_bert_classifier import get_predictor
import app
import logging
import os
import datetime


def main() -> None:
    """Punto de entrada principal."""
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
    # Reducir ruido de streaming HTTP (httpcore/httpx/openai) que imprime GeneratorExit al cerrar streams
    for noisy in ["httpcore", "httpx", "openai"]:
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

    # Pre-cargar modelo JointBERT (singleton global dentro de get_predictor)
    logger.info("Inicializando predictor")
    try:
        get_predictor()  # fuerza la inicialización una sola vez
        # logger.info("Predictor inicializado")
    except Exception as e:
        logger.warning(f"Predictor no disponible: {e}")

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
