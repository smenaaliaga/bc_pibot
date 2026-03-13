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
import json
from urllib.parse import urlsplit, urlunsplit

import streamlit as st
from dotenv import load_dotenv

from config import get_settings, PREDICT_URL
# from orchestrator.classifier.joint_bert_classifier import get_predictor
from registry import warmup_models
import app
from orchestrator.utils.http_client import get_json, post_json
from orchestrator.utils.run_detail_log import append_detail_trace, append_run_detail


_DETAIL_SEPARATOR = "************************************************************************************************************"


def _health_url_from_predict_url(predict_url: str) -> str:
    """Deriva la URL de health desde PREDICT_URL, forzando path '/health'."""
    raw = str(predict_url or "").strip() or "http://localhost:8000/predict"
    split = urlsplit(raw)
    path = split.path or ""
    if path.endswith("/predict"):
        path = path[: -len("/predict")] + "/health"
    else:
        path = "/health"
    return urlunsplit((split.scheme, split.netloc, path, "", ""))


def _log_remote_model_health(logger: logging.LoggerAdapter) -> None:
    """Consulta GET /health del modelo remoto y registra metadata útil."""
    enabled = os.getenv("PREDICT_HEALTHCHECK_ON_START", "1").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return

    health_url = _health_url_from_predict_url(PREDICT_URL)
    timeout = float(os.getenv("PREDICT_HEALTH_TIMEOUT_SECONDS", "5"))
    try:
        payload = get_json(health_url, timeout=timeout)
    except Exception as exc:
        logger.warning("[MODEL_HEALTH] GET %s failed: %s", health_url, exc)
        return

    if not isinstance(payload, dict):
        logger.warning("[MODEL_HEALTH] Unexpected payload type from %s: %s", health_url, type(payload).__name__)
        return

    # Log compacto para trazabilidad en arranque.
    logger.info(
        "[MODEL_HEALTH] status=%s model_loaded=%s router_loaded=%s device=%s source=%s",
        payload.get("status"),
        payload.get("model_loaded"),
        payload.get("router_loaded"),
        payload.get("device"),
        payload.get("model_source"),
    )
    logger.info(
        "[MODEL_HEALTH] model repo=%s revision=%s commit=%s",
        payload.get("model_hf_repo_id"),
        payload.get("model_hf_revision"),
        payload.get("model_hf_commit"),
    )
    logger.info(
        "[MODEL_HEALTH] router repo=%s revision=%s commit=%s",
        payload.get("router_hf_repo_id"),
        payload.get("router_hf_revision"),
        payload.get("router_hf_commit"),
    )


def _warmup_predict_endpoint(logger: logging.LoggerAdapter) -> None:
    """Hace un primer POST /predict para evitar cold-start en la primera consulta del usuario."""
    enabled = os.getenv("PREDICT_WARMUP_ON_START", "1").lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return

    warmup_text = str(os.getenv("PREDICT_WARMUP_TEXT", "cual es el valor del imacec") or "").strip()
    if not warmup_text:
        warmup_text = "cual es el valor del imacec"

    timeout = float(os.getenv("PREDICT_WARMUP_TIMEOUT_SECONDS", "30"))
    t0 = os.times()[4]
    try:
        payload = post_json(PREDICT_URL, {"text": warmup_text}, timeout=timeout)
    except Exception as exc:
        logger.warning("[MODEL_WARMUP] POST %s failed: %s", PREDICT_URL, exc)
        return

    elapsed = max(0.0, os.times()[4] - t0)
    routing = payload.get("routing") if isinstance(payload, dict) else {}
    routing = routing if isinstance(routing, dict) else {}
    intent = routing.get("intent") if isinstance(routing.get("intent"), dict) else {}
    context = routing.get("context") if isinstance(routing.get("context"), dict) else {}
    logger.info(
        "[MODEL_WARMUP] ok elapsed=%.3fs intent=%s context=%s",
        elapsed,
        intent.get("label"),
        context.get("label"),
    )


def _extract_classification_payload(classification: Any) -> Dict[str, Any]:
    if classification is None:
        return {}
    return {
        "intent": getattr(classification, "intent", None),
        "confidence": getattr(classification, "confidence", None),
        "entities": getattr(classification, "entities", None),
        "normalized": getattr(classification, "normalized", None),
        "macro": getattr(classification, "macro", None),
        "context": getattr(classification, "context", None),
        "calc_mode": getattr(classification, "calc_mode", None),
        "activity": getattr(classification, "activity", None),
        "region": getattr(classification, "region", None),
        "investment": getattr(classification, "investment", None),
        "req_form": getattr(classification, "req_form", None),
        "words": getattr(classification, "words", None),
        "slot_tags": getattr(classification, "slot_tags", None),
        "predict_raw": getattr(classification, "predict_raw", None),
    }


def _strip_streamlit_markers(text: str) -> str:
    collecting_csv = False
    collecting_chart = False
    collecting_followup = False
    out_lines: List[str] = []

    for line in str(text or "").splitlines(keepends=True):
        ls = line.strip()
        if ls == "##CSV_DOWNLOAD_START":
            collecting_csv = True
            continue
        if ls == "##CSV_DOWNLOAD_END":
            collecting_csv = False
            continue
        if ls == "##CHART_START":
            collecting_chart = True
            continue
        if ls == "##CHART_END":
            collecting_chart = False
            continue
        if ls == "##FOLLOWUP_START":
            collecting_followup = True
            continue
        if ls == "##FOLLOWUP_END":
            collecting_followup = False
            continue

        if collecting_csv or collecting_chart or collecting_followup:
            continue

        out_lines.append(line)

    cleaned = "".join(out_lines)
    return cleaned.replace("\\*", "*").strip()


def _log_streamlit_response_block(logger: logging.LoggerAdapter, question: str, response: str) -> None:
    """Registra en run_main.log la respuesta final con formato legible tipo Streamlit."""
    compact_question = " ".join(str(question or "").split())
    logger.info("[QA_TRACE] STREAMLIT_RESPONSE question=%s", compact_question[:180])
    logger.info("[QA_TRACE] STREAMLIT_RESPONSE_BEGIN")

    text = str(response or "").strip()
    if not text:
        logger.info("[QA_TRACE] (sin respuesta)")
    else:
        for line in text.splitlines():
            logger.info("[QA_TRACE] %s", line)

    logger.info("[QA_TRACE] STREAMLIT_RESPONSE_END")


def _route_to_response_class(route_decision: Any) -> str:
    decision = str(route_decision or "").strip().lower()
    if decision == "data":
        return "DATA"
    if decision == "rag":
        return "RAG"
    if decision == "fallback":
        return "PAYLOAD"
    return ""


def _resolve_response_classification(state: Dict[str, Any], final_output_raw: str) -> str:
    if isinstance(state, dict):
        route_class = _route_to_response_class(state.get("route_decision"))
        if route_class:
            return route_class

        intent_payload = state.get("intent")
        if isinstance(intent_payload, dict):
            route_class = _route_to_response_class(intent_payload.get("context_mode"))
            if route_class:
                return route_class

        # Señal de respaldo para consultas DATA cuando no quedó route_decision.
        if state.get("series") or state.get("data_classification"):
            return "DATA"

    raw_text = str(final_output_raw or "")
    if "##CSV_DOWNLOAD_START" in raw_text:
        return "DATA"
    if (
        "para mayor información, puedes consultar los documentos disponibles en la web oficial del banco central de chile"
        in raw_text.lower()
    ):
        return "RAG"
    return "PAYLOAD"


def _append_detail_trace_response_block(
    *,
    question: str,
    response_classification: str,
    response_text: str,
    session_id: Optional[str] = None,
) -> None:
    compact_question = " ".join(str(question or "").split())
    append_detail_trace(_DETAIL_SEPARATOR, session_id=session_id)
    append_detail_trace(f"PREGUNTA={compact_question}", session_id=session_id)
    append_detail_trace(
        f"CLASIFICACION_RESPUESTA={str(response_classification or '').strip() or 'PAYLOAD'}",
        session_id=session_id,
    )
    append_detail_trace(f"RESPUESTA_FINAL_TEXT=\n{str(response_text or '').strip()}", session_id=session_id)
    append_detail_trace(_DETAIL_SEPARATOR, session_id=session_id)


def _extract_marker_blocks(text: str, *, start_marker: str, end_marker: str) -> List[str]:
    blocks: List[str] = []
    source = str(text or "")
    cursor = 0

    while True:
        start_idx = source.find(start_marker, cursor)
        if start_idx < 0:
            break
        end_idx = source.find(end_marker, start_idx)
        if end_idx < 0:
            break
        end_idx += len(end_marker)
        block = source[start_idx:end_idx]
        if not block.endswith("\n"):
            block += "\n"
        blocks.append(block)
        cursor = end_idx

    return blocks


def _missing_followup_blocks(streamed_text: str, state_output: str) -> List[str]:
    streamed_blocks = _extract_marker_blocks(
        streamed_text,
        start_marker="##FOLLOWUP_START",
        end_marker="##FOLLOWUP_END",
    )
    state_blocks = _extract_marker_blocks(
        state_output,
        start_marker="##FOLLOWUP_START",
        end_marker="##FOLLOWUP_END",
    )
    if not state_blocks:
        return []

    streamed_signatures = {block.strip() for block in streamed_blocks if block.strip()}
    missing: List[str] = []
    for block in state_blocks:
        signature = block.strip()
        if not signature or signature in streamed_signatures:
            continue
        streamed_signatures.add(signature)
        missing.append(block)

    return missing


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
    # Usar siempre un único archivo de log en logs/.
    fixed_log = os.getenv("RUN_MAIN_LOG", "").strip() or "run_main.log"
    log_path = os.path.join(logs_dir, fixed_log)
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

    # Verificar estado del modelo remoto una vez al arranque inicial.
    _log_remote_model_health(logger)
    _warmup_predict_endpoint(logger)

    settings = get_settings()

    # # Pre-cargar modelo JointBERT (singleton global dentro de get_predictor)
    # logger.info("Inicializando predictor")
    # try:
    #     get_predictor()  # fuerza la inicialización una sola vez
    #     # logger.info("Predictor inicializado")
    # except Exception as e:
    #     logger.warning(f"Predictor no disponible: {e}")

    # Warm-up de modelos (router + interpreter) para evitar recargas en reruns
    try:
        warmup_models()
    except Exception as e:
        logger.warning(f"Warmup de modelos adicionales falló: {e}")

    preload_series_updates = os.getenv("PRELOAD_SERIES_UPDATES_ON_START", "1").lower() in {"1", "true", "yes", "on"}
    try:
        from orchestrator.data.get_data_serie import preload_series_updates_index, start_series_updates_scheduler
    except Exception as e:
        logger.warning(f"No se pudo importar SearchSeries updates: {e}")
    else:
        if preload_series_updates:
            try:
                series_updates = preload_series_updates_index(force=True)
                logger.info("SearchSeries precargado | series_con_updatedAt=%s", len(series_updates))
            except Exception as e:
                logger.warning(f"Precarga de SearchSeries falló: {e}")
        try:
            if start_series_updates_scheduler():
                logger.info(
                    "Scheduler diario SearchSeries activo | hora=%s",
                    os.getenv("SERIES_UPDATES_DAILY_AT", "09:10"),
                )
        except Exception as e:
            logger.warning(f"No se pudo iniciar scheduler diario de SearchSeries: {e}")

    warm_redis = os.getenv("WARM_REDIS_CACHE_ON_START", "0").lower() in {"1", "true", "yes", "on"}
    if warm_redis:
        try:
            from tools.warm_redis_cache import warm_cache_background
            warm_workers = int(os.getenv("WARM_REDIS_CACHE_WORKERS", "8"))
            warm_cache_background(workers=warm_workers, force=False)
        except Exception as e:
            logger.warning(f"No se pudo iniciar precalentamiento Redis: {e}")

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
        logger.info("[QA_TRACE] QUESTION_BEGIN session=%s question=%s", thread_id, " ".join(str(question).split())[:180])
        append_run_detail(
            "stream_start",
            {
                "question": question,
                "history_len": len(history or []),
                "checkpoint_ns": checkpoint_ns,
            },
            session_id=thread_id,
        )
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
        qa_trace_enabled = os.getenv("QA_TRACE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}

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
                    logger.info("[QA_TRACE] RESPUESTA_FINAL_RAW_TYPE=%s", type(final_output).__name__)

                if node_name == "classify":
                    classification = None
                    if isinstance(delta, dict):
                        classification = delta.get("classification")
                    if classification is None:
                        classification = current_state.get("classification")
                    append_run_detail(
                        "classification_detail",
                        {
                            "question": question,
                            "classification": _extract_classification_payload(classification),
                        },
                        session_id=thread_id,
                    )

                if node_name == "data":
                    data_cls = None
                    data_series = None
                    if isinstance(delta, dict):
                        data_cls = delta.get("data_classification")
                        data_series = delta.get("series")
                    if data_cls is None:
                        data_cls = current_state.get("data_classification")
                    if data_series is None:
                        data_series = current_state.get("series")
                    append_run_detail(
                        "data_state",
                        {
                            "question": question,
                            "series": data_series,
                            "data_classification": data_cls,
                        },
                        session_id=thread_id,
                    )

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
        stream_chunk_payload_events = 0
        stream_chunk_list_events = 0
        stream_chunk_reset_events = 0
        stream_piece_count = 0
        stream_piece_chars = 0

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
                    if qa_trace_enabled:
                        _log_qa_trace(payload)

                for chunk_payload in _extract_field(payload, "stream_chunks"):
                    stream_chunk_payload_events += 1
                    payload_items = chunk_payload
                    if isinstance(chunk_payload, (list, tuple)):
                        stream_chunk_list_events += 1
                        start_idx = seen_stream_chunk_count
                        if start_idx < 0:
                            start_idx = 0
                        if start_idx > len(chunk_payload):
                            stream_chunk_reset_events += 1
                            append_detail_trace(
                                (
                                    "STREAM_CHUNKS_RESET "
                                    f"prev_seen={seen_stream_chunk_count} "
                                    f"incoming_len={len(chunk_payload)}"
                                ),
                                session_id=thread_id,
                            )
                            start_idx = len(chunk_payload)
                        payload_items = chunk_payload[start_idx:]
                        seen_stream_chunk_count = len(chunk_payload)
                    for chunk_text in _iter_strings(payload_items):
                        if not chunk_text:
                            continue
                        stream_piece_count += 1
                        stream_piece_chars += len(chunk_text)
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

            state_output_text = str(current_state.get("output") or "")
            if got_any and state_output_text:
                # El output final del nodo memory puede incluir followups que no se emitieron como chunks.
                for followup_block in _missing_followup_blocks(
                    streamed_text=final_output_text,
                    state_output=state_output_text,
                ):
                    final_output_text += followup_block
                    yield followup_block

            final_output_raw = final_output_text or state_output_text
            streamlit_like = _strip_streamlit_markers(final_output_raw)

            append_run_detail(
                "stream_chunks_stats",
                {
                    "question": question,
                    "chunk_payload_events": stream_chunk_payload_events,
                    "chunk_list_events": stream_chunk_list_events,
                    "chunk_reset_events": stream_chunk_reset_events,
                    "pieces_emitted": stream_piece_count,
                    "chars_emitted": stream_piece_chars,
                    "raw_response_len": len(final_output_raw),
                    "clean_response_len": len(streamlit_like),
                    "has_csv_marker": "##CSV_DOWNLOAD_START" in final_output_raw,
                    "has_chart_marker": "##CHART_START" in final_output_raw,
                    "has_followup_marker": "##FOLLOWUP_START" in final_output_raw,
                },
                session_id=thread_id,
            )
            append_detail_trace(
                f"respuesta_final={json.dumps(final_output_raw, ensure_ascii=False)}",
                session_id=thread_id,
            )
            response_classification = _resolve_response_classification(current_state, final_output_raw)
            append_run_detail(
                "response_classification",
                {
                    "question": question,
                    "route_decision": current_state.get("route_decision"),
                    "intent_context_mode": (
                        current_state.get("intent", {}).get("context_mode")
                        if isinstance(current_state.get("intent"), dict)
                        else None
                    ),
                    "classification": response_classification,
                },
                session_id=thread_id,
            )
            _append_detail_trace_response_block(
                question=question,
                response_classification=response_classification,
                response_text=streamlit_like,
                session_id=thread_id,
            )

            _log_streamlit_response_block(logger, question, streamlit_like)
            append_run_detail(
                "streamlit_response",
                {
                    "question": question,
                    "response": streamlit_like,
                    "classification": response_classification,
                },
                session_id=thread_id,
            )
            logger.info("[QA_TRACE] QUESTION_END session=%s", thread_id)

            if not got_any:
                if final_output_text:
                    yield final_output_text
                else:
                    logger.warning("[STREAM_EMPTY] no chunks produced for session=%s", session_id)
        except Exception as e:
            logger.exception("stream_fn graph stream failed: %s", e)
            append_run_detail(
                "stream_error",
                {
                    "question": question,
                    "error": str(e),
                },
                session_id=thread_id,
            )
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
