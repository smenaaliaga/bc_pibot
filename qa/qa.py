#!/usr/bin/env python3
"""Ejecuta el grafo LangGraph y traza input/output por nodo.

Uso:
  ./qa.py "cual es el valor del imacec"
  ../.venv/bin/python qa.py "cual es el valor del imacec"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LOG_PATH = Path(__file__).resolve().parent / "qa_trace.log"
GRAPH_NAME = "pibot_trace_graph"
API_CALL_NOTE = (
    "Este script usa build_graph (mismo flujo que Streamlit). "
    "La clasificación puede llamar a APIs externas en "
    "orchestrator/classifier/classifier_agent._classify_with_jointbert() "
    "(post_json a PREDICT_URL e INTENT_CLASSIFIER_URL) si USE_JOINTBERT_CLASSIFIER=false."
)

NODE_DESCRIPTIONS = {
    "ingest": "Normaliza pregunta, carga contexto/memoria y construye estado base.",
    "classify": "Clasifica intención/entidades y construye intent_info.",
    "intent": "Decide ruta (data/rag/fallback) según clasificación.",
    "router": "Normaliza entidades y confirma route_decision.",
    "data": "Consulta pipeline de datos/series según la intención.",
    "rag": "Resuelve con RAG/LLM para preguntas de metodología.",
    "fallback": "Respuesta genérica LLM cuando no hay ruta clara.",
    "memory": "Persiste estado final en memoria/checkpoints.",
}


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("qa_trace")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception:
        return str(data)


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


def _format_params_table(values: Dict[str, Any], statuses: Dict[str, str]) -> str:
    rows = []
    rows.append("Key                                  | Status     | Value")
    rows.append("-------------------------------------+------------+--------------------------")
    for key in sorted(values.keys()):
        status = statuses.get(key, "MISSING")
        raw_value = values.get(key)
        value = "(empty)" if raw_value in (None, "", [], {}) else str(raw_value)
        rows.append(f"{key:<37} | {status:<10} | {value}")
    return "\n".join(rows)


def run_trace(question: str) -> None:
    os.environ["USE_JOINTBERT_CLASSIFIER"] = "false"
    from orchestrator.graph.agent_graph import build_graph  # noqa: E402

    logger = _setup_logger()
    graph = build_graph()

    state: Dict[str, Any] = {
        "question": question,
        "history": [],
        "context": {"session_id": f"qa-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"},
    }
    cfg = {
        "configurable": {"thread_id": state["context"]["session_id"], "checkpoint_ns": "memory"},
    }

    print(f"Grafo: {GRAPH_NAME}")
    print(f"Pregunta: {question}")
    print(API_CALL_NOTE)
    logger.info("Grafo: %s", GRAPH_NAME)
    logger.info("Pregunta: %s", question)
    logger.info("%s", API_CALL_NOTE)

    current_state: Dict[str, Any] = dict(state)

    for event in graph.stream(state, config=cfg, stream_mode="updates"):
        if not isinstance(event, dict):
            continue
        for node_name, delta in event.items():
            description = NODE_DESCRIPTIONS.get(node_name, "Nodo del grafo")
            input_snapshot = dict(current_state)

            print(f"\n[NODO] {node_name} :: {description}")
            print("INPUT:", _safe_json(input_snapshot))
            print("OUTPUT:", _safe_json(delta))

            logger.info("NODO=%s | %s", node_name, description)
            logger.info("INPUT=%s", _safe_json(input_snapshot))
            logger.info("OUTPUT=%s", _safe_json(delta))

            if node_name == "intent":
                decision = None
                if isinstance(delta, dict):
                    decision = delta.get("route_decision")
                if decision is None:
                    decision = current_state.get("route_decision")
                logger.info("DECISION: decide la ruta de clasificacion")
                logger.info("ROUTE_DECISION (intent)=%s", decision)
                print("ROUTE_DECISION (intent):", decision)

            if node_name == "router":
                decision = None
                if isinstance(delta, dict):
                    decision = delta.get("route_decision")
                if decision is None:
                    decision = current_state.get("route_decision")
                logger.info("DECISION: confirma ruta del grafo")
                logger.info("ROUTE_DECISION (router)=%s", decision)
                print("ROUTE_DECISION (router):", decision)
                next_step = decision if decision in {"data", "rag", "fallback"} else "fallback"
                logger.info("NEXT_NODE=%s", next_step)
                print("NEXT_NODE:", next_step)

            if node_name == "memory":
                final_output = None
                if isinstance(delta, dict):
                    final_output = delta.get("output")
                if final_output is None:
                    final_output = current_state.get("output")
                logger.info("DECISION: respuesta final almacenada en memory")
                logger.info("RESPUESTA_FINAL=%s", _safe_json(final_output))

            if node_name == "classify":
                classification = None
                if isinstance(delta, dict):
                    classification = delta.get("classification")
                if classification is None:
                    classification = current_state.get("classification")
                payload = _extract_classification_payload(classification)
                logger.info("CLASSIFICATION_FIELDS=%s", _safe_json(payload))

            if node_name == "data":
                parsed_point = None
                parsed_range = None
                series_id = None
                data_cls = None
                data_params = None
                data_params_status = None
                metadata_response = None
                metadata_key = None
                series_fetch_args = None
                series_fetch_result = None
                if isinstance(delta, dict):
                    parsed_point = delta.get("parsed_point")
                    parsed_range = delta.get("parsed_range")
                    series_id = delta.get("series")
                    data_cls = delta.get("data_classification")
                    data_params = delta.get("data_params")
                    data_params_status = delta.get("data_params_status")
                    metadata_response = delta.get("metadata_response")
                    metadata_key = delta.get("metadata_key")
                    series_fetch_args = delta.get("series_fetch_args")
                    series_fetch_result = delta.get("series_fetch_result")
                if parsed_point is None:
                    parsed_point = current_state.get("parsed_point")
                if parsed_range is None:
                    parsed_range = current_state.get("parsed_range")
                if series_id is None:
                    series_id = current_state.get("series")
                if data_cls is None:
                    data_cls = current_state.get("data_classification")
                if data_params is None:
                    data_params = current_state.get("data_params")
                if data_params_status is None:
                    data_params_status = current_state.get("data_params_status")
                if metadata_response is None:
                    metadata_response = current_state.get("metadata_response")
                if metadata_key is None:
                    metadata_key = current_state.get("metadata_key")
                if series_fetch_args is None:
                    series_fetch_args = current_state.get("series_fetch_args")
                if series_fetch_result is None:
                    series_fetch_result = current_state.get("series_fetch_result")

                response_type = None
                serie_default = None
                if isinstance(metadata_response, dict):
                    serie_default = metadata_response.get("serie_default")
                series_value = series_id if series_id is not None else serie_default
                if series_value is None or str(series_value).lower() == "none":
                    response_type = "general_response"
                else:
                    req_form_value = None
                    if isinstance(data_cls, dict):
                        req_form_value = data_cls.get("req_form") or data_cls.get("req_form_cls")
                    if str(req_form_value).lower() == "specific_point":
                        response_type = "specific_point_response"
                    else:
                        response_type = "specific_response"

                if response_type == "specific_point_response":
                    response_type_log = "SPECIFIC_POINT_RESPONSE"
                elif response_type == "specific_response":
                    response_type_log = "SPECIFIC_RESPONSE"
                else:
                    response_type_log = "GENERAL_RESPONSE"
                logger.info("TYPE_RESPONSE: %s", response_type_log)
                print("TYPE_RESPONSE:", response_type_log)
                logger.info(
                    "DATA_PARSED_PARAMS=%s",
                    _safe_json(
                        {
                            "parsed_point": parsed_point,
                            "parsed_range": parsed_range,
                            "series": series_id,
                            "data_classification": data_cls,
                        }
                    ),
                )
                if metadata_response is not None:
                    if metadata_key:
                        logger.info("METADATA_KEY=%s", metadata_key)
                        print("METADATA_KEY:", metadata_key)
                    logger.info("METADATA_RESPONSE=%s", _safe_json(metadata_response))
                    print("METADATA_RESPONSE:", _safe_json(metadata_response))
                if series_fetch_args is not None:
                    logger.info("SERIES_FETCH_ARGS=%s", _safe_json(series_fetch_args))
                    print("SERIES_FETCH_ARGS:", _safe_json(series_fetch_args))
                if series_fetch_result is not None:
                    logger.info("SERIES_FETCH_RESULT=%s", _safe_json(series_fetch_result))
                    print("SERIES_FETCH_RESULT:", _safe_json(series_fetch_result))
                if isinstance(data_params, dict) and isinstance(data_params_status, dict):
                    table = _format_params_table(data_params, data_params_status)
                    logger.info("DATA_PARAMS_TABLE=\n%s", table)
                    print("DATA_PARAMS_TABLE:\n" + table)

            if isinstance(delta, dict):
                current_state.update(delta)

    print(f"\nLog generado en: {LOG_PATH}")
    logger.info("Log generado en: %s", LOG_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traza LangGraph por nodo (QA)")
    parser.add_argument("question", nargs="?", default="cual es el valor del imacec")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_trace(args.question)


if __name__ == "__main__":
    main()
