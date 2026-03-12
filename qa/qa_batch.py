#!/usr/bin/env python3
"""Ejecución batch QA desde archivo de preguntas.

Lee preguntas línea a línea y genera un archivo consolidado con
Pregunta/Respuesta usando la misma lógica de qa_interactive.py.

Uso:
  ../.venv/bin/python qa_batch.py
  ../.venv/bin/python qa_batch.py --input questions.txt --output response.txt
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT_CANDIDATES = ("questions.txt", "question.txt")
DEFAULT_OUTPUT = "response.txt"
METADATA_PATH = ROOT / "orchestrator" / "catalog" / "metadata_q.json"
BLOCK_SEPARATOR = "################################################################################################################################"


def _resolve_input_path(raw_input: str | None) -> Path:
    qa_dir = Path(__file__).resolve().parent

    if raw_input:
        candidate = Path(raw_input)
        if not candidate.is_absolute():
            candidate = qa_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"No existe archivo de preguntas: {candidate}")
        return candidate

    for name in DEFAULT_INPUT_CANDIDATES:
        candidate = qa_dir / name
        if candidate.exists():
            return candidate

    tried = ", ".join(str(qa_dir / n) for n in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(f"No se encontró archivo de preguntas. Busqué en: {tried}")


def _resolve_output_path(raw_output: str | None) -> Path:
    qa_dir = Path(__file__).resolve().parent
    out_name = (raw_output or DEFAULT_OUTPUT).strip() or DEFAULT_OUTPUT
    output_path = Path(out_name)
    if not output_path.is_absolute():
        output_path = qa_dir / output_path
    return output_path


def _load_questions(input_path: Path) -> list[str]:
    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    questions: list[str] = []
    for line in raw_lines:
        question = line.strip()
        if not question:
            continue
        if question.startswith("#"):
            continue
        questions.append(question)
    return questions


def _load_metadata_keys(metadata_path: Path) -> set[str]:
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        return set()

    keys: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if key:
            keys.add(key)
    return keys


def _split_event(raw_event: Any) -> tuple[str | None, Any]:
    mode = None
    payload = raw_event
    if isinstance(raw_event, tuple):
        if len(raw_event) == 2 and isinstance(raw_event[0], str):
            mode, payload = raw_event
        elif len(raw_event) == 3 and isinstance(raw_event[1], str):
            mode, payload = raw_event[1], raw_event[2]
    return mode, payload


def _extract_field(value: Any, field: str):
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


def _strip_streamlit_markers(text: str) -> str:
    collecting_csv = False
    collecting_chart = False
    collecting_followup = False
    out_lines: list[str] = []

    for line in text.splitlines(keepends=True):
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
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _normalize_streamlit_output(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\\*", "*")
    return _strip_streamlit_markers(normalized)


def _run_trace_silent(
    *,
    graph: Any,
    question: str,
    session_id: str,
    checkpoint_ns: str,
) -> tuple[str, str]:
    sink_out = io.StringIO()
    sink_err = io.StringIO()
    with redirect_stdout(sink_out), redirect_stderr(sink_err):
        state: dict[str, Any] = {
            "question": question,
            "history": [],
            "context": {"session_id": session_id},
        }
        cfg = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": checkpoint_ns,
            },
        }

        current_state: dict[str, Any] = dict(state)
        final_output_text = ""
        seen_stream_chunk_count = 0

        for event in graph.stream(state, config=cfg, stream_mode=["updates", "custom"]):
            mode, payload = _split_event(event)

            if mode in (None, "updates", "values") and isinstance(payload, dict):
                for _, delta in payload.items():
                    if isinstance(delta, dict):
                        current_state.update(delta)

            for chunk_payload in _extract_field(payload, "stream_chunks"):
                payload_items = chunk_payload
                if isinstance(chunk_payload, (list, tuple)):
                    start_idx = seen_stream_chunk_count
                    if start_idx < 0 or start_idx > len(chunk_payload):
                        start_idx = len(chunk_payload)
                    payload_items = chunk_payload[start_idx:]
                    seen_stream_chunk_count = len(chunk_payload)

                for chunk_text in _iter_strings(payload_items):
                    if chunk_text:
                        final_output_text += chunk_text

        final_output = str(current_state.get("output") or "")
        response_raw = final_output_text or final_output
        response = _normalize_streamlit_output(response_raw)
        metadata_key = str(current_state.get("metadata_key") or "").strip()
        return response, metadata_key


def _render_blocks(
    *,
    graph: Any,
    questions: list[str],
    metadata_keys: set[str],
    metadata_enabled: bool,
    checkpoint_ns: str,
    isolate_session_per_question: bool,
) -> str:
    blocks: list[str] = []
    batch_session_id = f"qa-batch-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}"

    for idx, question in enumerate(questions, start=1):
        session_id = (
            f"{batch_session_id}-{idx}"
            if isolate_session_per_question
            else batch_session_id
        )
        response, metadata_key = _run_trace_silent(
            graph=graph,
            question=question,
            session_id=session_id,
            checkpoint_ns=checkpoint_ns,
        )
        response = response.strip()
        key_display = metadata_key or "N/A"
        if not metadata_enabled:
            key_status = "metadata_q.json no disponible (flujo usa catalogo/estado del grafo)"
        elif metadata_key and metadata_key in metadata_keys:
            key_status = "llave encontrada en metadata_q.json"
        else:
            key_status = "llave no encontrada en metadata_q.json"

        block = (
            f"Pregunta {idx}: {question}\n"
            f"Respuesta {idx}:\n"
            f"{response}\n\n"
            f"Key {idx}: {key_display}\n"
            f"Estado {idx}: {key_status}\n\n"
            f"{BLOCK_SEPARATOR}\n"
        )
        blocks.append(block)
    return "\n".join(blocks).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta preguntas QA en batch y guarda response.txt")
    parser.add_argument("--input", "-i", default=None, help="Archivo de preguntas (default: questions.txt o question.txt)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Archivo de salida (default: response.txt)")
    parser.add_argument(
        "--isolate-session-per-question",
        action="store_true",
        help=(
            "Usa session_id diferente por pregunta (legacy). "
            "Por defecto se reutiliza la misma sesion para homologar Streamlit."
        ),
    )
    parser.add_argument(
        "--checkpoint-ns",
        default=os.getenv("LANGGRAPH_CHECKPOINT_NS", "memory"),
        help="Namespace de checkpoints LangGraph (default: LANGGRAPH_CHECKPOINT_NS o 'memory').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = _resolve_input_path(args.input)
    output_path = _resolve_output_path(args.output)

    questions = _load_questions(input_path)
    if not questions:
        raise ValueError(f"El archivo {input_path} no contiene preguntas válidas")

    # Paridad con Streamlit: misma bandera de clasificacion remota/local.
    os.environ["USE_JOINTBERT_CLASSIFIER"] = os.getenv("USE_JOINTBERT_CLASSIFIER_STREAMLIT", "false")

    from orchestrator.graph.agent_graph import build_graph  # type: ignore

    graph = build_graph()

    metadata_enabled = METADATA_PATH.exists()
    metadata_keys = _load_metadata_keys(METADATA_PATH) if metadata_enabled else set()

    rendered = _render_blocks(
        graph=graph,
        questions=questions,
        metadata_keys=metadata_keys,
        metadata_enabled=metadata_enabled,
        checkpoint_ns=str(args.checkpoint_ns or "memory"),
        isolate_session_per_question=bool(args.isolate_session_per_question),
    )
    output_path.write_text(rendered, encoding="utf-8")

    print(f"Preguntas procesadas: {len(questions)}")
    print(f"Entrada: {input_path}")
    print(f"Salida: {output_path}")
    print(f"Session mode: {'isolated' if args.isolate_session_per_question else 'shared (streamlit-like)'}")
    print(f"Checkpoint namespace: {args.checkpoint_ns}")
    print(f"metadata_q.json: {'disponible' if metadata_enabled else 'no disponible'}")


if __name__ == "__main__":
    main()
