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
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

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


def _run_trace_silent(question: str) -> tuple[str, str]:
    sink_out = io.StringIO()
    sink_err = io.StringIO()
    with redirect_stdout(sink_out), redirect_stderr(sink_err):
        os.environ["USE_JOINTBERT_CLASSIFIER"] = "false"

        from orchestrator.graph.agent_graph import build_graph  # type: ignore

        formatter: Callable[[Any], str] | None = None
        try:
            from qa.qa import _format_final_response as formatter  # type: ignore
        except Exception:
            try:
                from qa import _format_final_response as formatter  # type: ignore
            except Exception:
                formatter = None

        graph = build_graph()
        state: dict[str, Any] = {
            "question": question,
            "history": [],
            "context": {"session_id": f"qa-batch-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}"},
        }
        cfg = {
            "configurable": {
                "thread_id": state["context"]["session_id"],
                "checkpoint_ns": "memory",
            },
        }

        current_state: dict[str, Any] = dict(state)
        for event in graph.stream(state, config=cfg, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for _, delta in event.items():
                if isinstance(delta, dict):
                    current_state.update(delta)

        final_output = current_state.get("output")
        response = formatter(final_output) if callable(formatter) else str(final_output or "")
        metadata_key = str(current_state.get("metadata_key") or "").strip()
        return response, metadata_key


def _render_blocks(questions: list[str], metadata_keys: set[str]) -> str:
    blocks: list[str] = []
    for idx, question in enumerate(questions, start=1):
        response, metadata_key = _run_trace_silent(question)
        response = response.strip()
        key_display = metadata_key or "N/A"
        if metadata_key and metadata_key in metadata_keys:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = _resolve_input_path(args.input)
    output_path = _resolve_output_path(args.output)

    questions = _load_questions(input_path)
    if not questions:
        raise ValueError(f"El archivo {input_path} no contiene preguntas válidas")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró metadata_q.json en: {METADATA_PATH}")
    metadata_keys = _load_metadata_keys(METADATA_PATH)

    rendered = _render_blocks(questions, metadata_keys)
    output_path.write_text(rendered, encoding="utf-8")

    print(f"Preguntas procesadas: {len(questions)}")
    print(f"Entrada: {input_path}")
    print(f"Salida: {output_path}")


if __name__ == "__main__":
    main()
