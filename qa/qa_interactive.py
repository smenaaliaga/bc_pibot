#!/usr/bin/env python3
"""Ejecución interactiva del trace QA.

Uso:
  ../.venv/bin/python qa_interactive.py
"""
from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _clear_console() -> None:
    os.system("clear")


def _run_trace_silent(question: str) -> str:
    sink_out = io.StringIO()
    sink_err = io.StringIO()
    with redirect_stdout(sink_out), redirect_stderr(sink_err):
        try:
            from qa.qa import run_trace  # type: ignore
        except Exception:
            from qa import run_trace  # type: ignore
        return run_trace(
            question,
            log_mode="a",
            echo_logs=False,
            print_node_io=False,
        )


def main() -> None:
    _clear_console()
    while True:
        try:
            question = input("\nPregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit"}:
            break

        _clear_console()
        print(f"\nPregunta: {question}\n")
        formatted_response = _run_trace_silent(question)
        print("\nRESPUESTA:\n")
        print(formatted_response)


if __name__ == "__main__":
    main()
