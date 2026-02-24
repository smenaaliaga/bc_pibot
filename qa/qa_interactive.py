#!/usr/bin/env python3
"""Ejecución interactiva del trace QA.

Uso:
  ../.venv/bin/python qa_interactive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa.qa import run_trace  # noqa: E402


def main() -> None:
    print("QA interactivo (escribe 'salir' para terminar).")
    while True:
        try:
            question = input("\nPregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if not question:
            print("Ingresa una pregunta o escribe 'salir'.")
            continue
        if question.lower() in {"salir", "exit", "quit"}:
            print("Saliendo...")
            break

        run_trace(question)


if __name__ == "__main__":
    main()
