#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless tester for PIBot orchestrator.
- Reads questions from `test/questions.json`
- Executes `orchestrator.stream_answer` for each question
- Writes per-question response logs into `test/` while normal logs continue in `logs/`
"""
import os
import json
import datetime as _dt
from typing import List, Dict

# Ensure we can import local modules
ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_DIR = os.path.join(ROOT, "test")
QUESTIONS_PATH = os.path.join(TEST_DIR, "questions.json")
TS = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_JSON_BASENAME = f"responses_{TS}.json"
OUTPUT_TEXT_BASENAME = f"log_{TS}.text"
OUTPUT_JSON_PATH = os.path.join(TEST_DIR, OUTPUT_JSON_BASENAME)
OUTPUT_TEXT_PATH = os.path.join(TEST_DIR, OUTPUT_TEXT_BASENAME)

# Import orchestrator streaming interface
try:
    from orchestrator import stream_answer, get_current_test_log_file  # type: ignore
except Exception as e:
    raise RuntimeError(f"No se pudo importar orchestrator: {e}")


def load_questions(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe el archivo de preguntas: {path}")
    data = json.loads(open(path, "r", encoding="utf-8").read())
    qs = data.get("questions") or []
    if not isinstance(qs, list):
        raise ValueError("Formato inválido en questions.json: 'questions' debe ser lista")
    return [str(q) for q in qs]


def run_headless(qs: List[str]) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    history: List[Dict[str, str]] = []  # simple history accumulation
    for idx, q in enumerate(qs, start=1):
        # Append to history as user turn
        history.append({"role": "user", "content": q})
        # Stream answer
        chunks = []
        for ch in stream_answer(q, history=history[:-1]):  # pass history excluding current turn
            chunks.append(ch)
        answer_text = "".join(chunks)
        results.append({
            "index": idx,
            "question": q,
            "answer": answer_text,
        })
        # Also append assistant turn to history for next routing context
        history.append({"role": "assistant", "content": answer_text})
    return results


def main():
    os.makedirs(TEST_DIR, exist_ok=True)
    qs = load_questions(QUESTIONS_PATH)
    results = run_headless(qs)
    payload = {
        "generated_at": _dt.datetime.now().isoformat(),
        "log_file": get_current_test_log_file() if callable(get_current_test_log_file) else None,
        "count": len(results),
        "results": results,
    }
    # Write consolidated JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # Write plain text log with question → answer pairs
    with open(OUTPUT_TEXT_PATH, "w", encoding="utf-8") as f:
        for item in results:
            q = item.get("question", "")
            a = item.get("answer", "")
            f.write("Pregunta: " + q + "\n")
            f.write("Respuesta: " + a + "\n")
            f.write("\n")
    print(f"OK: respuestas guardadas en {OUTPUT_JSON_PATH}")
    print(f"OK: log de preguntas/respuestas en {OUTPUT_TEXT_PATH}")
    if payload.get("log_file"):
        print(f"Log de sesión: {payload['log_file']}")


if __name__ == "__main__":
    main()
