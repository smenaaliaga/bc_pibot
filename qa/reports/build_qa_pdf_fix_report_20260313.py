from __future__ import annotations

import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["USE_JOINTBERT_CLASSIFIER"] = os.getenv("USE_JOINTBERT_CLASSIFIER_STREAMLIT", "false")

from orchestrator.graph.agent_graph import build_graph
from qa.qa_batch import _extract_field, _iter_strings, _split_event
from main import _missing_followup_blocks

ROOT = Path(__file__).resolve().parents[2]
QA_OUTPUT = ROOT / "qa" / "reports" / "response_qa_pdf_20260313.txt"
REPORT_OUT = ROOT / "qa" / "reports" / "qa_pdf_fix_validation_20260313.md"
TARGET_SERIES = "F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T"


def parse_blocks(path: Path) -> dict[int, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("################################################################################################################################")
    rows: dict[int, dict[str, str]] = {}
    for part in parts:
        m = re.search(r"Pregunta\s+(\d+):\s*(.+?)\nRespuesta\s+\1:\n(.+?)\n\nKey\s+\1:", part, re.S)
        if not m:
            continue
        idx = int(m.group(1))
        rows[idx] = {
            "question": m.group(2).strip(),
            "response": m.group(3).strip(),
        }
    return rows


def run_trace(graph: Any, question: str, session_id: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "question": question,
        "history": [],
        "context": {"session_id": session_id},
    }
    cfg = {
        "configurable": {
            "thread_id": session_id,
            "checkpoint_ns": os.getenv("LANGGRAPH_CHECKPOINT_NS", "memory"),
        }
    }

    current_state = dict(state)
    final_output_text = ""
    seen_stream_chunk_count = 0

    t0 = time.perf_counter()
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
    elapsed = time.perf_counter() - t0

    state_output = str(current_state.get("output") or "")
    bridged_raw = final_output_text or state_output
    if bridged_raw and state_output:
        for followup_block in _missing_followup_blocks(bridged_raw, state_output):
            bridged_raw += followup_block

    classification = current_state.get("classification")
    data_cls = current_state.get("data_classification") or {}

    return {
        "elapsed_s": elapsed,
        "route": current_state.get("route_decision"),
        "intent": getattr(classification, "intent", None) if classification is not None else None,
        "context": getattr(classification, "context", None) if classification is not None else None,
        "series": current_state.get("series"),
        "price": data_cls.get("price"),
        "raw": bridged_raw,
        "output": state_output,
    }


def main() -> None:
    rows = parse_blocks(QA_OUTPUT)
    graph = build_graph()

    # Check 1: primera consulta IMACEC no debe caer al mensaje fallback genérico.
    r1 = rows.get(1, {}).get("response", "").lower()
    check_imacec_first = (
        "no tengo informacion actualizada" not in r1
        and "imacec" in r1
        and "fuente" in r1
    )

    # Check 2: respuesta PIB en stream debe mantener marker de followups al puentear estado final.
    sid_followup = f"qa-pdf-followup-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}"
    trace_followup = run_trace(graph, "cual es el valor del pib", sid_followup)
    check_followup_marker = "##FOLLOWUP_START" in str(trace_followup.get("raw") or "")

    # Check 3: consistencia Cuanto/Cuánto con sesion compartida.
    shared_sid = f"qa-pdf-shared-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}"
    seq_questions = [
        "cual es el valor del imacec",
        "cual es el valor del pib",
        "Cuanto es el PIB de Chile?",
        "¿Cuánto es el PIB de Chile?",
        "¿Cua\u0301nto es el PIB de Chile?",
    ]
    seq_results = [run_trace(graph, q, shared_sid) for q in seq_questions]
    q3_q5 = seq_results[2:5]
    check_cuanto_consistency = all(
        r.get("route") == "data"
        and str(r.get("series") or "") == TARGET_SERIES
        and str(r.get("price") or "") == "co"
        for r in q3_q5
    )

    checks = [
        ("Primera consulta IMACEC evita fallback generico", check_imacec_first),
        ("Marker FOLLOWUP presente en respuesta PIB (bridge stream/state)", check_followup_marker),
        ("Paridad Cuanto/Cuánto/combining-accent en sesion compartida", check_cuanto_consistency),
    ]

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)

    lines: list[str] = []
    lines.append("# QA PDF Fix Validation Report (2026-03-13)")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now(UTC).isoformat()}")
    lines.append(f"- Source QA file: `{QA_OUTPUT.relative_to(ROOT)}`")
    lines.append(f"- Target series for Cuanto/Cuánto parity: `{TARGET_SERIES}`")
    lines.append("")
    lines.append("## Overall Status")
    lines.append("")
    lines.append(f"- Checks passed: **{passed}/{total}**")
    lines.append(f"- Status: **{'VALID' if passed == total else 'PARTIAL'}**")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for name, ok in checks:
        lines.append(f"- [{'OK' if ok else 'FAIL'}] {name}")
    lines.append("")
    lines.append("## Shared Session Diagnostics")
    lines.append("")
    for q, result in zip(seq_questions, seq_results):
        lines.append(
            "- "
            f"Q: {q} | route={result.get('route')} | intent={result.get('intent')} | "
            f"context={result.get('context')} | series={result.get('series')} | "
            f"price={result.get('price')} | elapsed_s={float(result.get('elapsed_s') or 0.0):.3f}"
        )
    lines.append("")
    lines.append("## Followup Marker Diagnostics")
    lines.append("")
    lines.append(
        "- "
        f"PIB query route={trace_followup.get('route')} | has_followup_marker={check_followup_marker} | "
        f"elapsed_s={float(trace_followup.get('elapsed_s') or 0.0):.3f}"
    )

    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_OUT)
    print(f"passed={passed} total={total}")


if __name__ == "__main__":
    main()
