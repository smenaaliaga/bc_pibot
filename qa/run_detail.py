#!/usr/bin/env python3
"""Log detallado del flujo DATA → LLM → Respuesta.

Módulo reutilizable: se integra con el flujo de Streamlit en main.py.
También permite ejecución standalone para pruebas.

Salida: logs/run_detail.log (acumula historial de cada consulta)

Standalone:
  uv run python qa/run_detail.py "Cual es el valor del imacec del ultimo mes"
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stderr
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_PATH = ROOT / "logs" / "run_detail.log"
SEPARATOR = "—" * 90


# ---------------------------------------------------------------------------
# Utilidades de formateo
# ---------------------------------------------------------------------------

def _pretty_json(data: Any, indent: int = 2) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=indent, default=str)
    except Exception:
        return str(data)


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def _extract_classification_info(classification: Any) -> Dict[str, Any]:
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
        "predict_raw": getattr(classification, "predict_raw", None),
    }


def _as_yes_no(value: bool) -> str:
    return "SI" if value else "NO"


def _format_final_response(output: Any) -> str:
    if output is None:
        return ""
    text = str(output)
    text = text.replace("\\*", "*")
    text = re.sub(r"##CSV_DOWNLOAD_START.*?##CSV_DOWNLOAD_END", "", text, flags=re.DOTALL)
    followup_match = re.search(r"##FOLLOWUP_START(.*?)##FOLLOWUP_END", text, flags=re.DOTALL)
    if followup_match:
        block = followup_match.group(1)
        suggestions = []
        for row in block.splitlines():
            row = row.strip()
            if row.startswith("suggestion_") and "=" in row:
                _, val = row.split("=", 1)
                if val.strip():
                    suggestions.append(val.strip())
        replacement = ""
        if suggestions:
            replacement = "\n\nSugerencias:\n" + "\n".join(f"  - {s}" for s in suggestions) + "\n"
        text = re.sub(r"##FOLLOWUP_START.*?##FOLLOWUP_END", replacement, text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Captura de mensajes OpenAI
# ---------------------------------------------------------------------------

def _rebuild_openai_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruye los mensajes que stream_data_response envía a OpenAI."""
    try:
        import orchestrator.data.response as resp_mod
    except ImportError:
        return []

    question = payload.get("question", "")
    observations = payload.get("observations") or {}
    entities_ctx = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    calc_mode_ctx = str(entities_ctx.get("calc_mode_cls") or "").strip().lower()
    if not calc_mode_ctx:
        calc_mode_ctx = str(
            (observations.get("classification") or {}).get("calc_mode") or ""
        ).strip().lower()

    messages = resp_mod._build_initial_messages()

    context_payload = {"calc_mode": calc_mode_ctx or None, "entities": entities_ctx}
    messages.append({
        "role": "system",
        "content": (
            "CONTEXTO DE CLASIFICACION (usar para priorizar metricas): "
            + json.dumps(context_payload, ensure_ascii=False)
        ),
    })

    builders = [
        (resp_mod._build_metric_priority_instruction, (calc_mode_ctx,)),
        (resp_mod._build_seasonality_strict_instruction, (), {"entities_ctx": entities_ctx, "observations": observations}),
        (resp_mod._build_incomplete_period_instruction, (), {"question": question, "entities_ctx": entities_ctx, "observations": observations}),
        (resp_mod._build_relative_period_fallback_instruction, (), {"question": question, "entities_ctx": entities_ctx, "observations": observations}),
        (resp_mod._build_no_explicit_period_latest_instruction, (), {"question": question, "entities_ctx": entities_ctx, "observations": observations}),
        (resp_mod._build_missing_activity_instruction, (), {"entities_ctx": entities_ctx, "observations": observations}),
    ]
    for entry in builders:
        fn = entry[0]
        args = entry[1] if len(entry) > 1 else ()
        kwargs = entry[2] if len(entry) > 2 else {}
        try:
            result = fn(*args, **kwargs)
        except Exception:
            result = None
        if result:
            messages.append({"role": "system", "content": result})

    messages.append({"role": "user", "content": question})
    return messages


class OpenAICapture:
    """Intercepta stream_data_response para capturar mensajes y tiempo."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.elapsed_seconds: float = 0.0
        self.stage_ms: Dict[str, float] = {}
        self._patched = False

    def install(self):
        """Aplica monkey-patch al nodo DATA para capturar I/O de OpenAI."""
        import orchestrator.graph.nodes.data as data_mod

        original_stream = data_mod.stream_data_response
        capture = self

        def _wrapped(payload: Dict[str, Any]):
            timing: Dict[str, Any] = {}
            instrumented_payload = dict(payload)
            instrumented_payload["_timing"] = timing
            capture.messages = _rebuild_openai_messages(payload)
            t0 = time.perf_counter()
            for chunk in original_stream(instrumented_payload):
                yield chunk
            if "_url_debug" in instrumented_payload:
                payload["_url_debug"] = instrumented_payload.get("_url_debug")
            capture.elapsed_seconds = time.perf_counter() - t0
            capture.stage_ms = {
                "openai_tool_calls_ms": float(timing.get("openai_tool_calls_ms", 0.0) or 0.0),
                "post_response_blocks_ms": float(timing.get("post_response_blocks_ms", 0.0) or 0.0),
            }

        data_mod.stream_data_response = _wrapped
        self._original = original_stream
        self._module = data_mod
        self._patched = True

    def uninstall(self):
        if self._patched:
            self._module.stream_data_response = self._original
            self._patched = False


# ---------------------------------------------------------------------------
# DetailTracer — recolecta datos por nodo y escribe el log
# ---------------------------------------------------------------------------

class DetailTracer:
    """Recolecta datos de cada nodo del grafo y escribe el log detallado.

    Se usa desde main.py como callback durante graph.stream.
    """

    def __init__(self, question: str, log_path: Optional[Path] = None):
        self._question = question
        self._log_path = log_path or LOG_PATH
        self._lines: List[str] = []
        self._classification = None
        self._predict_raw = None
        self._route_decision: Optional[str] = None
        self._data_classification: Optional[Dict] = None
        self._memory_input: Dict[str, Any] = {}
        self._current_state: Dict[str, Any] = {}
        self._url_debug: Dict[str, Any] = {}
        self._openai_capture: Optional[OpenAICapture] = None
        self._datastore_capture: Optional[DataStoreCapture] = None
        self._t0 = time.perf_counter()
        self._final_response_text: str = ""
        self._response_format_ms: float = 0.0

    def set_openai_capture(self, capture: OpenAICapture):
        self._openai_capture = capture

    def set_datastore_capture(self, capture: "DataStoreCapture"):
        self._datastore_capture = capture

    def on_node_update(self, node_name: str, delta: Any):
        """Alimenta el tracer con la salida de un nodo del grafo."""
        if not isinstance(delta, dict):
            return

        if node_name == "classify":
            self._classification = delta.get("classification")
            if self._classification is None:
                self._classification = self._current_state.get("classification")
            self._predict_raw = (
                getattr(self._classification, "predict_raw", None)
                if self._classification else None
            )

        if node_name in ("intent", "router"):
            decision = delta.get("route_decision")
            if decision:
                self._route_decision = decision
            elif not self._route_decision:
                self._route_decision = self._current_state.get("route_decision")

        if node_name == "data":
            self._data_classification = (
                delta.get("data_classification")
                or self._current_state.get("data_classification")
            )
            data_classification = self._data_classification if isinstance(self._data_classification, dict) else {}
            self._url_debug = (
                delta.get("url_debug")
                or self._current_state.get("url_debug")
                or data_classification.get("__url_debug")
                or {}
            )
            self._memory_input = {
                "output": delta.get("output", ""),
                "series": delta.get("series"),
                "parsed_point": delta.get("parsed_point"),
                "parsed_range": delta.get("parsed_range"),
            }

        self._current_state.update(delta)

    def flush(self):
        """Escribe el log completo al archivo."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._prepare_response()

        self._write_header()
        self._write_summary()
        self._write_response()

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._lines) + "\n")
        self._lines.clear()

    # -- secciones privadas --

    def _add(self, text: str = ""):
        self._lines.append(text)

    def _write_header(self):
        self._add(SEPARATOR)
        ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._add(f"TIMESTAMP: {ts}")
        self._add(f"\nQ: {self._question}")
        self._add("")

    def _write_summary(self):
        self._add("NODO CLASSIFY:")
        info = _extract_classification_info(self._classification)
        self._add(f"  INTENT      : {info.get('intent', '—')}")
        self._add(f"  CONFIDENCE  : {info.get('confidence', '—')}")
        self._add(f"  MACRO       : {info.get('macro', '—')}")
        self._add(f"  CONTEXT     : {info.get('context', '—')}")
        self._add(f"  CALC_MODE   : {info.get('calc_mode', '—')}")
        self._add(f"  ACTIVITY    : {info.get('activity', '—')}")
        self._add(f"  REQ_FORM    : {info.get('req_form', '—')}")
        route = (self._route_decision or "").strip().lower()
        fallback = (route == "fallback")
        self._add(f"  RUTA        : {self._route_decision or '—'}")
        self._add(f"  FALLBACK    : {_as_yes_no(fallback)}")
        self._add("  CLASIFICACION (predict_raw):")
        self._add(_indent(_pretty_json(info.get("predict_raw") or {}), 4))
        self._add("")
        self._add("TIEMPOS POR ETAPA (s):")
        timings = self._compute_stage_timings_seconds()
        self._add(f"  1. BUSQUEDA EN DATA_STORE: {timings.get('data_store_search_s', 0.0):.2f} s")
        self._add(f"  2. LLAMADA API OPEN AI (TOOL CALLS): {timings.get('openai_tool_calls_s', 0.0):.2f} s")
        self._add(f"  3. GENERACIÓN RESPUESTAS FUENTE,CSV y BLOQUE SUGERENCIAS: {timings.get('post_response_blocks_s', 0.0):.2f} s")
        self._add(f"  4. ORQUESTACIÓN GENERAL (INGEST/CLASSIFY/INTENT/ROUTER/MEMORY): {timings.get('general_orchestration_s', 0.0):.2f} s")
        self._add(f"  5. TIEMPO TOTAL: {timings.get('total_s', 0.0):.2f} s")
        self._add("")

        self._add("TRAZA URL (PRE-RESPUESTA):")
        llm_url_params = []
        url_builder_input = {}
        filtered_source_url = None
        if isinstance(self._url_debug, dict):
            llm_url_params = self._url_debug.get("llm_url_params") or []
            url_builder_input = self._url_debug.get("url_builder_input") or {}
            filtered_source_url = self._url_debug.get("filtered_source_url")
        self._add("  PARAMETROS URL PASADOS AL LLM (tool get_series_data):")
        self._add(_indent(_pretty_json(llm_url_params), 4))
        self._add("  PARAMETROS URL ANTES DE CONSTRUIR LINK FINAL:")
        self._add(_indent(_pretty_json(url_builder_input), 4))
        self._add("  URL FILTRADA GENERADA:")
        self._add(_indent(str(filtered_source_url or "—"), 4))
        self._add("")

    def _write_response(self):
        self._add("RESPUESTA:")
        self._add(_indent(self._final_response_text, 2))
        self._add("")
        self._add(SEPARATOR)
        self._add("")

    def _prepare_response(self):
        final_output = self._current_state.get("output", "")
        t0 = time.perf_counter()
        self._final_response_text = _format_final_response(final_output)
        self._response_format_ms = (time.perf_counter() - t0) * 1000.0

    def _compute_stage_timings_seconds(self) -> Dict[str, float]:
        data_store_ms = 0.0
        if self._datastore_capture is not None:
            data_store_ms = float(self._datastore_capture.elapsed_ms)

        openai_tool_ms = 0.0
        post_blocks_ms = self._response_format_ms
        if self._openai_capture is not None:
            openai_tool_ms = float(
                (self._openai_capture.stage_ms or {}).get("openai_tool_calls_ms", 0.0)
            )
            post_blocks_ms += float(
                (self._openai_capture.stage_ms or {}).get("post_response_blocks_ms", 0.0)
            )

        total_ms = (time.perf_counter() - self._t0) * 1000.0

        # Mostrar métricas que cuadren al nivel visible (2 decimales en segundos).
        data_store_s = round(data_store_ms / 1000.0, 2)
        openai_tool_s = round(openai_tool_ms / 1000.0, 2)
        post_blocks_s = round(post_blocks_ms / 1000.0, 2)
        total_s = round(total_ms / 1000.0, 2)
        general_orchestration_s = round(
            max(0.0, total_s - (data_store_s + openai_tool_s + post_blocks_s)),
            2,
        )

        return {
            "data_store_search_s": data_store_s,
            "openai_tool_calls_s": openai_tool_s,
            "post_response_blocks_s": post_blocks_s,
            "general_orchestration_s": general_orchestration_s,
            "total_s": total_s,
        }


class DataStoreCapture:
    """Intercepta búsqueda en data_store y acumula tiempo en milisegundos."""

    def __init__(self):
        self.elapsed_ms: float = 0.0
        self.calls: int = 0
        self._patched = False

    def install(self):
        import orchestrator.graph.nodes.data as data_mod

        original_search = data_mod.search_output_payloads
        capture = self

        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original_search(*args, **kwargs)
            finally:
                capture.elapsed_ms += (time.perf_counter() - t0) * 1000.0
                capture.calls += 1

        data_mod.search_output_payloads = _wrapped
        self._original = original_search
        self._module = data_mod
        self._patched = True

    def uninstall(self):
        if self._patched:
            self._module.search_output_payloads = self._original
            self._patched = False


# ---------------------------------------------------------------------------
# Ejecución standalone (CLI)
# ---------------------------------------------------------------------------

def run_detail_standalone(question: str) -> str:
    """Ejecuta el grafo completo y escribe el log detallado. Para uso CLI."""
    from orchestrator.graph.agent_graph import build_graph

    capture = OpenAICapture()
    capture.install()
    ds_capture = DataStoreCapture()
    ds_capture.install()

    tracer = DetailTracer(question)
    tracer.set_openai_capture(capture)
    tracer.set_datastore_capture(ds_capture)

    try:
        graph = build_graph()
        session_id = f"detail-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        state: Dict[str, Any] = {
            "question": question,
            "history": [],
            "context": {"session_id": session_id},
        }
        cfg = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "memory",
            },
        }

        for event in graph.stream(state, config=cfg, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for node_name, delta in event.items():
                tracer.on_node_update(node_name, delta)

        tracer.flush()
        print(f"Log generado en: {LOG_PATH}")
        return _format_final_response(tracer._current_state.get("output", ""))
    finally:
        capture.uninstall()
        ds_capture.uninstall()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Log detallado del flujo DATA → LLM → Respuesta"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Cual es el valor del imacec del ultimo mes",
    )
    args = parser.parse_args()

    sink = io.StringIO()
    with redirect_stderr(sink):
        response = run_detail_standalone(args.question)

    print(f"\nRESPUESTA:\n{response}")


if __name__ == "__main__":
    main()
