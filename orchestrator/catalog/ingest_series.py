"""
ingest_series.py
================
Lee catalog.json, obtiene observaciones de la API BDE vía BDEClient,
computa métricas derivadas (pct, yoy, delta, acceleration, direction, extrema, rankings),
agrega por suma (M→T→A, T→A) y guarda un JSON legible por cuadro en output/.

Uso:
    python ingest_series.py [--catalog catalog.json] [--output output]

Requiere variables de entorno BDE_USER y BDE_PASS (o archivo .env).
"""

import json
import logging
import re
import math
import unicodedata
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from orchestrator.api.bde_client import BDEClient
except ImportError:
    from bde_client import BDEClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROUND_DIGITS = 6
PIPELINE_VERSION = "v1.0.0"

# Keys whose arrays should be rendered inline (one line)
INLINE_ARRAY_KEYS: set[str] = set()

# Keys whose list items (dicts) should each be rendered on a single line
FLAT_RECORD_KEYS = {"records"}

# Fields to include in period-by-period records
_RECORD_FIELDS = ["value", "pct", "yoy_pct", "delta_abs", "yoy_delta_abs",
                  "acceleration_pct", "acceleration_yoy"]

# Fields dropped from output (not useful for LLM and consume tokens)
_DROP_FIELDS = {"comparable_prev_period", "comparable_yoy_period",
                "direction_pct", "direction_yoy"}


class CompactJSONEncoder(json.JSONEncoder):
    """Encoder que pone arrays de datos y dicts simples en una sola línea."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._indent_str = " " * self.indent if self.indent else ""

    def encode(self, o):
        return self._encode(o, level=0)

    # ------------------------------------------------------------------
    def _encode(self, o, level):
        indent = self._indent_str * level
        indent1 = self._indent_str * (level + 1)

        if isinstance(o, dict):
            if not o:
                return "{}"
            # Simple flat dict? (all values are scalars)
            if all(isinstance(v, (str, int, float, bool, type(None))) for v in o.values()):
                inner = ", ".join(
                    f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
                    for k, v in o.items()
                )
                one_line = "{" + inner + "}"
                if len(one_line) <= 120:
                    return one_line
            # Normal indented dict
            items = []
            for k, v in o.items():
                encoded_v = self._encode_value(k, v, level + 1)
                items.append(f"{indent1}{json.dumps(k, ensure_ascii=False)}: {encoded_v}")
            return "{\n" + ",\n".join(items) + f"\n{indent}}}"

        if isinstance(o, list):
            return self._encode_list(o, level)

        return json.dumps(o, ensure_ascii=False)

    # ------------------------------------------------------------------
    def _encode_value(self, key, v, level):
        """Encode a dict value; inline arrays for known data keys."""
        if isinstance(v, list) and key in INLINE_ARRAY_KEYS:
            return self._flat_list(v)
        if isinstance(v, list) and key in FLAT_RECORD_KEYS:
            return self._encode_flat_record_list(v, level)
        return self._encode(v, level)

    # ------------------------------------------------------------------
    def _encode_flat_record_list(self, lst, level):
        """Render a list of flat dicts, each on a single line."""
        if not lst:
            return "[]"
        indent1 = self._indent_str * (level + 1)
        indent = self._indent_str * level
        items = []
        for item in lst:
            if isinstance(item, dict):
                inner = ", ".join(
                    f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
                    for k, v in item.items()
                )
                items.append(f"{indent1}{{{inner}}}")
            else:
                items.append(f"{indent1}{json.dumps(item, ensure_ascii=False)}")
        return "[\n" + ",\n".join(items) + f"\n{indent}]"

    # ------------------------------------------------------------------
    @staticmethod
    def _flat_list(lst):
        """Render any list on a single line."""
        parts = [json.dumps(x, ensure_ascii=False) for x in lst]
        return "[" + ", ".join(parts) + "]"

    # ------------------------------------------------------------------
    def _encode_list(self, lst, level):
        if not lst:
            return "[]"
        # List of scalars → one line if short enough
        if all(isinstance(x, (str, int, float, bool, type(None))) for x in lst):
            one_line = self._flat_list(lst)
            if len(one_line) <= 120:
                return one_line
        # List of objects → one per line
        indent1 = self._indent_str * (level + 1)
        indent = self._indent_str * level
        items = [f"{indent1}{self._encode(x, level + 1)}" for x in lst]
        return "[\n" + ",\n".join(items) + f"\n{indent}]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    """Convierte texto a slug ASCII: minúsculas, sin acentos, underscores."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


def rnd(v: Optional[float]) -> Optional[float]:
    """Redondea a ROUND_DIGITS decimales, None-safe."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(v, ROUND_DIGITS)


def safe_pct(cur: float, prev: float) -> Optional[float]:
    """Calcula (cur/prev - 1) * 100 de forma segura."""
    if prev == 0 or prev is None:
        return None
    return (cur / prev - 1.0) * 100.0


def direction(val: Optional[float]) -> Optional[str]:
    if val is None:
        return None
    if val > 0:
        return "up"
    if val < 0:
        return "down"
    return "flat"


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def load_catalog(catalog_path: str) -> Dict[str, Any]:
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_contribution(classification: Dict) -> bool:
    cm = classification.get("calc_mode")
    if isinstance(cm, str):
        return cm == "contribution"
    if isinstance(cm, list):
        return "contribution" in cm
    return False


def is_share(classification: Dict) -> bool:
    cm = classification.get("calc_mode")
    if isinstance(cm, str):
        return cm == "share"
    if isinstance(cm, list):
        return "share" in cm
    return False


def should_skip_cuadro(cuadro_name: str, cuadro_data: Dict) -> Tuple[bool, str]:
    """Retorna (skip, reason)."""
    cls = cuadro_data.get("classification", {})
    series = cuadro_data.get("series", [])
    if not series:
        return True, "empty_series"
    return False, ""


# ---------------------------------------------------------------------------
# Frequency / period helpers
# ---------------------------------------------------------------------------

FREQ_MAP = {"M": "M", "T": "T", "A": "A"}
MONTH_TO_QUARTER = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,
                    7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}


def detect_frequency(series_id: str) -> str:
    """Extrae frecuencia del último carácter del series_id."""
    last = series_id.rsplit(".", 1)[-1]
    return FREQ_MAP.get(last, "T")  # default T si no se reconoce


def parse_date_string(date_str: str) -> Tuple[int, int, int]:
    """Parsea 'DD-MM-YYYY' -> (day, month, year)."""
    parts = date_str.split("-")
    return int(parts[0]), int(parts[1]), int(parts[2])


def make_period_id(date_str: str, freq: str) -> str:
    """Convierte date_str BDE a period_id normalizado."""
    _, month, year = parse_date_string(date_str)
    if freq == "M":
        return f"{year}-{month:02d}"
    elif freq == "T":
        q = MONTH_TO_QUARTER[month]
        return f"{year}-Q{q}"
    else:  # A
        return str(year)


def period_year(period_id: str) -> int:
    return int(period_id[:4])


def period_quarter(period_id: str) -> int:
    """Extrae el trimestre de un period_id tipo '1996-Q1'."""
    return int(period_id[-1])


def period_month(period_id: str) -> int:
    """Extrae el mes de un period_id tipo '1996-01'."""
    return int(period_id.split("-")[1])


# ---------------------------------------------------------------------------
# Data computation
# ---------------------------------------------------------------------------

def compute_derived(periods: List[str], values: List[float], freq: str) -> Dict[str, List]:
    """Calcula todas las métricas derivadas para un bloque de datos."""
    n = len(values)
    k_yoy = {"M": 12, "T": 4, "A": 1}[freq]

    pct = []
    yoy_pct = []
    delta_abs = []
    yoy_delta_abs = []
    direction_pct = []
    direction_yoy = []
    acceleration_pct_list = []
    acceleration_yoy_list = []
    comparable_prev = []
    comparable_yoy = []

    for i in range(n):
        # pct vs previous
        if i >= 1 and values[i - 1] is not None and values[i] is not None:
            p = safe_pct(values[i], values[i - 1])
            d = values[i] - values[i - 1]
        else:
            p = None
            d = None
        pct.append(rnd(p))
        delta_abs.append(rnd(d))
        direction_pct.append(direction(p))
        comparable_prev.append(periods[i - 1] if i >= 1 else None)

        # yoy
        if i >= k_yoy and values[i - k_yoy] is not None and values[i] is not None:
            yp = safe_pct(values[i], values[i - k_yoy])
            yd = values[i] - values[i - k_yoy]
        else:
            yp = None
            yd = None
        yoy_pct.append(rnd(yp))
        yoy_delta_abs.append(rnd(yd))
        direction_yoy.append(direction(yp))
        comparable_yoy.append(periods[i - k_yoy] if i >= k_yoy else None)

    # accelerations (differences of rates)
    for i in range(n):
        if i >= 1 and pct[i] is not None and pct[i - 1] is not None:
            acceleration_pct_list.append(rnd(pct[i] - pct[i - 1]))
        else:
            acceleration_pct_list.append(None)

        if i >= 1 and yoy_pct[i] is not None and yoy_pct[i - 1] is not None:
            acceleration_yoy_list.append(rnd(yoy_pct[i] - yoy_pct[i - 1]))
        else:
            acceleration_yoy_list.append(None)

    return {
        "pct": pct,
        "yoy_pct": yoy_pct,
        "delta_abs": delta_abs,
        "yoy_delta_abs": yoy_delta_abs,
        "direction_pct": direction_pct,
        "direction_yoy": direction_yoy,
        "acceleration_pct": acceleration_pct_list,
        "acceleration_yoy": acceleration_yoy_list,
        "comparable_prev_period": comparable_prev,
        "comparable_yoy_period": comparable_yoy,
    }


# ---------------------------------------------------------------------------
# Aggregation (always SUM, only complete periods)
# ---------------------------------------------------------------------------

def aggregate_monthly_to_quarterly(periods: List[str], values: List[float]) -> Tuple[List[str], List[float]]:
    """Agrupa meses en trimestres por suma (sólo trimestres con 3 meses)."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for pid, val in zip(periods, values):
        if val is None:
            continue
        y = period_year(pid)
        m = period_month(pid)
        q = MONTH_TO_QUARTER[m]
        q_key = f"{y}-Q{q}"
        buckets[q_key].append(val)

    result_periods = []
    result_values = []
    for q_key in sorted(buckets.keys()):
        vals = buckets[q_key]
        if len(vals) == 3:
            result_periods.append(q_key)
            result_values.append(rnd(sum(vals)))
    return result_periods, result_values


def aggregate_monthly_to_annual(periods: List[str], values: List[float]) -> Tuple[List[str], List[float]]:
    """Agrupa meses en años por suma (sólo años con 12 meses)."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for pid, val in zip(periods, values):
        if val is None:
            continue
        y = str(period_year(pid))
        buckets[y].append(val)

    result_periods = []
    result_values = []
    for y_key in sorted(buckets.keys()):
        vals = buckets[y_key]
        if len(vals) == 12:
            result_periods.append(y_key)
            result_values.append(rnd(sum(vals)))
    return result_periods, result_values


def aggregate_quarterly_to_annual(periods: List[str], values: List[float]) -> Tuple[List[str], List[float]]:
    """Agrupa trimestres en años por suma (sólo años con 4 trimestres)."""
    buckets: Dict[str, List[float]] = defaultdict(list)
    for pid, val in zip(periods, values):
        if val is None:
            continue
        y = str(period_year(pid))
        buckets[y].append(val)

    result_periods = []
    result_values = []
    for y_key in sorted(buckets.keys()):
        vals = buckets[y_key]
        if len(vals) == 4:
            result_periods.append(y_key)
            result_values.append(rnd(sum(vals)))
    return result_periods, result_values


# ---------------------------------------------------------------------------
# Build data block
# ---------------------------------------------------------------------------

def build_data_block(periods: List[str], values: List[float], freq: str, origin: str, *, skip_derived: bool = False) -> Dict:
    """Construye un bloque de datos con métricas derivadas.

    Args:
        skip_derived: Si True, omite el cálculo de tasas derivadas (pct, yoy_pct,
            delta_abs, etc.). Útil para cuadros de contribución/participación
            donde los valores ya son porcentajes y las tasas no tienen sentido.
    """
    block: Dict[str, Any] = {
        "frequency": freq,
        "origin": origin,
        "aggregation": "sum",
        "latest_period": periods[-1] if periods else None,
        "periods": periods,
        "value": values,
    }
    if not skip_derived:
        block.update(compute_derived(periods, values, freq))
    return block


# ---------------------------------------------------------------------------
# Extrema
# ---------------------------------------------------------------------------

def compute_extrema_for_block(block: Dict) -> Dict:
    """Computa max/min de value, pct, yoy_pct para un bloque de datos."""
    periods = block["periods"]
    result = {}

    for field in ["value", "pct", "yoy_pct"]:
        arr = block.get(field, [])
        valid = [(i, v) for i, v in enumerate(arr) if v is not None]
        if not valid:
            result[f"{field}_max"] = {"period": None, "value": None}
            result[f"{field}_min"] = {"period": None, "value": None}
        else:
            i_max = max(valid, key=lambda x: x[1])
            i_min = min(valid, key=lambda x: x[1])
            result[f"{field}_max"] = {"period": periods[i_max[0]], "value": i_max[1]}
            result[f"{field}_min"] = {"period": periods[i_min[0]], "value": i_min[1]}
    return result


# ---------------------------------------------------------------------------
# Latest snapshot
# ---------------------------------------------------------------------------

def build_snapshot_for_block(block: Dict) -> Optional[Dict]:
    """Extrae datos del último periodo de un bloque."""
    if not block["periods"]:
        return None
    i = len(block["periods"]) - 1
    snapshot = {"period": block["periods"][i]}
    for field in ["value", "pct", "yoy_pct", "delta_abs", "yoy_delta_abs",
                   "direction_pct", "direction_yoy", "acceleration_pct", "acceleration_yoy"]:
        arr = block.get(field, [])
        snapshot[field] = arr[i] if i < len(arr) else None
    return snapshot


# ---------------------------------------------------------------------------
# Convert block to period-by-period records
# ---------------------------------------------------------------------------

def _convert_block_to_records(block: Dict) -> None:
    """Convierte arrays paralelos a lista de dicts {period, value, pct, ...}.

    Elimina campos redundantes y reemplaza los arrays por una clave 'records'.
    Esto evita alucinaciones del LLM por desalineación de índices.
    """
    periods = block.get("periods")
    if not periods:
        return

    n = len(periods)
    records = []
    for i, p in enumerate(periods):
        rec = {"period": p}
        for field in _RECORD_FIELDS:
            arr = block.get(field)
            if arr and i < len(arr) and arr[i] is not None:
                rec[field] = arr[i]
        records.append(rec)

    # Remove parallel arrays and droppable fields
    block.pop("periods", None)
    for field in _RECORD_FIELDS:
        block.pop(field, None)
    for field in _DROP_FIELDS:
        block.pop(field, None)
    block["records"] = records


# ---------------------------------------------------------------------------
# Build one series
# ---------------------------------------------------------------------------

def build_series_output(series_info: Dict, client: BDEClient, fetched_at: str, *, skip_derived: bool = False) -> Optional[Dict]:
    """Construye el output enriquecido de una serie individual."""
    series_id = series_info["id"]
    freq = detect_frequency(series_id)

    # Fetch observations
    try:
        obs = client.fetch_series(series_id)
    except Exception as e:
        logger.error(f"  Error fetching {series_id}: {e}")
        return None

    if not obs:
        logger.warning(f"  No observations for {series_id}")
        return None

    # Parse and sort observations
    parsed = []
    for o in obs:
        date_str = o.get("date") or o.get("indexDateString")
        val = o.get("value")
        if date_str is None or val is None:
            continue
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            continue
        pid = make_period_id(date_str, freq)
        parsed.append((pid, rnd(val_f)))

    parsed.sort(key=lambda x: x[0])
    if not parsed:
        logger.warning(f"  No valid observations after parsing for {series_id}")
        return None

    base_periods = [p[0] for p in parsed]
    base_values = [p[1] for p in parsed]

    # Build data blocks per frequency
    data = {}
    latest_snapshot = {}
    extrema = {}

    # Native frequency block
    native_block = build_data_block(base_periods, base_values, freq, "official", skip_derived=skip_derived)
    data[freq] = native_block
    latest_snapshot[freq] = build_snapshot_for_block(native_block)
    extrema[freq] = compute_extrema_for_block(native_block)

    # Derived aggregations
    if freq == "M":
        # M -> T
        q_periods, q_values = aggregate_monthly_to_quarterly(base_periods, base_values)
        if q_periods:
            q_block = build_data_block(q_periods, q_values, "T", "derived_from_M", skip_derived=skip_derived)
            data["T"] = q_block
            latest_snapshot["T"] = build_snapshot_for_block(q_block)
            extrema["T"] = compute_extrema_for_block(q_block)
        # M -> A
        a_periods, a_values = aggregate_monthly_to_annual(base_periods, base_values)
        if a_periods:
            a_block = build_data_block(a_periods, a_values, "A", "derived_from_M", skip_derived=skip_derived)
            data["A"] = a_block
            latest_snapshot["A"] = build_snapshot_for_block(a_block)
            extrema["A"] = compute_extrema_for_block(a_block)

    elif freq == "T":
        # T -> A
        a_periods, a_values = aggregate_quarterly_to_annual(base_periods, base_values)
        if a_periods:
            a_block = build_data_block(a_periods, a_values, "A", "derived_from_T", skip_derived=skip_derived)
            data["A"] = a_block
            latest_snapshot["A"] = build_snapshot_for_block(a_block)
            extrema["A"] = compute_extrema_for_block(a_block)

    # freq == "A" -> no further aggregation

    return {
        "series_id": series_id,
        "short_title": series_info.get("short_title", ""),
        "long_title": series_info.get("long_title", ""),
        "classification_series": series_info.get("classification", {}),
        "series_freq": freq,
        "aggregation_rule": "sum",
        "is_contribution": skip_derived,
        "derived_rates_enabled": not skip_derived,
        "ingestion": {
            "fetched_at_utc": fetched_at,
            "source": "BDE",
            "source_series_id": series_id,
        },
        "data": data,
        "latest_snapshot": latest_snapshot,
        "extrema": extrema,
    }


# ---------------------------------------------------------------------------
# Cuadro summaries (rankings + cuadro extrema)
# ---------------------------------------------------------------------------

def build_cuadro_summaries(series_outputs: List[Dict]) -> Dict:
    """Construye rankings (todos los períodos) y extrema a nivel cuadro."""
    all_freqs: set[str] = set()
    for s in series_outputs:
        all_freqs.update(s["data"].keys())

    rankings: Dict[str, Dict[str, list]] = {}
    cuadro_extrema: Dict[str, Dict] = {}

    for freq in sorted(all_freqs):
        # Build period -> [entry, …] mapping across all series
        period_entries: Dict[str, list] = {}
        for s in series_outputs:
            block = s["data"].get(freq)
            if not block or not block["periods"]:
                continue
            periods = block["periods"]
            values = block["value"]
            pcts = block.get("pct") or [None] * len(periods)
            yoys = block.get("yoy_pct") or [None] * len(periods)
            for i, period in enumerate(periods):
                period_entries.setdefault(period, []).append({
                    "series_id": s["series_id"],
                    "value": values[i],
                    "pct": pcts[i],
                    "yoy_pct": yoys[i],
                })

        rankings[freq] = {p: entries for p, entries in sorted(period_entries.items())}

        # Cuadro-level extrema across all series for this freq
        fex = {}
        for field in ["value", "pct", "yoy_pct"]:
            all_entries = []
            for s in series_outputs:
                block = s["data"].get(freq)
                if not block:
                    continue
                arr = block.get(field, [])
                pids = block["periods"]
                for i, v in enumerate(arr):
                    if v is not None:
                        all_entries.append((v, pids[i], s["series_id"]))
            if all_entries:
                mx = max(all_entries, key=lambda x: x[0])
                mn = min(all_entries, key=lambda x: x[0])
                fex[f"{field}_max"] = {"series_id": mx[2], "period": mx[1], "value": mx[0]}
                fex[f"{field}_min"] = {"series_id": mn[2], "period": mn[1], "value": mn[0]}
            else:
                fex[f"{field}_max"] = {"series_id": None, "period": None, "value": None}
                fex[f"{field}_min"] = {"series_id": None, "period": None, "value": None}
        cuadro_extrema[freq] = fex

    return {
        "rankings": rankings,
        "cuadro_extrema": cuadro_extrema,
    }


# ---------------------------------------------------------------------------
# Build one cuadro
# ---------------------------------------------------------------------------

def build_cuadro_output(cuadro_name: str, cuadro_data: Dict, client: BDEClient) -> Dict:
    """Construye el JSON completo de un cuadro."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    classification = cuadro_data.get("classification", {})
    series_list = cuadro_data.get("series", [])

    skip_derived = is_contribution(classification) or is_share(classification)

    series_outputs = []
    for i, s_info in enumerate(series_list):
        sid = s_info.get("id", "")
        logger.info(f"  [{i+1}/{len(series_list)}] Processing {sid}")
        result = build_series_output(s_info, client, now_utc, skip_derived=skip_derived)
        if result:
            series_outputs.append(result)

    # Determine latest_available per freq
    latest_available = {}
    for s in series_outputs:
        for freq, block in s["data"].items():
            lp = block.get("latest_period")
            if lp:
                if freq not in latest_available or lp > latest_available[freq]:
                    latest_available[freq] = lp

    cuadro_summaries = build_cuadro_summaries(series_outputs) if series_outputs else {}

    # Convert parallel arrays to period-by-period records for LLM consumption
    for s in series_outputs:
        for _freq_key, block in s["data"].items():
            _convert_block_to_records(block)
        s.pop("ingestion", None)

    return {
        "dataset_meta": {
            "generated_at_utc": now_utc,
            "pipeline_version": PIPELINE_VERSION,
            "source": "BDE",
            "source_type": "catalog_cuadro_payload",
            "build_scope": "full_cuadro",
            "null_policy": "insufficient_history_or_division_by_zero",
            "rounding": {
                "value": ROUND_DIGITS,
                "pct": ROUND_DIGITS,
                "yoy_pct": ROUND_DIGITS,
                "delta_abs": ROUND_DIGITS,
                "yoy_delta_abs": ROUND_DIGITS,
                "acceleration_pct": ROUND_DIGITS,
                "acceleration_yoy": ROUND_DIGITS,
            },
        },
        "cuadro_id": slugify(cuadro_name),
        "cuadro_name": cuadro_name,
        "source_url": cuadro_data.get("source_url", ""),
        "classification": classification,
        "notes": {
            "period_format": {"M": "YYYY-MM", "T": "YYYY-QN", "A": "YYYY"},
            "period_examples": {"M": "1996-01", "T": "1996-Q1", "A": "1996"},
            "aggregation_rule_default": "sum",
            "rates_definition": {
                "pct": "(value_t / value_t-1 - 1) * 100",
                "yoy_pct": "(value_t / value_t-k - 1) * 100, con k=12 para M, k=4 para T, k=1 para A",
                "delta_abs": "value_t - value_t-1",
                "yoy_delta_abs": "value_t - value_t-k",
                "acceleration_pct": "pct_t - pct_t-1",
                "acceleration_yoy": "yoy_pct_t - yoy_pct_t-1",
            },
            "contribution_policy": "no_derived_rates",
        },
        "latest_available": latest_available,
        "series": series_outputs,
        "cuadro_summaries": cuadro_summaries,
    }


# ---------------------------------------------------------------------------
# Split by frequency
# ---------------------------------------------------------------------------

def split_by_frequency(cuadro_json: dict) -> dict[str, dict]:
    """Split a cuadro JSON into one dict per frequency, keeping freq-keyed structure."""
    all_freqs: set[str] = set()
    for s in cuadro_json.get("series", []):
        all_freqs.update(s.get("data", {}).keys())
    if not all_freqs:
        return {}

    result: dict[str, dict] = {}
    for freq in sorted(all_freqs):
        fj = {k: v for k, v in cuadro_json.items()
              if k not in ("latest_available", "series", "cuadro_summaries")}
        fj["frequency"] = freq

        la = cuadro_json.get("latest_available", {}).get(freq)
        fj["latest_available"] = {freq: la} if la else {}

        fj["series"] = []
        for s in cuadro_json.get("series", []):
            if freq not in s.get("data", {}):
                continue
            fs = {k: v for k, v in s.items()
                  if k not in ("data", "latest_snapshot", "extrema")}
            fs["data"] = {freq: s["data"][freq]}
            snap = s.get("latest_snapshot", {})
            fs["latest_snapshot"] = {freq: snap[freq]} if freq in snap else {}
            ext = s.get("extrema", {})
            fs["extrema"] = {freq: ext[freq]} if freq in ext else {}
            fj["series"].append(fs)

        fsum = cuadro_json.get("cuadro_summaries", {})
        fj["cuadro_summaries"] = {}
        rankings = fsum.get("rankings", {})
        if freq in rankings:
            fj["cuadro_summaries"]["rankings"] = {freq: rankings[freq]}
        fextrema = fsum.get("cuadro_extrema", {})
        if freq in fextrema:
            fj["cuadro_summaries"]["cuadro_extrema"] = {freq: fextrema[freq]}

        result[freq] = fj
    return result


# ---------------------------------------------------------------------------
# Programmatic entry point (called from main.py)
# ---------------------------------------------------------------------------

def run_ingest(catalog_path: str, output_dir: str, store_dir: str = None) -> int:
    """Ejecuta el ingest completo. Retorna cantidad de cuadros procesados."""
    catalog_p = Path(catalog_path)
    output_p = Path(output_dir)
    output_p.mkdir(parents=True, exist_ok=True)

    store_p = Path(store_dir) if store_dir else Path(__file__).resolve().parent / "data_store"
    client = BDEClient(store_dir=store_p)

    logger.info(f"Loading catalog from {catalog_p}")
    catalog = load_catalog(str(catalog_p))
    logger.info(f"Catalog has {len(catalog)} cuadros")

    encoder = CompactJSONEncoder(indent=2, ensure_ascii=False)
    processed = 0
    skipped = 0

    for cuadro_name, cuadro_data in catalog.items():
        skip, reason = should_skip_cuadro(cuadro_name, cuadro_data)
        if skip:
            logger.info(f"SKIP [{reason}]: {cuadro_name}")
            skipped += 1
            continue

        logger.info(f"PROCESSING: {cuadro_name}")
        cuadro_json = build_cuadro_output(cuadro_name, cuadro_data, client)

        slug = slugify(cuadro_name)
        freq_splits = split_by_frequency(cuadro_json)
        for freq, freq_json in freq_splits.items():
            out_path = output_p / f"{slug}_{freq}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(encoder.encode(freq_json))
                f.write("\n")

        freqs = ", ".join(sorted(freq_splits.keys()))
        n_series = len(cuadro_json["series"])
        logger.info(f"  -> Saved {slug}_[{freqs}].json ({n_series} series)")
        processed += 1

    logger.info(f"Done. Processed: {processed}, Skipped: {skipped}")
    return processed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest series from BDE catalog and build cuadro JSONs")
    parser.add_argument("--catalog", default="catalog.json", help="Path to catalog.json")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--store-dir", default=None, help="BDE data store directory")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    store_dir = Path(args.store_dir) if args.store_dir else Path(__file__).resolve().parent / "data_store"
    client = BDEClient(store_dir=store_dir)

    logger.info(f"Loading catalog from {catalog_path}")
    catalog = load_catalog(str(catalog_path))
    logger.info(f"Catalog has {len(catalog)} cuadros")

    encoder = CompactJSONEncoder(indent=2, ensure_ascii=False)
    processed = 0
    skipped = 0

    for cuadro_name, cuadro_data in catalog.items():
        skip, reason = should_skip_cuadro(cuadro_name, cuadro_data)
        if skip:
            logger.info(f"SKIP [{reason}]: {cuadro_name}")
            skipped += 1
            continue

        logger.info(f"PROCESSING: {cuadro_name}")
        cuadro_json = build_cuadro_output(cuadro_name, cuadro_data, client)

        slug = slugify(cuadro_name)
        freq_splits = split_by_frequency(cuadro_json)
        for freq, freq_json in freq_splits.items():
            out_path = output_dir / f"{slug}_{freq}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(encoder.encode(freq_json))
                f.write("\n")

        freqs = ", ".join(sorted(freq_splits.keys()))
        n_series = len(cuadro_json["series"])
        logger.info(f"  -> Saved {slug}_[{freqs}].json ({n_series} series)")
        processed += 1

    logger.info(f"Done. Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
