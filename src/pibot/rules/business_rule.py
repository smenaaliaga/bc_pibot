"""Business rules for PIB/IMACEC queries.

Siempre se consultará por datos y no por consultas metodológicas.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

LABEL_FLAGS: Dict[str, bool] = {
    "seasonality": False,
    "price": False,
    "history": False,
}

_METADATA_CACHE: Dict[str, Any] = {}

PRICE_RULES = {
    "co": re.compile(r"\b(precio|precios)\s+corrientes?\b|\bcorrientes?\b", re.IGNORECASE),
    "en": re.compile(r"\b(precio|precios)\s+encadenados?\b|\bencadenados?\b", re.IGNORECASE),
}

SEASONALITY_RULES = {
    "sa": re.compile(r"desestacionalizad[oa]s?|ajustad[oa]s?\s+estacional", re.IGNORECASE),
    "nsa": re.compile(r"\b(sin\s+desestacionalizar|original)\b", re.IGNORECASE),
}

HISTORY_RULES = {
    "inf_historic": re.compile(r"\bhist[óo]ric[oa]s?\b|\bantes\s+de\s+1996\b", re.IGNORECASE),
}

YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")

CALC_MODE_INTERANUAL_PATTERN = re.compile(
    r"\b(interanual|a/a|anual\s+anterior|mismo\s+periodo\s+del\s+ano\s+anterior|mismo\s+per[ií]odo\s+del\s+a[nñ]o\s+anterior)\b",
    re.IGNORECASE,
)
CALC_MODE_PREV_PATTERN = re.compile(
    r"\b(periodo\s+anterior|per[ií]odo\s+anterior|trimestre\s+anterior|mes\s+anterior|qoq|mom|t/t|m/m|c/r\s+al\s+periodo\s+anterior|c/r\s+al\s+per[ií]odo\s+anterior)\b",
    re.IGNORECASE,
)


def _first_or_none(value: Any) -> Any:
    if isinstance(value, list):
        return next((item for item in value if item not in (None, "")), None)
    return value


def resolve_region_value(value: Any) -> Optional[str]:
    if value in (None, "", [], {}):
        return None

    if isinstance(value, list):
        tokens = [str(item).strip().lower() for item in value if str(item).strip()]
        if not tokens:
            return None
        if len(tokens) >= 2:
            return tokens[1]
        return tokens[0]

    if isinstance(value, str):
        tokens = [token for token in value.strip().lower().split() if token]
        if not tokens:
            return None
        if len(tokens) >= 2:
            return tokens[1]
        return tokens[0]

    text = str(value).strip().lower()
    return text or None


def _extract_indicator(predict_raw: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(predict_raw, dict):
        return None
    interpretation = predict_raw.get("interpretation") or {}
    entities_normalized = interpretation.get("entities_normalized") or {}
    indicator = _first_or_none(entities_normalized.get("indicator"))
    if indicator:
        return str(indicator).lower()
    entities = interpretation.get("entities") or {}
    indicator = _first_or_none(entities.get("indicator"))
    if indicator:
        return str(indicator).lower()
    return None


def classify_seasonality(question: str) -> Optional[str]:
    for label, pattern in SEASONALITY_RULES.items():
        if pattern.search(question):
            return label
    return None


def classify_price(question: str) -> Optional[str]:
    for label, pattern in PRICE_RULES.items():
        if pattern.search(question):
            return label
    return "co"


def classify_history(question: str, indicator: Optional[str]) -> str:
    if indicator and indicator.lower() == "pib":
        if HISTORY_RULES["inf_historic"].search(question):
            return "inf_historic"
        years = [int(y) for y in YEAR_PATTERN.findall(question)]
        if any(year < 1996 for year in years):
            return "inf_historic"
    return "2018"


def classify_by_regex(question: str, indicator: Optional[str]) -> Dict[str, Optional[str]]:
    return {
        "seasonality": classify_seasonality(question),
        "price": classify_price(question),
        "history": classify_history(question, indicator),
    }


def resolve_calc_mode_cls(
    *,
    question: str,
    calc_mode_cls: Any,
    intent_cls: Any,
    req_form_cls: Any,
    frequency: Any,
) -> str:
    """Normalize calc_mode to align business rules and response generation.

    Priority:
    1) Explicit calc_mode from classifier (normalized aliases).
    2) Strong textual clues in user question.
    3) Business defaults for value/latest queries.
    """

    raw_mode = str(calc_mode_cls or "").strip().lower()
    intent = str(intent_cls or "").strip().lower()
    req_form = str(req_form_cls or "").strip().lower()
    freq = str(frequency or "").strip().lower()
    question_text = str(question or "")

    def _normalize_alias(mode: str) -> Optional[str]:
        if not mode or mode in {"none", "null", "nan"}:
            return None
        if mode in {"yoy", "ypct", "interanual", "annual", "anual"}:
            return "yoy"
        if mode in {"prev_period", "pct", "mom", "qoq", "period_previous", "periodo_anterior"}:
            return "prev_period"
        if mode in {"original", "value", "level", "valor"}:
            return "original"
        return mode

    normalized_explicit = _normalize_alias(raw_mode)
    if normalized_explicit:
        if (
            normalized_explicit == "original"
            and intent == "value"
            and req_form in {"latest", "point", "specific_point", "range"}
            and freq in {"q", "m", "a"}
        ):
            return "yoy"
        return normalized_explicit

    if CALC_MODE_INTERANUAL_PATTERN.search(question_text):
        return "yoy"
    if CALC_MODE_PREV_PATTERN.search(question_text):
        return "prev_period"

    if intent == "value" and req_form in {"latest", "point", "specific_point", "range"}:
        if freq in {"q", "m", "a"}:
            return "yoy"
        return "original"

    if req_form in {"latest", "point", "specific_point", "range"}:
        return "yoy"

    return "yoy"


def classify_headers(
    question: str,
    predict_raw: Optional[Dict[str, Any]] = None,
    enabled: Optional[Dict[str, bool]] = None,
) -> Dict[str, Optional[str]]:
    flags = dict(LABEL_FLAGS)
    if enabled:
        flags.update(enabled)

    indicator = _extract_indicator(predict_raw)
    results = classify_by_regex(question, indicator)

    if not flags.get("seasonality", False):
        results["seasonality"] = None
    if not flags.get("price", False):
        results["price"] = None
    if not flags.get("history", False):
        results["history"] = None

    return results


def _resolve_question(row: Dict[str, Any]) -> Optional[str]:
    for key in ("question", "pregunta", "text", "query"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def benchmark(csv_path: str, enabled: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: Iterable[Dict[str, Any]]
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    totals = {"seasonality": 0, "price": 0, "history": 0}
    hits = {"seasonality": 0, "price": 0, "history": 0}

    t0 = time.perf_counter()
    for row in rows:
        question = _resolve_question(row)
        if not question:
            continue
        predicted = classify_headers(question, enabled=enabled)
        for label in ("seasonality", "price", "history"):
            expected = row.get(label)
            if expected is None or str(expected).strip() == "":
                continue
            totals[label] += 1
            if str(predicted.get(label, "")).lower() == str(expected).lower():
                hits[label] += 1
    elapsed = time.perf_counter() - t0

    accuracy = {
        label: (hits[label] / totals[label]) if totals[label] else None
        for label in totals
    }
    speed = {
        "total_seconds": elapsed,
        "avg_ms_per_row": (elapsed / max(len(rows), 1)) * 1000.0,
    }
    return {"accuracy": accuracy, "speed": speed, "totals": totals}


def _load_metadata_q() -> Dict[str, Any]:
    if _METADATA_CACHE:
        return _METADATA_CACHE
    path = Path(__file__).resolve().parents[1] / "orchestrator" / "catalog" / "metadata_q.json"
    if not path.exists():
        _METADATA_CACHE["__error__"] = f"metadata_q.json not found: {path}"
        return _METADATA_CACHE
    _METADATA_CACHE.update(json.loads(path.read_text(encoding="utf-8")))
    return _METADATA_CACHE


def build_metadata_key(data_params: Dict[str, Any]) -> str:
    order = [
        "activity_cls",
        "frequency",
        "calc_mode_cls",
        "region_cls",
        "investment_cls",
        "req_form_cls",
        "activity_value",
        "sub_activity_value",
        "region_value",
        "investment_value",
        "indicator",
        "seasonality",
        "gasto_value",
        "price",
        "history",
    ]
    values = []
    for key in order:
        value = data_params.get(key)
        values.append(str(value))
    return "::".join(values)


def build_metadata_response(data_params: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _load_metadata_q()
    if "__error__" in metadata:
        return {"error": metadata["__error__"]}
    key = build_metadata_key(data_params)
    data = metadata.get("data") or []
    if not isinstance(data, list):
        return {"key": key, "match": None}
    lookup = {
        entry.get("key"): entry
        for entry in data
        if isinstance(entry, dict) and entry.get("key")
    }
    entry = lookup.get(key)
    resolved_key = key

    if not entry:
        req_form = str(data_params.get("req_form_cls") or "").lower()
        if req_form in {"point", "range"}:
            fallback_order = ("latest", "range") if req_form == "point" else ("range", "latest")
            for fallback_req_form in fallback_order:
                params_fallback = dict(data_params)
                params_fallback["req_form_cls"] = fallback_req_form
                candidate_key = build_metadata_key(params_fallback)
                candidate_entry = lookup.get(candidate_key)
                if candidate_entry:
                    entry = candidate_entry
                    resolved_key = candidate_key
                    break

    if not entry:
        return {"key": key, "match": None}

    calc_mode = str(data_params.get("calc_mode_cls") or "").lower()
    entry_series = entry.get("series") if isinstance(entry, dict) else None
    if calc_mode == "contribution" and (not isinstance(entry_series, dict) or not entry_series):
        params_contrib = dict(data_params)
        params_contrib["activity_cls"] = "general"
        params_contrib["activity_value"] = "none"
        contrib_key = build_metadata_key(params_contrib)
        contrib_entry = lookup.get(contrib_key)
        contrib_series = contrib_entry.get("series") if isinstance(contrib_entry, dict) else None
        if isinstance(contrib_series, dict) and contrib_series:
            entry = contrib_entry
            resolved_key = contrib_key

    classification = entry.get("classification") or {}
    indicator = str(classification.get("indicator") or "").lower()
    req_form = str(classification.get("req_form_cls") or "").lower()

    latest_update = entry.get("latest_update")
    if not latest_update:
        latest_update = "none"
        if req_form == "latest":
            if indicator == "imacec":
                latest_update = "2025-12-01"
            elif indicator == "pib":
                latest_update = "2025-10-01"

    return {
        "key": resolved_key,
        "label": entry.get("label"),
        "serie_default": entry.get("serie_default"),
        "title_serie_default": entry.get("title_serie_default"),
        "series": entry.get("series"),
        "sources_url": entry.get("sources_url"),
        "latest_update": latest_update,
    }


def apply_latest_update_period(
    data_params: Dict[str, Any],
    metadata_response: Dict[str, Any],
) -> Optional[List[str]]:
    if not isinstance(metadata_response, dict):
        return None

    req_form = str(data_params.get("req_form_cls") or "").lower()
    if req_form != "latest":
        return None

    latest_update = str(metadata_response.get("latest_update") or "").strip()
    if not latest_update or latest_update.lower() == "none":
        return None

    frequency = str(data_params.get("frequency") or "").lower()
    try:
        year, month, day = [int(part) for part in latest_update.split("-")[:3]]
    except Exception:
        return None

    if frequency == "m":
        start = f"{year:04d}-{month:02d}-01"
        end = f"{year:04d}-{month:02d}-{day:02d}"
        return [start, end]

    if frequency == "q":
        start_month = month - 3
        start_year = year
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        start = f"{start_year:04d}-{start_month:02d}-01"
        end = f"{year:04d}-{month:02d}-{day:02d}"
        return [start, end]

    if frequency == "a":
        complete_year = year if (month == 12 and day >= 31) else (year - 1)
        if complete_year < 1900:
            return None
        return [f"{complete_year:04d}-01-01", f"{complete_year:04d}-12-31"]

    return None


def resolve_pib_annual_validity(
    data_params: Dict[str, Any],
    metadata_response: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate if requested PIB annual year is available given latest_update.

    Returns a dict with:
      - applies: bool
      - is_valid: bool
      - requested_year: Optional[int]
      - max_valid_year: Optional[int]
      - resolved_period: Optional[List[str]]
      - message: Optional[str]
    """

    indicator = str(data_params.get("indicator") or "").strip().lower()
    frequency = str(data_params.get("frequency") or "").strip().lower()
    req_form = str(data_params.get("req_form_cls") or "").strip().lower()

    applies = indicator == "pib" and frequency == "a" and req_form in {"latest", "point"}
    base = {
        "applies": applies,
        "is_valid": True,
        "requested_year": None,
        "max_valid_year": None,
        "resolved_period": None,
        "message": None,
    }
    if not applies:
        return base

    latest_update = str(metadata_response.get("latest_update") or "").strip()
    if not latest_update or latest_update.lower() == "none":
        return base

    try:
        year, month, day = [int(part) for part in latest_update.split("-")[:3]]
    except Exception:
        return base

    max_valid_year = year if (month == 12 and day >= 31) else (year - 1)
    if max_valid_year < 1900:
        return base

    period = data_params.get("period")
    requested_year: Optional[int] = None
    if isinstance(period, list) and period:
        first = str(period[0])
        if len(first) >= 4 and first[:4].isdigit():
            requested_year = int(first[:4])

    if req_form == "latest":
        return {
            **base,
            "requested_year": max_valid_year,
            "max_valid_year": max_valid_year,
            "resolved_period": [f"{max_valid_year:04d}-01-01", f"{max_valid_year:04d}-12-31"],
        }

    if requested_year is None:
        return {
            **base,
            "max_valid_year": max_valid_year,
        }

    if requested_year > max_valid_year:
        return {
            **base,
            "is_valid": False,
            "requested_year": requested_year,
            "max_valid_year": max_valid_year,
            "message": f"El pib para el año {requested_year} no esta disponible aun",
        }

    return {
        **base,
        "requested_year": requested_year,
        "max_valid_year": max_valid_year,
        "resolved_period": [f"{requested_year:04d}-01-01", f"{requested_year:04d}-12-31"],
    }
