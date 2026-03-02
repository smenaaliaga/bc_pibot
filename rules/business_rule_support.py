"""Support business rules for PIB/IMACEC queries.

Este módulo contiene utilidades de reglas no relacionadas al enrutamiento
final de tipo de respuesta.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_METADATA_CACHE: Dict[str, Any] = {}
_RULES_JSON_CACHE: Dict[str, Dict[str, Any]] = {}

_RULES_DIR = Path(__file__).resolve().parent
_HEADER_RULES_PATH = _RULES_DIR / "header_rules.json"
_RESPONSE_RULES_PATH = _RULES_DIR / "response_rules.json"

_DEFAULT_HEADER_RULES: Dict[str, Any] = {
    "label_flags": {
        "seasonality": False,
        "price": False,
        "history": False,
    },
    "price_rules": {
        "co": r"\b(precio|precios)\s+corrientes?\b|\bcorrientes?\b",
        "en": r"\b(precio|precios)\s+encadenados?\b|\bencadenados?\b",
    },
    "seasonality_rules": {
        "sa": r"desestacionalizad[oa]s?|ajustad[oa]s?\s+estacional",
        "nsa": r"\b(sin\s+desestacionalizar|original)\b",
    },
    "history_rules": {
        "inf_historic": r"\bhist[óo]ric[oa]s?\b|\bantes\s+de\s+1996\b",
    },
    "calc_mode_aliases": {
        "yoy": ["yoy", "ypct", "interanual", "annual", "anual"],
        "prev_period": ["prev_period", "pct", "mom", "qoq", "period_previous", "periodo_anterior"],
        "original": ["original", "value", "level", "valor"],
    },
    "calc_mode_patterns": {
        "interanual": r"\b(interanual|a/a|anual\s+anterior|mismo\s+periodo\s+del\s+ano\s+anterior|mismo\s+per[ií]odo\s+del\s+a[nñ]o\s+anterior)\b",
        "prev_period": r"\b(periodo\s+anterior|per[ií]odo\s+anterior|trimestre\s+anterior|mes\s+anterior|qoq|mom|t/t|m/m|c/r\s+al\s+periodo\s+anterior|c/r\s+al\s+per[ií]odo\s+anterior)\b",
    },
}

_DEFAULT_RESPONSE_RULES: Dict[str, Any] = {
    "response_general": {
        "title": "Regla 1: response_general",
        "description": "Responder con cuadro general cuando no existe serie válida.",
        "conditions": {"has_series": False},
        "result": "response_general",
    },
    "response_specific": {
        "title": "Regla 2: response_specific",
        "description": "Responder con salida específica para latest/range/point con serie válida.",
        "conditions": {"has_series": True, "req_form_in": ["latest", "range", "point"]},
        "result": "response_specific",
    },
    "response_specific_point": {
        "title": "Regla 3: response_specific_point",
        "description": "Responder con salida específica de punto para specific_point.",
        "conditions": {"has_series": True, "req_form_in": ["specific_point"]},
        "result": "response_specific_point",
    },
}

YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _load_rules_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    cache_key = str(path)
    if cache_key in _RULES_JSON_CACHE:
        return _RULES_JSON_CACHE[cache_key]

    if not path.exists():
        _RULES_JSON_CACHE[cache_key] = dict(default)
        return _RULES_JSON_CACHE[cache_key]

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            loaded = {}
    except Exception:
        loaded = {}

    merged = dict(default)
    merged.update(loaded)
    _RULES_JSON_CACHE[cache_key] = merged
    return merged


def _compile_patterns(patterns: Dict[str, str]) -> Dict[str, re.Pattern[str]]:
    return {
        label: re.compile(str(pattern), re.IGNORECASE)
        for label, pattern in (patterns or {}).items()
        if str(pattern).strip()
    }


def _header_rules_config() -> Dict[str, Any]:
    return _load_rules_json(_HEADER_RULES_PATH, _DEFAULT_HEADER_RULES)


def _response_rules_config() -> Dict[str, Any]:
    return _load_rules_json(_RESPONSE_RULES_PATH, _DEFAULT_RESPONSE_RULES)


def _label_flags() -> Dict[str, bool]:
    config = _header_rules_config()
    flags = config.get("label_flags") or {}
    return {
        "seasonality": bool(flags.get("seasonality", False)),
        "price": bool(flags.get("price", False)),
        "history": bool(flags.get("history", False)),
    }


def _seasonality_patterns() -> Dict[str, re.Pattern[str]]:
    config = _header_rules_config()
    return _compile_patterns(config.get("seasonality_rules") or {})


def _price_patterns() -> Dict[str, re.Pattern[str]]:
    config = _header_rules_config()
    return _compile_patterns(config.get("price_rules") or {})


def _history_patterns() -> Dict[str, re.Pattern[str]]:
    config = _header_rules_config()
    return _compile_patterns(config.get("history_rules") or {})


def _calc_mode_aliases() -> Dict[str, List[str]]:
    config = _header_rules_config()
    aliases = config.get("calc_mode_aliases") or {}
    normalized: Dict[str, List[str]] = {}
    for canonical, values in aliases.items():
        if isinstance(values, list):
            normalized[str(canonical)] = [str(item).strip().lower() for item in values if str(item).strip()]
    return normalized


def _calc_mode_patterns() -> Dict[str, re.Pattern[str]]:
    config = _header_rules_config()
    return _compile_patterns(config.get("calc_mode_patterns") or {})


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
    for label, pattern in _seasonality_patterns().items():
        if pattern.search(question):
            return label
    return None


def classify_price(question: str) -> Optional[str]:
    for label, pattern in _price_patterns().items():
        if pattern.search(question):
            return label
    return "co"


def classify_history(question: str, indicator: Optional[str]) -> str:
    if indicator and indicator.lower() == "pib":
        history_rules = _history_patterns()
        inf_historic = history_rules.get("inf_historic")
        if inf_historic and inf_historic.search(question):
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
    """Normalize calc_mode to align business rules and response generation."""

    raw_mode = str(calc_mode_cls or "").strip().lower()
    intent = str(intent_cls or "").strip().lower()
    req_form = str(req_form_cls or "").strip().lower()
    freq = str(frequency or "").strip().lower()
    question_text = str(question or "")

    def _normalize_alias(mode: str) -> Optional[str]:
        if not mode or mode in {"none", "null", "nan"}:
            return None

        aliases = _calc_mode_aliases()
        for canonical, values in aliases.items():
            if mode == canonical or mode in values:
                return canonical
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

    calc_mode_patterns = _calc_mode_patterns()
    interanual_pattern = calc_mode_patterns.get("interanual")
    prev_period_pattern = calc_mode_patterns.get("prev_period")

    if interanual_pattern and interanual_pattern.search(question_text):
        return "yoy"
    if prev_period_pattern and prev_period_pattern.search(question_text):
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
    flags = _label_flags()
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
    """Validate if requested PIB annual year is available given latest_update."""

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


__all__ = [
    "_response_rules_config",
    "apply_latest_update_period",
    "benchmark",
    "build_metadata_key",
    "build_metadata_response",
    "classify_by_regex",
    "classify_headers",
    "classify_history",
    "classify_price",
    "classify_seasonality",
    "resolve_calc_mode_cls",
    "resolve_pib_annual_validity",
    "resolve_region_value",
]
