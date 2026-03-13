"""Business rules for PIB/IMACEC queries.

Siempre se consultará por datos y no por consultas metodológicas.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
import unicodedata
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

PIB_CURRENT_PRICE_PATTERN = re.compile(
    r"\b(precios?\s+corrientes?|corriente|nominal|en\s+pesos?|asciende)\b",
    re.IGNORECASE,
)
PIB_CUANTO_ES_PATTERN = re.compile(
    r"\b(?:cuanto|a\s+cuanto|de\s+cuanto)\s+"
    r"(?:es|era|fue|vale)\s+"
    r"(?:el\s+)?(?:pib|producto\s+interno\s+bruto)\b"
    r"(?:\s+(?:de\s+chile|chileno))?",
    re.IGNORECASE,
)
PIB_INDICATOR_PATTERN = re.compile(
    r"\b(?:pib|producto\s+interno\s+bruto)\b",
    re.IGNORECASE,
)
PIB_SHARE_PATTERN = re.compile(
    r"(\bparticipacion\b|\bcuanto\s+pesa\b|\bpesa\b|\bque\s+porcentaje\b|\bporcentaje\b)",
    re.IGNORECASE,
)
PIB_SHARE_GASTO_PATTERN = re.compile(
    r"\b(consumo|inversion|exportaciones?|importaciones?|demanda\s+interna)\b",
    re.IGNORECASE,
)
PIB_PER_CAPITA_PATTERN = re.compile(
    r"\b(per\s*capita|por\s+persona)\b",
    re.IGNORECASE,
)

PIB_PER_CAPITA_UNAVAILABLE_MESSAGE = (
    "No tengo una serie de PIB per capita habilitada en el catalogo actual del chatbot. "
    "Si quieres, puedo entregarte el PIB total (a precios corrientes o encadenados) "
    "para el periodo que te interese."
)

PIB_SHARE_GASTO_UNAVAILABLE_MESSAGE = (
    "La participacion del PIB por componentes del gasto requiere revisar las series "
    "publicadas. Las series las puedes revisar en el siguiente cuadro segun los datos "
    "de la BDE publicados por el Banco Central de Chile: "
    "https://si3.bcentral.cl/Siete/ES/Siete/Cuadro/CAP_CCNN/MN_CCNN76/CCNN_EP18_03_ratio"
)

RULE_TOGGLES_PATH_ENV = "PIBOT_RULE_TOGGLES_PATH"
RULE_TOGGLES_FILENAME = "rule_toggles.json"

_RULE_TOGGLE_KEY_ALIASES: Dict[str, str] = {
    "pib_corrientes": "ENABLE_RULE_PIB_CORRIENTES",
    "pib_share": "ENABLE_RULE_PIB_SHARE",
    "pib_per_capita": "ENABLE_RULE_PIB_PER_CAPITA",
    "pib_share_gasto_guardrail": "ENABLE_RULE_PIB_SHARE_GASTO_GUARDRAIL",
}


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


def _env_flag(name: str, default: bool) -> bool:
    # 1) Prioridad: variable de entorno explícita.
    raw = os.getenv(name)
    if raw is None:
        # 2) Fallback: archivo JSON de toggles en rules/.
        toggles = _load_rule_toggles()
        if name in toggles:
            return toggles[name]
        # 3) Fallback final: valor por defecto del código.
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "si"}


def _resolve_rule_toggles_path() -> Path:
    raw = str(os.getenv(RULE_TOGGLES_PATH_ENV, "")).strip()
    if raw:
        return Path(raw)
    return Path(__file__).resolve().parent / RULE_TOGGLES_FILENAME


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "si"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _load_rule_toggles() -> Dict[str, bool]:
    path = _resolve_rule_toggles_path()
    if not path.exists() or not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    # Permite dos formatos:
    # - {"rules": {"pib_corrientes": true, ...}}
    # - {"ENABLE_RULE_PIB_CORRIENTES": true, ...}
    source = payload.get("rules") if isinstance(payload.get("rules"), dict) else payload
    if not isinstance(source, dict):
        return {}

    result: Dict[str, bool] = {}
    for raw_key, raw_value in source.items():
        key = str(raw_key).strip()
        if not key:
            continue
        parsed = _coerce_bool(raw_value)
        if parsed is None:
            continue
        target_key = _RULE_TOGGLE_KEY_ALIASES.get(key, key)
        result[target_key] = parsed

    return result


def _normalize_for_matching(text: str) -> str:
    lowered = str(text or "").lower()
    no_accents = "".join(
        ch
        for ch in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"\s+", " ", no_accents).strip()


def resolve_data_node_overrides(
    *,
    question: str,
    indicator_ent: Any,
    calc_mode_cls: Any,
    req_form_cls: Any,
    frequency_ent: Any,
) -> Dict[str, Any]:
    """Resuelve overrides de negocio para el nodo DATA.

    Feature flags (env):
    - ``ENABLE_RULE_PIB_CORRIENTES`` (default: true)
    - ``ENABLE_RULE_PIB_SHARE`` (default: true)
    - ``ENABLE_RULE_PIB_PER_CAPITA`` (default: true)
    """

    indicator = str(indicator_ent or "").strip().lower()
    normalized_q = _normalize_for_matching(question)
    mentions_pib = bool(PIB_INDICATOR_PATTERN.search(normalized_q))
    if indicator != "pib" and not mentions_pib:
        return {}

    result: Dict[str, Any] = {}
    if indicator != "pib" and mentions_pib:
        # Respaldo cuando el clasificador no logró fijar indicador.
        result["indicator_ent"] = "pib"

    enable_corrientes = _env_flag("ENABLE_RULE_PIB_CORRIENTES", True)
    enable_share = _env_flag("ENABLE_RULE_PIB_SHARE", True)
    enable_share_gasto_guardrail = _env_flag("ENABLE_RULE_PIB_SHARE_GASTO_GUARDRAIL", True)
    enable_per_capita = _env_flag("ENABLE_RULE_PIB_PER_CAPITA", True)

    if enable_per_capita and PIB_PER_CAPITA_PATTERN.search(normalized_q):
        # Enruta a la serie anual de PIB per capita (USD) sin activar guardrail.
        result["activity_ent"] = "pib_percapita"
        result["activity_cls"] = "specific"
        result["activity_cls_resolved"] = "specific"
        # En conversacion multi-turno pueden venir arrastres de region/inversion.
        # Para PIB per capita se fuerza el agregado nacional sin desglose.
        result["region_ent"] = None
        result["region_cls"] = "none"
        result["investment_ent"] = None
        result["investment_cls"] = "none"
        result["frequency_ent"] = "a"
        result["seasonality_ent"] = "nsa"
        result["price"] = None
        result["feature"] = "pib_per_capita"
        return result

    if enable_share and PIB_SHARE_PATTERN.search(normalized_q):
        if enable_share_gasto_guardrail and PIB_SHARE_GASTO_PATTERN.search(normalized_q):
            result["short_circuit_message"] = PIB_SHARE_GASTO_UNAVAILABLE_MESSAGE
            result["feature"] = "pib_share_gasto_unavailable"
            return result
        result["calc_mode_cls"] = "share"
        # Las familias de participacion de PIB disponibles en catalogo son anuales.
        result["frequency_ent"] = "a"
        result["price"] = "co"
        result["feature"] = "pib_share"

    if enable_corrientes and PIB_CURRENT_PRICE_PATTERN.search(normalized_q):
        result["price"] = "co"
        if "feature" not in result:
            result["feature"] = "pib_corrientes"
    elif enable_corrientes and PIB_CUANTO_ES_PATTERN.search(normalized_q):
        result["price"] = "co"
        if "feature" not in result:
            result["feature"] = "pib_corrientes_cuanto_es"

    # Permite ajustes futuros por req_form/frequency si se necesita granularidad extra.
    _ = str(req_form_cls or "").strip().lower()
    _ = str(frequency_ent or "").strip().lower()

    return result
