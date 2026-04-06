"""Funciones utilitarias puras para el nodo de datos del grafo PIBot.

Contiene helpers de parsing, coerción y construcción de URLs que no
dependen de estado ni de servicios externos.
"""

from __future__ import annotations

import calendar
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Coerción de valores
# ---------------------------------------------------------------------------

def first_non_empty(value: Any) -> Any:
    """Retorna el primer elemento que no sea vacío/nulo de una lista,
    o el propio valor si no es una lista y no está vacío."""
    if isinstance(value, list):
        for item in value:
            if item not in (None, "", [], {}, ()):
                return item
        return None
    if value in (None, "", [], {}, ()):
        return None
    return value


def coerce_period(period_value: Any) -> List[Any]:
    """Normaliza el valor de periodo a una lista."""
    if period_value in (None, "", [], {}, ()):
        return []
    if isinstance(period_value, list):
        return period_value
    return [period_value]


def extract_year(value: Any) -> Optional[int]:
    """Extrae el primer año (19xx/20xx) de un texto."""
    match = re.search(r"(19|20)\d{2}", str(value or "").strip())
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Conversión a fin de periodo
# ---------------------------------------------------------------------------

def to_period_end_str(date_str: Optional[str], freq: Optional[str]) -> Optional[str]:
    """Convierte una fecha ISO de inicio-de-periodo al último día del periodo.

    >>> to_period_end_str("2026-01-01", "M")
    '2026-01-31'
    """
    if not date_str or not freq:
        return date_str
    try:
        parts = str(date_str).strip()[:10].split("-")
        if len(parts) != 3:
            return date_str
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        freq_up = freq.upper()
        if freq_up == "M":
            _, last_day = calendar.monthrange(year, month)
            return f"{year:04d}-{month:02d}-{last_day:02d}"
        if freq_up in ("Q", "T"):
            quarter_end_month = ((month - 1) // 3 + 1) * 3
            _, last_day = calendar.monthrange(year, quarter_end_month)
            return f"{year:04d}-{quarter_end_month:02d}-{last_day:02d}"
        if freq_up == "A":
            return f"{year:04d}-12-31"
        return date_str
    except Exception:
        return date_str


# ---------------------------------------------------------------------------
# Parsing de fechas ISO
# ---------------------------------------------------------------------------

def parse_iso_date(value: Any) -> Optional[Tuple[int, int, int]]:
    """Parsea una fecha ISO (YYYY-MM-DD) y retorna (año, mes, día)."""
    date_text = str(value or "").strip()
    if not date_text:
        return None
    try:
        parts = date_text[:10].split("-")
        if len(parts) != 3:
            return None
        return int(parts[0]), int(parts[1]), int(parts[2])
    except Exception:
        return None


def quarter_from_date(value: Any) -> Optional[Tuple[int, int]]:
    """Extrae (año, trimestre) de una fecha ISO."""
    date_text = str(value or "").strip()
    if not date_text:
        return None
    try:
        parts = date_text[:10].split("-")
        if len(parts) != 3:
            return None
        year = int(parts[0])
        month = int(parts[1])
        if month < 1 or month > 12:
            return None
        quarter = ((month - 1) // 3) + 1
        return year, quarter
    except Exception:
        return None


def sort_observations_by_date_desc(
    observations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ordena observaciones por fecha descendente."""
    valid_rows = [row for row in (observations or []) if isinstance(row, dict)]
    return sorted(
        valid_rows,
        key=lambda row: parse_iso_date(row.get("date")) or (0, 0, 0),
        reverse=True,
    )


def same_requested_period(
    requested_date: Optional[str],
    observed_date: Optional[str],
    frequency: Optional[str],
) -> bool:
    """Compara si la fecha solicitada y observada corresponden al mismo periodo
    según la frecuencia (anual, trimestral o mensual)."""
    requested_parts = parse_iso_date(requested_date)
    observed_parts = parse_iso_date(observed_date)
    if requested_parts is None or observed_parts is None:
        return False

    req_year, req_month, _ = requested_parts
    obs_year, obs_month, _ = observed_parts
    freq_norm = str(frequency or "").strip().lower()

    if freq_norm in {"a", "annual", "anual"}:
        return req_year == obs_year
    if freq_norm in {"q", "t", "quarterly", "trimestral"}:
        req_quarter = ((req_month - 1) // 3) + 1
        obs_quarter = ((obs_month - 1) // 3) + 1
        return req_year == obs_year and req_quarter == obs_quarter
    return req_year == obs_year and req_month == obs_month


def has_full_quarterly_year(
    observations: List[Dict[str, Any]], year: int
) -> bool:
    """Verifica si existen los 4 trimestres de un año en las observaciones."""
    quarters: set[int] = set()
    for row in observations or []:
        if not isinstance(row, dict):
            continue
        qk = quarter_from_date(row.get("date"))
        if qk is None:
            continue
        row_year, row_quarter = qk
        if row_year == year:
            quarters.add(row_quarter)
    return len(quarters) == 4


def latest_annual_observation_before_year(
    observations: List[Dict[str, Any]],
    year_limit: int,
) -> Optional[Dict[str, Any]]:
    """Retorna la observación anual más reciente anterior a *year_limit*."""
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for row in observations or []:
        if not isinstance(row, dict):
            continue
        date_text = str(row.get("date") or "").strip()
        row_year = extract_year(date_text)
        if row_year is None or row_year >= year_limit:
            continue
        candidates.append((date_text, row))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------------
# Construcción de URL del explorador de series
# ---------------------------------------------------------------------------

def build_target_series_url(
    *,
    source_url: Optional[str],
    series_id: Optional[str],
    period: Optional[List[Any]],
    req_form: Optional[str] = None,
    observations: Optional[List[Dict[str, Any]]] = None,
    frequency: Optional[str] = None,
    calc_mode: Optional[str] = None,
) -> Optional[str]:
    """Construye la URL de consulta para el explorador de series del Banco Central.

    Parámetros clave:
      - *source_url*: base URL de la familia de series.
      - *series_id*: identificador de la serie.
      - *period*: lista con fechas de inicio/fin.
      - *observations*: datos observados (se usan para ajustar el rango).
    """
    if not source_url or not series_id:
        return None

    def _extract_year_local(value: Any) -> Optional[str]:
        match = re.search(r"(19|20)\d{2}", str(value or "").strip())
        return match.group(0) if match else None

    period_values = period or []
    requested_end_year = _extract_year_local(period_values[-1]) if period_values else None
    req = str(req_form or "").strip().lower()
    observed_rows = [
        row for row in (observations or [])
        if isinstance(row, dict) and row.get("date")
    ]
    observed_end_year = (
        _extract_year_local(observed_rows[-1].get("date")) if observed_rows else None
    )

    use_observed_end = req == "latest"
    if requested_end_year and observed_end_year and requested_end_year != observed_end_year:
        use_observed_end = True

    end_year = (
        observed_end_year if use_observed_end and observed_end_year else requested_end_year
    )
    start_year = None
    if end_year is not None:
        try:
            start_year = str(int(end_year) - 10)
        except Exception:
            start_year = None

    frequency_param = {
        "a": "ANNUAL",
        "q": "QUARTERLY",
        "m": "MONTHLY",
    }.get(str(frequency or "").strip().lower())

    requested_calc_mode = str(calc_mode or "").strip().lower()
    if requested_calc_mode == "original":
        requested_calc_mode = "prev_period"

    calc_param = {
        "yoy": "YTYPCT",
        "prev_period": "PCT",
        "none": "NONE",
    }.get(requested_calc_mode)

    def _has_requested_calc_value(rows: List[Dict[str, Any]], mode: str) -> bool:
        if mode == "yoy":
            candidate_keys = ("yoy", "yoy_pct")
        elif mode == "prev_period":
            candidate_keys = ("prev_period", "pct")
        else:
            return False
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in candidate_keys:
                if row.get(key) is not None:
                    return True
        return False

    if calc_param and observations is not None:
        if not _has_requested_calc_value(observations, requested_calc_mode):
            calc_param = "NONE"

    separator = "&" if "?" in str(source_url) else "?"
    query_parts = [f"id5=SI", f"idSerie={series_id}"]
    if start_year:
        query_parts.append(f"cbFechaInicio={start_year}")
    if end_year:
        query_parts.append(f"cbFechaTermino={end_year}")
    if frequency_param:
        query_parts.append(f"cbFrecuencia={frequency_param}")
    if calc_param:
        query_parts.append(f"cbCalculo={calc_param}")

    return f"{source_url}{separator}{'&'.join(query_parts)}"
