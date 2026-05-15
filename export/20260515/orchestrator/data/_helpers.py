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
    date_direction: Optional[str] = None,
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

    def _extract_row_year(row: Dict[str, Any]) -> Optional[str]:
        return _extract_year_local(row.get("date") or row.get("period"))

    period_values = period or []
    requested_start_year = _extract_year_local(period_values[0]) if period_values else None
    requested_end_year = _extract_year_local(period_values[-1]) if period_values else None
    requested_calc_mode = str(calc_mode or "").strip().lower()
    is_contribution_link = requested_calc_mode == "contribution"
    req = str(req_form or "").strip().lower()
    observed_rows = [row for row in (observations or []) if isinstance(row, dict)]
    observed_years = [
        int(year)
        for year in (_extract_row_year(row) for row in observed_rows)
        if year is not None
    ]
    observed_start_year_num = min(observed_years) if observed_years else None
    observed_end_year_num = max(observed_years) if observed_years else None
    observed_start_year = (
        str(observed_start_year_num) if observed_start_year_num is not None else None
    )
    observed_end_year = (
        str(observed_end_year_num) if observed_end_year_num is not None else None
    )

    # date_direction='earliest' → la pregunta es por el primer/más antiguo dato.
    # El enlace al cuadro debe abrir el año más antiguo observado, no años
    # recientes que es lo que produce la lógica por defecto (que prioriza el
    # período pedido o el último observado). Se hace antes de los cálculos
    # subsiguientes para que toda la resolución de start/end use ese ancla.
    is_earliest_request = str(date_direction or "").strip().lower() == "earliest"
    if is_earliest_request and observed_start_year:
        requested_start_year = observed_start_year
        requested_end_year = observed_start_year

    use_observed_end = req == "latest" and not is_contribution_link

    end_year = (
        observed_end_year if use_observed_end and observed_end_year else requested_end_year
    )
    start_year = requested_start_year
    if start_year is None and end_year is not None:
        try:
            start_year = str(int(end_year) - 10)
        except Exception:
            start_year = None

    # Solo para contribution en URL: si inicio y término quedan iguales,
    # abrir el término al 2025 para mostrar ventana útil de datos.
    if is_contribution_link and start_year and end_year and start_year == end_year:
        end_year = "2025"

    # Solo para URL de referencia: asegurar rango válido en el cuadro.
    # Si fecha inicio > fecha término, forzar término a 2025.
    if start_year and end_year:
        try:
            start_num = int(start_year)
            end_num = int(end_year)
            if start_num > end_num:
                end_num = 2025
                if start_num > end_num:
                    start_num = 2015
                start_year = str(start_num)
                end_year = str(end_num)
        except Exception:
            pass

    frequency_param = {
        "a": "ANNUAL",
        "q": "QUARTERLY",
        "m": "MONTHLY",
    }.get(str(frequency or "").strip().lower())

    # Regla para URL de referencia:
    # - PCT solo cuando se pidió explícitamente prev_period.
    # - NONE solo cuando se pide explícitamente none (o se mapea upstream
    #   para casos especiales como precios corrientes / PIB per cápita).
    # - Todo lo demás, incluido original, cae a YTYPCT por defecto.
    if requested_calc_mode == "prev_period":
        resolved_calc_mode = "prev_period"
        calc_param = "PCT"
    elif is_contribution_link:
        resolved_calc_mode = "none"
        calc_param = "NONE"
    elif requested_calc_mode == "none":
        resolved_calc_mode = "none"
        calc_param = "NONE"
    else:
        resolved_calc_mode = "yoy"
        calc_param = "YTYPCT"

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
        if not _has_requested_calc_value(observations, resolved_calc_mode):
            calc_param = "NONE"

    def _row_has_requested_calc_value(row: Dict[str, Any], mode: str) -> bool:
        if mode == "yoy":
            keys = ("yoy", "yoy_pct")
        elif mode == "prev_period":
            keys = ("prev_period", "pct")
        else:
            return False
        return any(row.get(key) is not None for key in keys)

    # Para referencias URL en consultas fuera de rango (o con cálculo no disponible
    # en el año pedido), anclar al último período observable de la serie para evitar
    # enlaces que abran años sin dato útil en el cuadro BDE.
    # Excepción: cuando la consulta es por el dato más antiguo (date_direction='earliest'),
    # el ancla ya quedó fijada al observed_start_year y NO debe desplazarse al
    # último año aunque ese punto no tenga variación interanual.
    if (
        not is_contribution_link
        and not is_earliest_request
        and observed_start_year_num is not None
        and observed_end_year_num is not None
        and start_year
        and end_year
    ):
        try:
            start_num = int(start_year)
            end_num = int(end_year)

            latest_calc_year_num = observed_end_year_num
            if resolved_calc_mode in {"yoy", "prev_period"}:
                calc_years = [
                    int(year)
                    for year in (
                        _extract_row_year(row)
                        for row in observed_rows
                        if _row_has_requested_calc_value(row, resolved_calc_mode)
                    )
                    if year is not None
                ]
                if calc_years:
                    latest_calc_year_num = max(calc_years)

            no_overlap = end_num < observed_start_year_num or start_num > observed_end_year_num
            if no_overlap:
                start_num = latest_calc_year_num
                end_num = latest_calc_year_num

            # Caso típico de PIB 1960 con YTYPCT: año válido, pero sin variación
            # interanual en el primer punto; URL debe apuntar al último dato con cálculo.
            if (
                resolved_calc_mode in {"yoy", "prev_period"}
                and start_num == end_num
            ):
                has_calc_in_requested_year = any(
                    _extract_row_year(row) == str(start_num)
                    and _row_has_requested_calc_value(row, resolved_calc_mode)
                    for row in observed_rows
                )
                if not has_calc_in_requested_year:
                    start_num = latest_calc_year_num
                    end_num = latest_calc_year_num

            start_year = str(start_num)
            end_year = str(end_num)
        except Exception:
            pass

    separator = "&" if "?" in str(source_url) else "?"
    query_parts: List[str] = []
    if not is_contribution_link:
        query_parts.extend([f"id5=SI", f"idSerie={series_id}"])
    if start_year:
        query_parts.append(f"cbFechaInicio={start_year}")
    if end_year:
        query_parts.append(f"cbFechaTermino={end_year}")
    if frequency_param:
        query_parts.append(f"cbFrecuencia={frequency_param}")
    if calc_param:
        query_parts.append(f"cbCalculo={calc_param}")
    if is_contribution_link:
        query_parts.append("cbFechaBase=")

    return f"{source_url}{separator}{'&'.join(query_parts)}"
