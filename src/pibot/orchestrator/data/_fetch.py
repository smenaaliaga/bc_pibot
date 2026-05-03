"""Funciones de obtención de datos para cada rama de cálculo del nodo DATA.

Tres modos principales de consulta:
  1. **Contribución** (``calc_mode == "contribution"``): recorre todas las series
     de la familia y alinea las fechas de referencia.
  2. **Desglose general sin contribución**: similar a contribución pero con
     variación (yoy / prev_period) sobre cada componente.
  3. **Serie específica / sin desglose**: consulta una sola serie y, para PIB
     anual, valida la completitud trimestral del año solicitado.

Las funciones ``load_series_observations`` y ``fetch_series_by_req_form`` se
definen aquí como implementaciones reales, pero las funciones de alto nivel
(``fetch_contribution``, ``fetch_general_breakdown``, ``fetch_specific_series``)
reciben un parámetro ``fetch_fn`` para que ``data.py`` inyecte la referencia
que los tests pueden monkeypatchar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._helpers import (
    build_target_series_url,
    extract_year,
    has_full_quarterly_year,
    latest_annual_observation_before_year,
    same_requested_period,
)

logger = logging.getLogger(__name__)

# Tipo de las funciones de fetch/load que se inyectan desde data.py
FetchFn = Callable[..., Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]
LoadFn = Callable[..., Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]]


# ---------------------------------------------------------------------------
# Carga de observaciones (wrapper liviano del servicio Redis)
# ---------------------------------------------------------------------------

def load_series_observations(
    *,
    series_id: Optional[str],
    firstdate: Optional[str],
    lastdate: Optional[str],
    target_frequency: Optional[str],
    agg_mode: str,
    calc_mode: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Obtiene observaciones de una serie desde Redis y adapta *value*/*selected*
    según el modo de cálculo (prev_period / yoy)."""
    from orchestrator.data.get_data_serie import get_series_from_redis

    series_data = get_series_from_redis(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=target_frequency,
        agg=agg_mode,
        use_fallback=True,
    )

    metadata = (
        (series_data or {}).get("meta") if isinstance(series_data, dict) else None
    )
    observations = [
        obs
        for obs in ((series_data or {}).get("observations") or [])
        if isinstance(obs, dict)
    ]

    mode = str(calc_mode or "").strip().lower()
    if mode not in {"prev_period", "yoy"}:
        return observations, metadata

    adapted: List[Dict[str, Any]] = []
    for obs in observations:
        row = dict(obs)
        original_value = row.get("value")
        if mode == "prev_period":
            selected_value = row.get("pct")
            if selected_value is None:
                selected_value = row.get("prev_period")
            row["prev_period"] = selected_value
            row.pop("pct", None)
        else:
            selected_value = row.get("yoy_pct")
            if selected_value is None:
                selected_value = row.get("yoy")
            row["yoy"] = selected_value
            row.pop("yoy_pct", None)
        row["value"] = original_value
        row["selected"] = selected_value
        adapted.append(row)

    return adapted, metadata


def fetch_series_by_req_form(
    *,
    series_id: Optional[str],
    req_form: Optional[str],
    frequency: Optional[str],
    indicator: Optional[str],
    firstdate: Optional[str],
    lastdate: Optional[str],
    calc_mode: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Obtiene observaciones según el ``req_form`` (latest / point / range).

    Para *latest* no se limita el rango de fechas; para *point*/*range* se
    usan los extremos recibidos, con fallback sin rango si no hay datos.
    """
    req = str(req_form or "").strip().lower()
    target_frequency = str(frequency or "").upper() or None
    agg_mode = "sum"

    if req == "latest":
        observations, metadata = load_series_observations(
            series_id=series_id,
            firstdate=None,
            lastdate=None,
            target_frequency=target_frequency,
            agg_mode=agg_mode,
            calc_mode=calc_mode,
        )
        latest_obs = observations[-1] if observations else None
        return observations, latest_obs, metadata

    observations, metadata = load_series_observations(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=target_frequency,
        agg_mode=agg_mode,
        calc_mode=calc_mode,
    )
    latest_obs = observations[-1] if observations else None

    if req in {"point", "range"} and not observations:
        fallback_obs, _ = load_series_observations(
            series_id=series_id,
            firstdate=None,
            lastdate=None,
            target_frequency=target_frequency,
            agg_mode=agg_mode,
            calc_mode=calc_mode,
        )
        if fallback_obs:
            latest_obs = fallback_obs[-1]

    return observations, latest_obs, metadata


# ---------------------------------------------------------------------------
# Resultado unificado de las ramas de fetch
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    """Datos obtenidos por cualquiera de las tres ramas de fetch."""

    observations: List[Dict[str, Any]] = field(default_factory=list)
    observations_all: List[Dict[str, Any]] = field(default_factory=list)
    latest_obs: Optional[Dict[str, Any]] = None
    observations_meta: Optional[Dict[str, Any]] = None
    used_latest_fallback_for_point: bool = False
    req_form_for_payload: Optional[str] = None
    out_of_range_lt_first: bool = False
    incomplete_frequency_note: Optional[str] = None
    effective_reference_period: Optional[str] = None
    # Pueden actualizarse por la rama de contribución
    target_series_id: Optional[str] = None
    target_series_title: Optional[str] = None
    target_series_url: Optional[str] = None
    period_ent: Optional[List[Any]] = None
    period_values: Optional[List[Any]] = None


# ---------------------------------------------------------------------------
# Alineación de fecha de referencia (lógica compartida entre contribution
# y general breakdown)
# ---------------------------------------------------------------------------

def _align_reference_date(
    *,
    fetch_fn: FetchFn,
    target_series_id: Optional[str],
    family_series: List[Dict[str, Any]],
    req_form_cls: Any,
    frequency_ent: Optional[str],
    indicator_ent: Optional[str],
    firstdate: Optional[str],
    lastdate: Optional[str],
    calc_mode: Optional[str] = None,
) -> Tuple[
    Optional[str],                  # target_reference_date
    Optional[Dict[str, Any]],       # observations_meta
    List[Dict[str, Any]],           # target_observations
    Optional[Dict[str, Any]],       # target_latest_obs
]:
    """Obtiene datos de la serie objetivo y alinea la fecha de referencia
    con la fecha más reciente disponible entre los componentes de la familia."""
    req_form_norm = str(req_form_cls or "").strip().lower()
    target_reference_date: Optional[str] = None

    if not target_series_id:
        return None, None, [], None

    target_observations, target_latest_obs, observations_meta = fetch_fn(
        series_id=target_series_id,
        req_form=req_form_cls,
        frequency=frequency_ent,
        indicator=indicator_ent,
        firstdate=firstdate,
        lastdate=lastdate,
        calc_mode=calc_mode,
    )

    if target_observations:
        target_reference_date = (
            str(target_observations[-1].get("date") or "").strip() or None
        )
    elif isinstance(target_latest_obs, dict):
        target_reference_date = (
            str(target_latest_obs.get("date") or "").strip() or None
        )

    # Para "latest", alinear con el máximo de los componentes
    if req_form_norm == "latest" and target_reference_date:
        component_dates: List[str] = []
        for series in family_series:
            sid = series.get("id") if isinstance(series, dict) else None
            if not sid or sid == target_series_id:
                continue
            comp_obs, comp_latest, _ = fetch_fn(
                series_id=sid,
                req_form=req_form_cls,
                frequency=frequency_ent,
                indicator=indicator_ent,
                firstdate=firstdate,
                lastdate=lastdate,
                calc_mode=calc_mode,
            )
            comp_date = None
            if comp_obs:
                comp_date = str(comp_obs[-1].get("date") or "").strip() or None
            elif isinstance(comp_latest, dict):
                comp_date = str(comp_latest.get("date") or "").strip() or None
            if comp_date:
                component_dates.append(comp_date)

        if component_dates and target_reference_date not in component_dates:
            aligned = max(component_dates)
            logger.info(
                "[DATA_NODE] aligning target date from %s to common component date %s",
                target_reference_date,
                aligned,
            )
            target_reference_date = aligned

    return target_reference_date, observations_meta, target_observations, target_latest_obs


# ---------------------------------------------------------------------------
# Rama 1: Contribución
# ---------------------------------------------------------------------------

def fetch_contribution(
    *,
    fetch_fn: FetchFn,
    family_series: List[Dict[str, Any]],
    target_series_id: Optional[str],
    target_series_title: Optional[str],
    source_family_series: Optional[str],
    family_name: Optional[str],
    req_form_cls: Any,
    frequency_ent: Optional[str],
    indicator_ent: Optional[str],
    period_ent: List[Any],
    period_values: List[Any],
    calc_mode_cls: Any,
    activity_cls_resolved: Any,
    activity_ent: Optional[str],
    region_cls: Any,
    investment_cls: Any,
) -> FetchResult:
    """Obtiene datos en modo contribución: agrega una observación por cada serie
    de la familia, alineando todas a una fecha de referencia común."""
    firstdate = str(period_values[0]) if period_values else None
    lastdate = str(period_values[-1]) if period_values else None
    res = FetchResult(
        req_form_for_payload=req_form_cls,
        effective_reference_period=str(period_values[-1]) if period_values else None,
        target_series_id=target_series_id,
        target_series_title=target_series_title,
    )
    req_form_norm = str(req_form_cls or "").strip().lower()
    requested_period_end = str(period_values[-1]) if period_values else None

    # Alinear fecha de referencia
    target_reference_date, _, _, _ = _align_reference_date(
        fetch_fn=fetch_fn,
        target_series_id=target_series_id,
        family_series=family_series,
        req_form_cls=req_form_cls,
        frequency_ent=frequency_ent,
        indicator_ent=indicator_ent,
        firstdate=firstdate,
        lastdate=lastdate,
    )

    if req_form_norm == "latest" and target_reference_date:
        res.effective_reference_period = target_reference_date

    if (
        req_form_norm in {"point", "specific_point", "range"}
        and requested_period_end
        and target_reference_date
        and not same_requested_period(
            requested_period_end, target_reference_date, frequency_ent
        )
    ):
        res.used_latest_fallback_for_point = True
        res.req_form_for_payload = "point"

    # Recorrer toda la familia y extraer una observación por serie
    for series in family_series:
        series_id = series.get("id")
        if not series_id:
            continue

        series_observations, series_latest_obs, series_meta = fetch_fn(
            series_id=series_id,
            req_form=req_form_cls,
            frequency=frequency_ent,
            indicator=indicator_ent,
            firstdate=firstdate,
            lastdate=lastdate,
        )

        if isinstance(series_meta, dict):
            if (
                str(series_meta.get("lastdate_position") or "").strip().lower()
                == "lt_first"
            ):
                res.out_of_range_lt_first = True

        selected_obs = _select_observation_at_reference(
            series_observations,
            series_latest_obs,
            target_reference_date,
            req_form_norm,
        )
        if not isinstance(selected_obs, dict):
            continue

        row_title = str(series.get("short_title") or series_id).strip()
        series_cls = (
            (series.get("classification") or {})
            if isinstance(series, dict)
            else {}
        )

        res.observations.append(
            {
                "series_id": series_id,
                "title": row_title,
                "activity": _normalize_cls_value(series_cls.get("activity")),
                "investment": _normalize_cls_value(series_cls.get("investment")),
                "date": selected_obs.get("date"),
                "value": selected_obs.get("value"),
            }
        )

    res.observations_all = list(res.observations)

    # --- Filtrado post-fetch para contribución específica o general ----------
    _filter_contribution_observations(
        res,
        family_series=family_series,
        target_series_id=target_series_id,
        source_family_series=source_family_series,
        family_name=family_name,
        activity_cls_resolved=activity_cls_resolved,
        activity_ent=activity_ent,
        region_cls=region_cls,
        investment_cls=investment_cls,
        period_ent=period_ent,
        req_form_cls=req_form_cls,
        frequency_ent=frequency_ent,
        calc_mode_cls=calc_mode_cls,
    )

    return res


def _filter_contribution_observations(
    res: FetchResult,
    *,
    family_series: List[Dict[str, Any]],
    target_series_id: Optional[str],
    source_family_series: Optional[str],
    family_name: Optional[str],
    activity_cls_resolved: Any,
    activity_ent: Optional[str],
    region_cls: Any,
    investment_cls: Any,
    period_ent: List[Any],
    req_form_cls: Any,
    frequency_ent: Optional[str],
    calc_mode_cls: Any,
) -> None:
    """Filtra observaciones de contribución según la dimensión solicitada."""
    activity_values = _distinct_dimension_values(family_series, "activity")
    region_values = _distinct_dimension_values(family_series, "region")
    investment_values = _distinct_dimension_values(family_series, "investment")

    is_specific = (
        (
            str(activity_cls_resolved or "").strip().lower() == "specific"
            and len(activity_values) > 1
        )
        or (
            str(region_cls or "").strip().lower() == "specific"
            and len(region_values) > 1
        )
        or (
            str(investment_cls or "").strip().lower() == "specific"
            and len(investment_values) > 1
        )
    )

    if is_specific:
        target_row = _find_row_by_series_id(res.observations, target_series_id)
        if isinstance(target_row, dict):
            res.observations = [target_row]
            res.observations_all = [target_row]
            res.target_series_title = str(
                target_row.get("title")
                or res.target_series_title
                or family_name
                or ""
            ).strip()

    if activity_cls_resolved == "general" and activity_ent is None:
        aggregate_row = _find_row_by_series_id(res.observations, target_series_id)
        if aggregate_row is None:
            for row in res.observations:
                title_norm = str(row.get("title") or "").strip().lower()
                if title_norm in {"pib", "imacec"}:
                    aggregate_row = row
                    break
                if aggregate_row is None and row.get("activity") in (
                    None,
                    "",
                    "total",
                ):
                    aggregate_row = row

        if isinstance(aggregate_row, dict):
            res.target_series_id = aggregate_row.get("series_id")
            res.target_series_title = str(
                aggregate_row.get("title") or family_name or ""
            ).strip()
            res.target_series_url = build_target_series_url(
                source_url=source_family_series,
                series_id=res.target_series_id,
                period=period_ent if isinstance(period_ent, list) else None,
                req_form=req_form_cls,
                observations=res.observations,
                frequency=frequency_ent,
                calc_mode=calc_mode_cls,
            )
            res.observations = [aggregate_row]


# ---------------------------------------------------------------------------
# Rama 2: Desglose general sin contribución
# ---------------------------------------------------------------------------

def fetch_general_breakdown(
    *,
    fetch_fn: FetchFn,
    family_series: List[Dict[str, Any]],
    target_series_id: Optional[str],
    target_series_title: Optional[str],
    source_family_series: Optional[str],
    family_name: Optional[str],
    req_form_cls: Any,
    frequency_ent: Optional[str],
    indicator_ent: Optional[str],
    period_values: List[Any],
    calc_mode_cls: Any,
    activity_cls: Any,
    region_cls: Any,
    investment_cls: Any,
) -> FetchResult:
    """Obtiene datos para desglose general con variación (yoy / prev_period)."""
    firstdate = str(period_values[0]) if period_values else None
    lastdate = str(period_values[-1]) if period_values else None
    calc_mode_norm = str(calc_mode_cls or "").strip().lower()
    breakdown_variation = (
        calc_mode_norm if calc_mode_norm in {"yoy", "prev_period"} else "yoy"
    )
    req_form_norm = str(req_form_cls or "").strip().lower()
    requested_period_end = str(period_values[-1]) if period_values else None

    res = FetchResult(
        req_form_for_payload=req_form_cls,
        effective_reference_period=str(period_values[-1]) if period_values else None,
        target_series_id=target_series_id,
        target_series_title=target_series_title,
    )

    # Alinear fecha de referencia
    target_reference_date, observations_meta, _, _ = _align_reference_date(
        fetch_fn=fetch_fn,
        target_series_id=target_series_id,
        family_series=family_series,
        req_form_cls=req_form_cls,
        frequency_ent=frequency_ent,
        indicator_ent=indicator_ent,
        firstdate=firstdate,
        lastdate=lastdate,
        calc_mode=breakdown_variation,
    )
    res.observations_meta = observations_meta

    if req_form_norm == "latest" and target_reference_date:
        res.effective_reference_period = target_reference_date

    if (
        req_form_norm in {"point", "specific_point", "range"}
        and requested_period_end
        and target_reference_date
        and not same_requested_period(
            requested_period_end, target_reference_date, frequency_ent
        )
    ):
        res.used_latest_fallback_for_point = True
        res.req_form_for_payload = "point"

    # Recorrer familia y obtener una observación con variación por serie
    for series in family_series:
        series_id = series.get("id") if isinstance(series, dict) else None
        if not series_id:
            continue

        series_observations, series_latest_obs, series_meta = fetch_fn(
            series_id=series_id,
            req_form=req_form_cls,
            frequency=frequency_ent,
            indicator=indicator_ent,
            firstdate=firstdate,
            lastdate=lastdate,
            calc_mode=breakdown_variation,
        )

        if isinstance(series_meta, dict):
            if (
                str(series_meta.get("lastdate_position") or "").strip().lower()
                == "lt_first"
            ):
                res.out_of_range_lt_first = True

        # Para latest con target_reference_date, solo coinc. exacta
        selected_obs: Optional[Dict[str, Any]] = None
        if target_reference_date and series_observations:
            selected_obs = next(
                (
                    row
                    for row in series_observations
                    if isinstance(row, dict)
                    and str(row.get("date") or "").strip()
                    == target_reference_date
                ),
                None,
            )

        if selected_obs is None and series_observations:
            if req_form_norm == "latest" and target_reference_date:
                selected_obs = None
            else:
                selected_obs = series_observations[-1]

        if selected_obs is None and isinstance(series_latest_obs, dict):
            if not (req_form_norm == "latest" and target_reference_date):
                selected_obs = series_latest_obs
                res.used_latest_fallback_for_point = True
                res.req_form_for_payload = "point"

        if not isinstance(selected_obs, dict):
            continue

        row_title = str(series.get("short_title") or series_id).strip()
        cls_row = (
            (series.get("classification") or {})
            if isinstance(series, dict)
            else {}
        )
        comparison_value = selected_obs.get("yoy")
        if comparison_value is None:
            comparison_value = selected_obs.get("prev_period")

        res.observations.append(
            {
                "series_id": series_id,
                "title": row_title,
                "activity": _normalize_cls_value(cls_row.get("activity")),
                "region": _normalize_cls_value(cls_row.get("region")),
                "investment": _normalize_cls_value(cls_row.get("investment")),
                "date": selected_obs.get("date"),
                "value": selected_obs.get("value"),
                "yoy": selected_obs.get("yoy"),
                "prev_period": selected_obs.get("prev_period"),
                "comparison_value": comparison_value,
            }
        )

    res.observations_all = list(res.observations)

    # Reducir a la serie objetivo
    target_row = _find_row_by_series_id(res.observations, target_series_id)
    if isinstance(target_row, dict):
        res.target_series_title = str(
            target_row.get("title") or family_name or ""
        ).strip()
        res.observations = [target_row]

    return res


# ---------------------------------------------------------------------------
# Rama 3: Serie específica / sin desglose
# ---------------------------------------------------------------------------

def fetch_specific_series(
    *,
    fetch_fn: FetchFn,
    load_fn: LoadFn,
    target_series_id: Optional[str],
    req_form_cls: Any,
    frequency_ent: Optional[str],
    indicator_ent: Optional[str],
    period_values: List[Any],
    period_ent: List[Any],
    calc_mode_cls: Any,
) -> FetchResult:
    """Obtiene datos de una serie individual (sin desglose).

    Para PIB anual, valida que el año solicitado tenga los 4 trimestres
    completos; de lo contrario retorna el último año completo con una nota.
    """
    firstdate = str(period_values[0]) if period_values else None
    lastdate = str(period_values[-1]) if period_values else None
    agg_mode = "sum"
    variation_mode = (
        calc_mode_cls if calc_mode_cls in {"prev_period", "yoy"} else "yoy"
    )

    res = FetchResult(
        req_form_for_payload=req_form_cls,
        effective_reference_period=str(period_values[-1]) if period_values else None,
        target_series_id=target_series_id,
        period_ent=list(period_ent),
        period_values=list(period_values),
    )

    observations, latest_obs, observations_meta = fetch_fn(
        series_id=target_series_id,
        req_form=req_form_cls,
        frequency=frequency_ent,
        indicator=indicator_ent,
        firstdate=firstdate,
        lastdate=lastdate,
        calc_mode=variation_mode,
    )
    res.observations = observations
    res.latest_obs = latest_obs
    res.observations_meta = observations_meta

    # Fallback si se pidió periodo específico sin datos
    if (
        str(req_form_cls or "").strip().lower()
        in {"point", "specific_point", "range"}
        and not observations
        and isinstance(latest_obs, dict)
    ):
        res.observations = [latest_obs]
        res.used_latest_fallback_for_point = True
        res.req_form_for_payload = "point"

    # Validar completitud anual para PIB anual
    _validate_annual_completeness(
        res,
        load_fn=load_fn,
        target_series_id=target_series_id,
        indicator_ent=indicator_ent,
        frequency_ent=frequency_ent,
        period_values=period_values,
        req_form_cls=req_form_cls,
        calc_mode_cls=calc_mode_cls,
        agg_mode=agg_mode,
        firstdate=firstdate,
        lastdate=lastdate,
    )

    return res


def _validate_annual_completeness(
    res: FetchResult,
    *,
    load_fn: LoadFn,
    target_series_id: Optional[str],
    indicator_ent: Optional[str],
    frequency_ent: Optional[str],
    period_values: List[Any],
    req_form_cls: Any,
    calc_mode_cls: Any,
    agg_mode: str,
    firstdate: Optional[str],
    lastdate: Optional[str],
) -> None:
    """Para PIB anual, verifica que el año solicitado tenga 4 trimestres
    con datos. Si no, busca el último año completo y genera una nota."""
    requested_start_year = extract_year(period_values[0]) if period_values else None
    requested_end_year = extract_year(period_values[-1]) if period_values else None
    req_form_norm = str(req_form_cls or "").strip().lower()

    latest_available_year = None
    if res.observations:
        latest_available_year = extract_year(
            str(res.observations[-1].get("date") or "")
        )
    elif isinstance(res.latest_obs, dict):
        latest_available_year = extract_year(
            str(res.latest_obs.get("date") or "")
        )

    validation_year = requested_end_year
    if validation_year is None and req_form_norm == "latest":
        validation_year = latest_available_year

    should_validate = (
        str(indicator_ent or "").strip().lower() == "pib"
        and str(frequency_ent or "").strip().lower() == "a"
        and validation_year is not None
        and (
            req_form_norm in {"point", "specific_point", "latest"}
            or (
                req_form_norm == "range"
                and requested_start_year is not None
                and requested_end_year is not None
                and requested_start_year == requested_end_year
            )
        )
        and (bool(res.observations) or isinstance(res.latest_obs, dict))
    )

    if not should_validate or validation_year is None:
        return

    variation_mode = (
        calc_mode_cls if calc_mode_cls in {"prev_period", "yoy"} else "yoy"
    )
    q_firstdate = firstdate if req_form_norm != "latest" else None
    q_lastdate = lastdate if req_form_norm != "latest" else None

    quarterly_observations, _ = load_fn(
        series_id=target_series_id,
        firstdate=q_firstdate,
        lastdate=q_lastdate,
        target_frequency="Q",
        agg_mode=agg_mode,
        calc_mode=variation_mode,
    )

    if has_full_quarterly_year(quarterly_observations, validation_year):
        return

    annual_observations_full, _ = load_fn(
        series_id=target_series_id,
        firstdate=None,
        lastdate=None,
        target_frequency="A",
        agg_mode=agg_mode,
        calc_mode=variation_mode,
    )

    fallback_obs = latest_annual_observation_before_year(
        annual_observations_full, validation_year
    )
    if not isinstance(fallback_obs, dict):
        return

    fallback_date = str(fallback_obs.get("date") or "").strip()
    fallback_year = extract_year(fallback_date)
    res.observations = [fallback_obs]
    res.latest_obs = fallback_obs
    res.req_form_for_payload = "point"
    res.used_latest_fallback_for_point = True

    if fallback_date:
        res.period_ent = [fallback_date, fallback_date]
        res.period_values = [fallback_date, fallback_date]

    res.incomplete_frequency_note = (
        f"Como la información de {validation_year} aún no está completa, "
        f"se informa el resultado de {fallback_year or 'N/D'}."
    )


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _select_observation_at_reference(
    series_observations: List[Dict[str, Any]],
    series_latest_obs: Optional[Dict[str, Any]],
    target_reference_date: Optional[str],
    req_form_norm: str,
) -> Optional[Dict[str, Any]]:
    """Selecciona la observación que coincida con la fecha de referencia,
    o la más reciente disponible como fallback."""
    selected: Optional[Dict[str, Any]] = None

    if target_reference_date and series_observations:
        selected = next(
            (
                row
                for row in series_observations
                if isinstance(row, dict)
                and str(row.get("date") or "").strip() == target_reference_date
            ),
            None,
        )
        # Para latest, buscar la más cercana anterior a la referencia
        if selected is None and req_form_norm == "latest":
            candidates = [
                row
                for row in series_observations
                if isinstance(row, dict)
                and str(row.get("date") or "").strip()
                and str(row.get("date") or "").strip()
                <= str(target_reference_date)
            ]
            if candidates:
                selected = max(
                    candidates,
                    key=lambda item: str(item.get("date") or "").strip(),
                )

    if selected is None and series_observations:
        selected = series_observations[-1]

    if selected is None and isinstance(series_latest_obs, dict):
        if not (req_form_norm == "latest" and target_reference_date):
            selected = series_latest_obs

    return selected


def _find_row_by_series_id(
    observations: List[Dict[str, Any]], series_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Busca una observación por ID de serie."""
    return next(
        (
            row
            for row in observations
            if str(row.get("series_id") or "").strip()
            == str(series_id or "").strip()
        ),
        None,
    )


def _normalize_cls_value(value: Any) -> Optional[str]:
    """Normaliza un valor de clasificación a lowercase, o None si está vacío."""
    if value in (None, ""):
        return None
    return str(value).strip().lower()


def _distinct_dimension_values(
    family_series: List[Dict[str, Any]], dimension_key: str
) -> set[str]:
    """Extrae los valores distintos de una dimensión de clasificación,
    excluyendo vacíos, 'none', 'null', 'general' y 'total'."""
    values: set[str] = set()
    for series in family_series:
        if not isinstance(series, dict):
            continue
        classification = series.get("classification")
        if not isinstance(classification, dict):
            continue
        raw = str(classification.get(dimension_key) or "").strip().lower()
        if raw in {"", "none", "null", "general", "total"}:
            continue
        values.add(raw)
    return values
