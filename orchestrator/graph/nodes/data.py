"""Data node and supporting helpers for PIBot graph."""

from __future__ import annotations

import os
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langgraph.types import StreamWriter

from orchestrator.data import flow_data
from orchestrator.data.response import build_no_series_message, format_period_labels

from ..state import (
    AgentState,
    _clone_entities,
    _emit_stream_chunk,
)

logger = logging.getLogger(__name__)


def _first_non_empty(value: Any) -> Any:
    if isinstance(value, list):
        for item in value:
            if item not in (None, "", [], {}, ()):
                return item
        return None
    if value in (None, "", [], {}, ()):
        return None
    return value


def _coerce_period(period_value: Any) -> List[Any]:
    if period_value in (None, "", [], {}, ()):
        return []
    if isinstance(period_value, list):
        return period_value
    return [period_value]


def _extract_year(value: Any) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", str(value or "").strip())
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _build_target_series_url(
    *,
    source_url: Optional[str],
    series_id: Optional[str],
    period: Optional[List[Any]],
    req_form: Optional[str] = None,
    observations: Optional[List[Dict[str, Any]]] = None,
    frequency: Optional[str] = None,
    calc_mode: Optional[str] = None,
) -> Optional[str]:
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
    observed_end_year = _extract_year_local(observed_rows[-1].get("date")) if observed_rows else None

    use_observed_end = req == "latest"
    if requested_end_year and observed_end_year and requested_end_year != observed_end_year:
        use_observed_end = True

    end_year = observed_end_year if use_observed_end and observed_end_year else requested_end_year
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
                value = row.get(key)
                if value is not None:
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

def _load_series_observations(
    *,
    series_id: Optional[str],
    firstdate: Optional[str],
    lastdate: Optional[str],
    target_frequency: Optional[str],
    agg_mode: str,
    calc_mode: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    from orchestrator.data.get_data_serie import get_series_from_redis

    series_data = get_series_from_redis(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=target_frequency,
        agg=agg_mode,
        use_fallback=True,
    )
    metadata = (series_data or {}).get("meta") if isinstance(series_data, dict) else None
    observations = [
        obs for obs in ((series_data or {}).get("observations") or []) if isinstance(obs, dict)
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

def _fetch_series_by_req_form(
    *,
    series_id: Optional[str],
    req_form: Optional[str],
    frequency: Optional[str],
    indicator: Optional[str],
    firstdate: Optional[str],
    lastdate: Optional[str],
    calc_mode: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    req = str(req_form or "").strip().lower()
    target_frequency = str(frequency or "").upper() or None
    agg_mode = "sum" # if str(indicator or "").strip().lower() == "pib" else "avg"

    if req == "latest":
        observations, metadata = _load_series_observations(
            series_id=series_id,
            firstdate=None,
            lastdate=None,
            target_frequency=target_frequency,
            agg_mode=agg_mode,
            calc_mode=calc_mode,
        )
        latest_obs = observations[-1] if observations else None
        return observations, latest_obs, metadata

    # point/range/etc: usar rango entregado
    observations, metadata = _load_series_observations(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=target_frequency,
        agg_mode=agg_mode,
        calc_mode=calc_mode,
    )
    latest_obs = observations[-1] if observations else None

    if req in {"point", "range"} and not observations:
        fallback_obs, _ = _load_series_observations(
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


def _quarter_from_date(value: Any) -> Optional[Tuple[int, int]]:
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


def _parse_iso_date(value: Any) -> Optional[Tuple[int, int, int]]:
    date_text = str(value or "").strip()
    if not date_text:
        return None
    try:
        parts = date_text[:10].split("-")
        if len(parts) != 3:
            return None
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        return year, month, day
    except Exception:
        return None


def _sort_observations_by_date_desc(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid_rows = [row for row in (observations or []) if isinstance(row, dict)]
    return sorted(
        valid_rows,
        key=lambda row: _parse_iso_date(row.get("date")) or (0, 0, 0),
        reverse=True,
    )


def _same_requested_period(requested_date: Optional[str], observed_date: Optional[str], frequency: Optional[str]) -> bool:
    requested_parts = _parse_iso_date(requested_date)
    observed_parts = _parse_iso_date(observed_date)
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


def _has_full_quarterly_year(observations: List[Dict[str, Any]], year: int) -> bool:
    quarters: set[int] = set()
    for row in observations or []:
        if not isinstance(row, dict):
            continue
        quarter_key = _quarter_from_date(row.get("date"))
        if quarter_key is None:
            continue
        row_year, row_quarter = quarter_key
        if row_year == year:
            quarters.add(row_quarter)
    return len(quarters) == 4


def _latest_annual_observation_before_year(
    observations: List[Dict[str, Any]],
    year_limit: int,
) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for row in observations or []:
        if not isinstance(row, dict):
            continue
        date_text = str(row.get("date") or "").strip()
        row_year = _extract_year(date_text)
        if row_year is None or row_year >= year_limit:
            continue
        candidates.append((date_text, row))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def make_data_node(memory_adapter: Any):
    def data_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        question = state.get("question", "")
        session_id = state.get("session_id", "")
        entities_state = _clone_entities(state.get("entities"))

        classification = state.get("classification")
        predict_raw = getattr(classification, "predict_raw", None) if classification else None
        predict_raw = predict_raw if isinstance(predict_raw, dict) else {} # BORRAR !!
        
        calc_mode_cls = getattr(classification, "calc_mode", None) or {}
        activity_cls = getattr(classification, "activity", None) or {}
        region_cls = getattr(classification, "region", None) or {}
        investment_cls = getattr(classification, "investment", None) or {}
        req_form_cls = getattr(classification, "req_form", None) or {}

        classification_entities = getattr(classification, "entities", None) or {}
        normalized_from_classification = getattr(classification, "normalized", None) or {}

        entities: List[Dict[str, Any]]
        if isinstance(classification_entities, dict):
            entities = [dict(classification_entities)]
        else:
            entities = entities_state

        interpretation_root = predict_raw.get("interpretation")
        if not isinstance(interpretation_root, dict):
            interpretation_root = predict_raw
        interpretation_root = interpretation_root if isinstance(interpretation_root, dict) else {}

        normalized_from_predict = interpretation_root.get("entities_normalized")
        normalized_from_predict = normalized_from_predict if isinstance(normalized_from_predict, dict) else {}

        # For follow-up turns, intent node mutates predict_raw.entities_normalized.
        # Prefer that payload so downstream uses the resolved indicator/time fields.
        normalized = normalized_from_predict or (
            normalized_from_classification if isinstance(normalized_from_classification, dict) else {}
        )

        indicator_ent = _first_non_empty(normalized.get("indicator"))
        seasonality_ent = _first_non_empty(normalized.get("seasonality"))
        frequency_ent = _first_non_empty(normalized.get("frequency"))
        activity_ent = _first_non_empty(normalized.get("activity"))
        region_ent = _first_non_empty(normalized.get("region"))
        investment_ent = _first_non_empty(normalized.get("investment"))
        period_ent = _coerce_period(normalized.get("period"))
        period_values = period_ent if isinstance(period_ent, list) else []
        period_reference_year = _extract_year(period_values[-1]) if period_values else None
        
        # REGLAS DE NEGOCIO PARA TRASLADAR !
        activity_cls_resolved = activity_cls
        activity_ent_resolved = activity_ent

        if (
            calc_mode_cls == "contribution"
            and activity_cls in (None, "none")
            and investment_cls in (None, "none")
            and region_cls in (None, "none")
        ):
            activity_cls_resolved = "general"

        if str(indicator_ent or "").strip().lower() == "imacec" and str(frequency_ent or "").strip().lower() != "m":
            frequency_ent = "m"
            logger.info("[DATA_NODE] IMACEC: forzando frecuencia mensual (m)")

        if indicator_ent == "imacec" and activity_ent is None:
            activity_ent_resolved = "imacec"
            
        price = None
        if indicator_ent == "pib" and activity_cls == "none" and region_cls == "none" and investment_cls == "none":
            price = None
        else:
            price = "enc"

        hist = 1 if indicator_ent == "pib" and str(frequency_ent or "").strip().lower() == "a" and period_reference_year is not None and period_reference_year < 1996 else None
        monthly_frequency_note: Optional[str] = None
        
        logger.info("[DATA_NODE] =========================================================")
        logger.info("[DATA_NODE] calc_mode=%s", calc_mode_cls)
        logger.info("[DATA_NODE] activity=%s", activity_cls_resolved)
        logger.info("[DATA_NODE] region=%s", region_cls)
        logger.info("[DATA_NODE] investment=%s", investment_cls)
        logger.info("[DATA_NODE] req_form=%s", req_form_cls)
        logger.info("[DATA_NODE] entities=%s", entities)
        logger.info("[DATA_NODE] normalized=%s", normalized)        
        logger.info("[DATA_NODE] indicator=%s", indicator_ent)
        logger.info("[DATA_NODE] seasonality=%s", seasonality_ent)
        logger.info("[DATA_NODE] frequency=%s", frequency_ent)
        logger.info("[DATA_NODE] activity=%s", activity_ent_resolved)
        logger.info("[DATA_NODE] region=%s", region_ent)
        logger.info("[DATA_NODE] investment=%s", investment_ent)
        logger.info("[DATA_NODE] price=%s", price)
        logger.info("[DATA_NODE] hist=%s", hist)
        logger.info("[DATA_NODE] period=%s", period_ent)
        logger.info("[DATA_NODE] =========================================================")

        if indicator_ent == "pib" and str(frequency_ent or "").strip().lower() == "m":
            req_form_norm = str(req_form_cls or "").strip().lower()
            requested_month_label = None
            if period_values:
                requested_month_label = format_period_labels(str(period_values[-1]), "m")[0]
                if requested_month_label == "--":
                    requested_month_label = None

            if str(calc_mode_cls or "").strip().lower() == "contribution":
                monthly_frequency_note = (
                    "Las contribuciones al PIB no se calculan de forma mensual; sin embargo, te comparto la última contribución trimestral disponible."
                )
            else:
                monthly_frequency_note = (
                    "El PIB no se calcula de forma mensual; sin embargo, te comparto el último trimestre disponible."
                )
            logger.warning("[DATA_NODE] %s", monthly_frequency_note)
            frequency_ent = "q"
            req_form_cls = "latest"
            period_ent = []
            period_values = []
        
        
        ## Obtener ID Series
        #####################
        
        from orchestrator.catalog.series_search import (
            family_to_series_rows,
            find_family_by_classification,
            select_target_series_by_classification,
        )

        family_frequency = None if indicator_ent == "imacec" else frequency_ent
        if calc_mode_cls != "contribution":
            family_frequency = None
        family_price = None if indicator_ent == "imacec" else price
        is_pib_aggregate = (
            indicator_ent == "pib"
            and activity_cls_resolved in (None, "none", "general")
            and region_cls in (None, "none")
            and investment_cls in (None, "none")
        )
        if calc_mode_cls == "contribution":
            family_calc_mode = "contribution"
        else:
            family_calc_mode = "original"
        requested_seasonality = str(seasonality_ent or "").strip().lower()
        has_requested_seasonality = requested_seasonality not in {"", "none", "null"}
        if is_pib_aggregate and calc_mode_cls != "contribution":
            family_seasonality = None
        elif is_pib_aggregate and not has_requested_seasonality:
            family_seasonality = None
        else:
            family_seasonality = seasonality_ent

        # Buscar una sola familia de series en el catalogo agrupado
        family_dict = find_family_by_classification(
            "orchestrator/catalog/catalog.json",
            indicator=indicator_ent,
            activity_value=activity_ent_resolved if activity_ent_resolved is not None else activity_cls_resolved,
            region_value=region_ent if region_ent is not None else region_cls,
            investment_value=investment_ent if investment_ent is not None else investment_cls,
            calc_mode=family_calc_mode,
            price=family_price,
            seasonality=family_seasonality,
            frequency=family_frequency,
            hist=hist,
        )
        family_series = family_to_series_rows(family_dict) if isinstance(family_dict, dict) else []
        source_family_series = family_dict.get("source_url") if isinstance(family_dict, dict) else None
        family_name = family_dict.get("family_name") if isinstance(family_dict, dict) else None
        
        logger.info(
            "[DATA_NODE] family_name=%s",
            family_name,
        )
        logger.info(
            "[DATA_NODE] family_source_url=%s",
            source_family_series,
        )
        logger.info("[DATA_NODE] =========================================================")

        # Buscar serie especifica a partir de la familia de series
        series_eq = {
            "indicator": indicator_ent,
            "calc_mode": calc_mode_cls if calc_mode_cls == "contribution" else "original",
            "seasonality": seasonality_ent,
            "activity": activity_ent_resolved,
            "region": region_ent,
            "investment": investment_ent,
        }

        if (
            calc_mode_cls == "contribution"
            and activity_cls_resolved in (None, "none", "general")
            and region_cls in (None, "none")
            and investment_cls in (None, "none")
        ):
            indicator_norm = str(indicator_ent or "").strip().lower()
            if indicator_norm in {"pib", "imacec"}:
                activity_tokens_in_family = {
                    str(((row.get("classification") or {}).get("activity") or "")).strip().lower()
                    for row in family_series
                    if isinstance(row, dict)
                }
                if indicator_norm in activity_tokens_in_family:
                    series_eq["activity"] = indicator_norm
                else:
                    series_eq.pop("activity", None)
                    series_eq["indicator"] = indicator_norm

        if calc_mode_cls == "contribution":
            series_eq["frequency"] = frequency_ent

        if activity_cls_resolved == "specific" and activity_ent_resolved is None:
            series_eq["activity"] = "__missing_specific_activity__"

        target_series = select_target_series_by_classification(
            family_series,
            eq=series_eq,
            fallback_to_first=True,
        )
        
        target_series_id = target_series.get("id") if isinstance(target_series, dict) else None
        target_series_long_raw = target_series.get("long_title") if isinstance(target_series, dict) else None
        target_series_display_raw = target_series.get("display_title") if isinstance(target_series, dict) else None
        target_series_title = str(target_series_long_raw or target_series_display_raw or "").strip()
        target_series_url = _build_target_series_url(
            source_url=source_family_series,
            series_id=target_series_id,
            period=period_ent if isinstance(period_ent, list) else None,
            frequency=frequency_ent,
            calc_mode=calc_mode_cls,
        )

        logger.info(
            "[DATA_NODE] target_series_id=%s",
            target_series_id,
        )
        logger.info(
            "[DATA_NODE] target_series_title=%s",
            target_series_title,
        )
        logger.info(
            "[DATA_NODE] target_series_url=%s",
            target_series_url,
        )
        logger.info("[DATA_NODE] =========================================================")

        if not source_family_series or not target_series_id:
            requested_activity = None
            primary_entity = entities[0] if isinstance(entities, list) and entities else {}
            if isinstance(primary_entity, dict):
                requested_activity = _first_non_empty(primary_entity.get("activity"))
                if requested_activity is not None:
                    requested_activity = str(requested_activity).strip()

            if requested_activity in {None, "", "[]", "none", "null"}:
                requested_activity = None

            indicator_candidate = indicator_ent
            if indicator_candidate in (None, "", [], {}, ()):  # fallback a entidad cruda
                indicator_candidate = _first_non_empty(primary_entity.get("indicator")) if isinstance(primary_entity, dict) else None
            indicator_label = str(indicator_candidate or "").strip().upper()
            if indicator_label in {"", "[]", "NONE", "NULL"}:
                indicator_label = None

            normalized_activity = str(activity_ent_resolved or "").strip()
            text = build_no_series_message(
                question=question,
                requested_activity=requested_activity,
                normalized_activity=normalized_activity,
                indicator_label=indicator_label,
            )

            logger.warning("[DATA_NODE] %s", text)
            _emit_stream_chunk(text, writer)
            return {
                "output": text,
                "entities": entities,
                "parsed_point": None,
                "parsed_range": None,
                "series": None,
                "data_classification": {
                    "indicator": indicator_ent,
                    "seasonality": seasonality_ent,
                    "frequency": frequency_ent,
                    "period": period_ent,
                    "calc_mode_cls": calc_mode_cls,
                    "activity_cls": activity_cls_resolved,
                    "region_cls": region_cls,
                    "investment_cls": investment_cls,
                    "req_form_cls": req_form_cls,
                    "activity_value": activity_ent,
                    "region_value": region_ent,
                    "investment_value": investment_ent,
                },
            }
        
        
        ## Obtener data 
        ####################
        
        firstdate = str(period_values[0]) if period_ent else None if period_values else None
        lastdate = str(period_values[-1]) if period_ent else None if period_values else None
        observations: List[Dict[str, Any]] = []
        observations_all: List[Dict[str, Any]] = []
        latest_obs: Optional[Dict[str, Any]] = None
        observations_meta: Optional[Dict[str, Any]] = None
        used_latest_fallback_for_point = False
        req_form_for_payload = req_form_cls
        out_of_range_lt_first = False
        incomplete_frequency_note: Optional[str] = None
        effective_reference_period: Optional[str] = str(period_values[-1]) if period_values else None
        agg_mode = "sum"
        calc_mode_norm = str(calc_mode_cls or "").strip().lower()
        is_non_contribution_general_breakdown = (
            calc_mode_norm != "contribution"
            and any(
                str(slot or "").strip().lower() == "general"
                for slot in (activity_cls, region_cls, investment_cls)
            )
        )
        breakdown_variation_mode = calc_mode_norm if calc_mode_norm in {"yoy", "prev_period"} else "yoy"
        
        if calc_mode_cls == "contribution":
            req_form_norm = str(req_form_cls or "").strip().lower()
            requested_period_end = str(period_values[-1]) if period_values else None
            target_reference_date: Optional[str] = None

            if target_series_id:
                target_observations, target_latest_obs, _ = _fetch_series_by_req_form(
                    series_id=target_series_id,
                    req_form=req_form_cls,
                    frequency=frequency_ent,
                    indicator=indicator_ent,
                    firstdate=firstdate,
                    lastdate=lastdate,
                )

                if target_observations:
                    target_reference_date = str(target_observations[-1].get("date") or "").strip() or None
                elif isinstance(target_latest_obs, dict):
                    target_reference_date = str(target_latest_obs.get("date") or "").strip() or None

                if req_form_norm == "latest" and target_reference_date:
                    component_latest_dates: List[str] = []
                    for component_series in family_series:
                        component_series_id = component_series.get("id") if isinstance(component_series, dict) else None
                        if not component_series_id or component_series_id == target_series_id:
                            continue

                        component_observations, component_latest_obs, _ = _fetch_series_by_req_form(
                            series_id=component_series_id,
                            req_form=req_form_cls,
                            frequency=frequency_ent,
                            indicator=indicator_ent,
                            firstdate=firstdate,
                            lastdate=lastdate,
                        )

                        component_latest_date = None
                        if component_observations:
                            component_latest_date = str(component_observations[-1].get("date") or "").strip() or None
                        elif isinstance(component_latest_obs, dict):
                            component_latest_date = str(component_latest_obs.get("date") or "").strip() or None

                        if component_latest_date:
                            component_latest_dates.append(component_latest_date)

                    if component_latest_dates and target_reference_date not in component_latest_dates:
                        aligned_reference_date = max(component_latest_dates)
                        logger.info(
                            "[DATA_NODE] contribution latest: aligning target date from %s to common component date %s",
                            target_reference_date,
                            aligned_reference_date,
                        )
                        target_reference_date = aligned_reference_date

                if req_form_norm == "latest" and target_reference_date:
                    effective_reference_period = target_reference_date

                if (
                    req_form_norm in {"point", "specific_point", "range"}
                    and requested_period_end
                    and target_reference_date
                    and not _same_requested_period(requested_period_end, target_reference_date, frequency_ent)
                ):
                    used_latest_fallback_for_point = True
                    req_form_for_payload = "point"

            # recorrer family_series y obtener data de cada serie para luego agregar info a data_params y metadata_response
            for series in family_series:
                series_id = series.get("id")
                if not series_id:
                    continue
                series_observations, series_latest_obs, series_meta = _fetch_series_by_req_form(
                    series_id=series_id,
                    req_form=req_form_cls,
                    frequency=frequency_ent,
                    indicator=indicator_ent,
                    firstdate=firstdate,
                    lastdate=lastdate,
                )

                if isinstance(series_meta, dict):
                    position = str(series_meta.get("lastdate_position") or "").strip().lower()
                    if position == "lt_first":
                        out_of_range_lt_first = True

                selected_series_obs: Optional[Dict[str, Any]] = None
                if target_reference_date and series_observations:
                    selected_series_obs = next(
                        (
                            row
                            for row in series_observations
                            if isinstance(row, dict)
                            and str(row.get("date") or "").strip() == target_reference_date
                        ),
                        None,
                    )

                    if selected_series_obs is None and req_form_norm == "latest":
                        candidates_at_or_before_ref = [
                            row
                            for row in series_observations
                            if isinstance(row, dict)
                            and str(row.get("date") or "").strip()
                            and str(row.get("date") or "").strip() <= str(target_reference_date)
                        ]
                        if candidates_at_or_before_ref:
                            selected_series_obs = max(
                                candidates_at_or_before_ref,
                                key=lambda item: str(item.get("date") or "").strip(),
                            )

                if selected_series_obs is None and series_observations:
                    selected_series_obs = series_observations[-1]

                if selected_series_obs is None and isinstance(series_latest_obs, dict):
                    if not (req_form_norm == "latest" and target_reference_date):
                        selected_series_obs = series_latest_obs
                        used_latest_fallback_for_point = True
                        req_form_for_payload = "point"

                if not isinstance(selected_series_obs, dict):
                    continue

                latest_series_obs = selected_series_obs

                row_title = str(series.get("short_title") or series_id).strip()
                series_activity = (
                    (series.get("classification") or {}).get("activity")
                    if isinstance(series, dict)
                    else None
                )
                series_activity_normalized = str(series_activity).strip().lower() if series_activity not in (None, "") else None

                observations.append(
                    {
                        "series_id": series_id,
                        "title": row_title,
                        "activity": series_activity_normalized,
                        "date": latest_series_obs.get("date"),
                        "value": latest_series_obs.get("value"),
                    }
                )
            observations_all = list(observations)

            def _distinct_dimension_values(dimension_key: str) -> set[str]:
                values: set[str] = set()
                for series in family_series:
                    if not isinstance(series, dict):
                        continue
                    classification = series.get("classification")
                    if not isinstance(classification, dict):
                        continue
                    raw_value = str(classification.get(dimension_key) or "").strip().lower()
                    if raw_value in {"", "none", "null", "general", "total"}:
                        continue
                    values.add(raw_value)
                return values

            activity_values = _distinct_dimension_values("activity")
            region_values = _distinct_dimension_values("region")
            investment_values = _distinct_dimension_values("investment")

            has_activity_dimension = len(activity_values) > 1
            has_region_dimension = len(region_values) > 1
            has_investment_dimension = len(investment_values) > 1

            is_specific_contribution_dimension = (
                (str(activity_cls_resolved or "").strip().lower() == "specific" and has_activity_dimension)
                or (str(region_cls or "").strip().lower() == "specific" and has_region_dimension)
                or (str(investment_cls or "").strip().lower() == "specific" and has_investment_dimension)
            )
            if is_specific_contribution_dimension:
                target_row_specific = next(
                    (
                        row
                        for row in observations
                        if str(row.get("series_id") or "").strip() == str(target_series_id or "").strip()
                    ),
                    None,
                )
                if isinstance(target_row_specific, dict):
                    observations = [target_row_specific]
                    observations_all = [target_row_specific]
                    target_series_title = str(
                        target_row_specific.get("title") or target_series_title or family_name or ""
                    ).strip()

            if activity_cls_resolved == "general" and activity_ent is None:
                aggregate_row = next(
                    (
                        row
                        for row in observations
                        if str(row.get("series_id") or "").strip() == str(target_series_id or "").strip()
                    ),
                    None,
                )
                for row in observations:
                    if isinstance(aggregate_row, dict):
                        break
                    title_norm = str(row.get("title") or "").strip().lower()
                    if title_norm in {"pib", "imacec"}:
                        aggregate_row = row
                        break
                    if aggregate_row is None and row.get("activity") in (None, "", "total"):
                        aggregate_row = row

                if isinstance(aggregate_row, dict):
                    target_series_id = aggregate_row.get("series_id")
                    target_series_title = str(
                        aggregate_row.get("title") or family_name or ""
                    ).strip()
                    target_series_url = _build_target_series_url(
                        source_url=source_family_series,
                        series_id=target_series_id,
                        period=period_ent if isinstance(period_ent, list) else None,
                        req_form=req_form_cls,
                        observations=observations,
                        frequency=frequency_ent,
                        calc_mode=calc_mode_cls,
                    )
                    observations = [aggregate_row]

        elif is_non_contribution_general_breakdown:
            req_form_norm = str(req_form_cls or "").strip().lower()
            requested_period_end = str(period_values[-1]) if period_values else None
            target_reference_date: Optional[str] = None

            if target_series_id:
                target_observations, target_latest_obs, observations_meta = _fetch_series_by_req_form(
                    series_id=target_series_id,
                    req_form=req_form_cls,
                    frequency=frequency_ent,
                    indicator=indicator_ent,
                    firstdate=firstdate,
                    lastdate=lastdate,
                    calc_mode=breakdown_variation_mode,
                )

                if target_observations:
                    target_reference_date = str(target_observations[-1].get("date") or "").strip() or None
                elif isinstance(target_latest_obs, dict):
                    target_reference_date = str(target_latest_obs.get("date") or "").strip() or None

                if req_form_norm == "latest" and target_reference_date:
                    breakdown_latest_dates: List[str] = []
                    for breakdown_series in family_series:
                        breakdown_series_id = breakdown_series.get("id") if isinstance(breakdown_series, dict) else None
                        if not breakdown_series_id or breakdown_series_id == target_series_id:
                            continue

                        breakdown_observations, breakdown_latest_obs, _ = _fetch_series_by_req_form(
                            series_id=breakdown_series_id,
                            req_form=req_form_cls,
                            frequency=frequency_ent,
                            indicator=indicator_ent,
                            firstdate=firstdate,
                            lastdate=lastdate,
                            calc_mode=breakdown_variation_mode,
                        )

                        breakdown_latest_date = None
                        if breakdown_observations:
                            breakdown_latest_date = str(breakdown_observations[-1].get("date") or "").strip() or None
                        elif isinstance(breakdown_latest_obs, dict):
                            breakdown_latest_date = str(breakdown_latest_obs.get("date") or "").strip() or None

                        if breakdown_latest_date:
                            breakdown_latest_dates.append(breakdown_latest_date)

                    if breakdown_latest_dates and target_reference_date not in breakdown_latest_dates:
                        aligned_reference_date = max(breakdown_latest_dates)
                        logger.info(
                            "[DATA_NODE] non-contribution general latest: aligning target date from %s to common breakdown date %s",
                            target_reference_date,
                            aligned_reference_date,
                        )
                        target_reference_date = aligned_reference_date

                if req_form_norm == "latest" and target_reference_date:
                    effective_reference_period = target_reference_date

                if (
                    req_form_norm in {"point", "specific_point", "range"}
                    and requested_period_end
                    and target_reference_date
                    and not _same_requested_period(requested_period_end, target_reference_date, frequency_ent)
                ):
                    used_latest_fallback_for_point = True
                    req_form_for_payload = "point"

            for series in family_series:
                series_id = series.get("id") if isinstance(series, dict) else None
                if not series_id:
                    continue

                series_observations, series_latest_obs, series_meta = _fetch_series_by_req_form(
                    series_id=series_id,
                    req_form=req_form_cls,
                    frequency=frequency_ent,
                    indicator=indicator_ent,
                    firstdate=firstdate,
                    lastdate=lastdate,
                    calc_mode=breakdown_variation_mode,
                )

                if isinstance(series_meta, dict):
                    position = str(series_meta.get("lastdate_position") or "").strip().lower()
                    if position == "lt_first":
                        out_of_range_lt_first = True

                selected_series_obs: Optional[Dict[str, Any]] = None
                if target_reference_date and series_observations:
                    selected_series_obs = next(
                        (
                            row
                            for row in series_observations
                            if isinstance(row, dict)
                            and str(row.get("date") or "").strip() == target_reference_date
                        ),
                        None,
                    )

                if selected_series_obs is None and series_observations:
                    if req_form_norm == "latest" and target_reference_date:
                        selected_series_obs = None
                    else:
                        selected_series_obs = series_observations[-1]

                if selected_series_obs is None and isinstance(series_latest_obs, dict):
                    if not (req_form_norm == "latest" and target_reference_date):
                        selected_series_obs = series_latest_obs
                        used_latest_fallback_for_point = True
                        req_form_for_payload = "point"

                if not isinstance(selected_series_obs, dict):
                    continue

                row_title = str(series.get("short_title") or series_id).strip()
                classification_row = (series.get("classification") or {}) if isinstance(series, dict) else {}
                series_activity = classification_row.get("activity")
                series_region = classification_row.get("region")
                series_investment = classification_row.get("investment")
                series_activity_normalized = str(series_activity).strip().lower() if series_activity not in (None, "") else None
                series_region_normalized = str(series_region).strip().lower() if series_region not in (None, "") else None
                series_investment_normalized = str(series_investment).strip().lower() if series_investment not in (None, "") else None

                comparison_value = selected_series_obs.get("yoy")
                if comparison_value is None:
                    comparison_value = selected_series_obs.get("prev_period")

                observations.append(
                    {
                        "series_id": series_id,
                        "title": row_title,
                        "activity": series_activity_normalized,
                        "region": series_region_normalized,
                        "investment": series_investment_normalized,
                        "date": selected_series_obs.get("date"),
                        "value": selected_series_obs.get("value"),
                        "yoy": selected_series_obs.get("yoy"),
                        "prev_period": selected_series_obs.get("prev_period"),
                        "comparison_value": comparison_value,
                    }
                )

            observations_all = list(observations)

            target_row = next(
                (
                    row
                    for row in observations
                    if str(row.get("series_id") or "").strip() == str(target_series_id or "").strip()
                ),
                None,
            )

            if isinstance(target_row, dict):
                target_series_title = str(target_row.get("title") or family_name or "").strip()
                observations = [target_row]

        elif activity_cls_resolved in ("specific", "none") and region_cls in ("specific", "none") and investment_cls in ("specific", "none"):

            observations, latest_obs, observations_meta = _fetch_series_by_req_form(
                series_id=target_series_id,
                req_form=req_form_cls,
                frequency=frequency_ent,
                indicator=indicator_ent,
                firstdate=firstdate,
                lastdate=lastdate,
                calc_mode=calc_mode_cls if calc_mode_cls in {"prev_period", "yoy"} else "yoy",
            )

            if (
                str(req_form_cls or "").strip().lower() in {"point", "specific_point", "range"}
                and not observations
                and isinstance(latest_obs, dict)
            ):
                observations = [latest_obs]
                used_latest_fallback_for_point = True
                req_form_for_payload = "point"

            requested_start_year = _extract_year(period_values[0]) if period_values else None
            requested_end_year = _extract_year(period_values[-1]) if period_values else None
            req_form_norm_local = str(req_form_cls or "").strip().lower()
            latest_available_year = None
            if observations:
                latest_available_year = _extract_year(str(observations[-1].get("date") or ""))
            elif isinstance(latest_obs, dict):
                latest_available_year = _extract_year(str(latest_obs.get("date") or ""))

            validation_year = requested_end_year
            if validation_year is None and req_form_norm_local == "latest":
                validation_year = latest_available_year

            should_validate_annual_completeness = (
                str(indicator_ent or "").strip().lower() == "pib"
                and str(frequency_ent or "").strip().lower() == "a"
                and validation_year is not None
                and (
                    req_form_norm_local in {"point", "specific_point", "latest"}
                    or (
                        req_form_norm_local == "range"
                        and requested_start_year is not None
                        and requested_end_year is not None
                        and requested_start_year == requested_end_year
                    )
                )
                and (bool(observations) or isinstance(latest_obs, dict))
            )

            if should_validate_annual_completeness and validation_year is not None:
                q_firstdate = firstdate
                q_lastdate = lastdate
                if req_form_norm_local == "latest":
                    q_firstdate = None
                    q_lastdate = None
                quarterly_observations, _ = _load_series_observations(
                    series_id=target_series_id,
                    firstdate=q_firstdate,
                    lastdate=q_lastdate,
                    target_frequency="Q",
                    agg_mode=agg_mode,
                    calc_mode=calc_mode_cls if calc_mode_cls in {"prev_period", "yoy"} else "yoy",
                )

                if not _has_full_quarterly_year(quarterly_observations, validation_year):
                    annual_observations_full, _ = _load_series_observations(
                        series_id=target_series_id,
                        firstdate=None,
                        lastdate=None,
                        target_frequency="A",
                        agg_mode=agg_mode,
                        calc_mode=calc_mode_cls if calc_mode_cls in {"prev_period", "yoy"} else "yoy",
                    )
                    fallback_annual_obs = _latest_annual_observation_before_year(
                        annual_observations_full,
                        validation_year,
                    )
                    if isinstance(fallback_annual_obs, dict):
                        fallback_date = str(fallback_annual_obs.get("date") or "").strip()
                        fallback_year = _extract_year(fallback_date)
                        observations = [fallback_annual_obs]
                        latest_obs = fallback_annual_obs
                        req_form_for_payload = "point"
                        used_latest_fallback_for_point = True
                        if fallback_date:
                            period_ent = [fallback_date, fallback_date]
                            period_values = [fallback_date, fallback_date]
                        incomplete_frequency_note = (
                            f"Como la información de {validation_year} aún no está completa, "
                            f"se informa el resultado de {fallback_year or 'N/D'}."
                        )

        req_form_norm = str(req_form_cls or "").strip().lower()
        if isinstance(observations_meta, dict):
            if str(observations_meta.get("lastdate_position") or "").strip().lower() == "lt_first":
                out_of_range_lt_first = True

        if out_of_range_lt_first and req_form_norm in {"point", "specific_point", "range"}:
            if not observations and isinstance(latest_obs, dict):
                observations = [latest_obs]
                used_latest_fallback_for_point = True
                req_form_for_payload = "point"

        if not observations and not observations_all:
            text = "No hay datos disponibles para la serie solicitada."
            logger.warning("[DATA_NODE] %s", text)
            _emit_stream_chunk(text, writer)
            return {
                "output": text,
                "entities": entities,
                "parsed_point": None,
                "parsed_range": (str(period_values[0]), str(period_values[-1])) if period_values else None,
                "series": target_series_id,
                "data_classification": {
                    "indicator": indicator_ent,
                    "seasonality": seasonality_ent,
                    "frequency": frequency_ent,
                    "period": period_ent,
                    "calc_mode_cls": calc_mode_cls,
                    "activity_cls": activity_cls_resolved,
                    "region_cls": region_cls,
                    "investment_cls": investment_cls,
                    "req_form_cls": req_form_for_payload,
                    "activity_value": activity_ent,
                    "region_value": region_ent,
                    "investment_value": investment_ent,
                },
            }

        
        ## Construir payload 
        ######################
        
        if observations is not None or observations_all is not None: 
            
            target_series_url = _build_target_series_url(
                source_url=source_family_series,
                series_id=target_series_id,
                period=period_ent if isinstance(period_ent, list) else None,
                req_form=req_form_cls,
                observations=observations if observations else observations_all,
                frequency=frequency_ent,
                calc_mode=calc_mode_cls,
            )
            
            payload = {
                "intent": "value",
                "classification": {
                    "indicator": indicator_ent,
                    "seasonality": seasonality_ent,
                    "frequency": frequency_ent,
                    "period": period_ent,
                    "calc_mode_cls": calc_mode_cls,
                    "activity_cls": activity_cls_resolved,
                    "region_cls": region_cls,
                    "investment_cls": investment_cls,
                    "req_form_cls": req_form_for_payload,
                    "macro_cls": "1",
                    "intent_cls": "value",
                    "context_cls": "followup",
                    "enable": None,
                    "enable_all": None,
                    "activity_value": activity_ent,
                    "sub_activity_value": activity_ent,
                    "region_value": region_ent,
                    "investment_value": investment_ent,
                    "gasto_value": investment_ent,
                    "price": None,
                    "history": hist
                },
                "series": target_series_id,
                "series_title": target_series_title or family_name or None,
                "parsed_point": str(period_values[-1]) if (req_form_for_payload != "range" and period_ent and period_values) else None,
                "parsed_range": (str(period_values[0]), str(period_values[-1])) if period_ent else None if period_values else None,
                "reference_period": effective_reference_period,
                "used_latest_fallback_for_point": used_latest_fallback_for_point,
                "result": observations,
                "all_series_data": observations_all or None,
                "source_url": target_series_url,
            }

            collected: List[str] = []
            try:
                if monthly_frequency_note:
                    collected.append(f"{monthly_frequency_note} ")
                    _emit_stream_chunk(f"{monthly_frequency_note} ", writer)
                if incomplete_frequency_note:
                    collected.append(f"{incomplete_frequency_note} ")
                    _emit_stream_chunk(f"{incomplete_frequency_note} ", writer)
                stream = flow_data.stream_data_flow(payload, session_id=session_id)
                for chunk in stream:
                    chunk_text = str(chunk)
                    if not chunk_text:
                        continue
                    collected.append(chunk_text)
                    _emit_stream_chunk(chunk_text, writer)
            except Exception:
                logger.exception("[DATA_NODE] Flujo fallido")
                if not collected:
                    fallback = "Ocurrió un problema al obtener los datos solicitados."
                    collected.append(fallback)
                    _emit_stream_chunk(fallback, writer)

            trace_info = {
                "parsed_point": payload.get("parsed_point"),
                "parsed_range": payload.get("parsed_range"),
                "series": payload.get("series"),
                "data_classification": payload.get("classification"),
                # "data_params": data_params,
                # "data_params_status": data_params_status,
                # "metadata_response": metadata_response,
                # "metadata_key": metadata_key,
                # "series_fetch_args": series_fetch_args,
                # "series_fetch_result": series_fetch_result,
                # "annual_validation": annual_validation,
            }
            return {"output": "".join(collected), "entities": entities, **trace_info}

        text = "[GRAPH] No se recibió clasificación para el nodo DATA."
        logger.warning("[=====DATA_NODE=====] %s", text)
        _emit_stream_chunk(text, writer)
        return {
            "output": text,
            "entities": entities,
            # "data_params": data_params,
            # "data_params_status": data_params_status,
            # "metadata_response": metadata_response,
            # "metadata_key": metadata_key,
            # "series_fetch_args": series_fetch_args,
            # "series_fetch_result": series_fetch_result,
            # "annual_validation": annual_validation,
        }


    return data_node


__all__ = ["make_data_node"]
