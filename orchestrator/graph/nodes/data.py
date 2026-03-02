"""Data node and supporting helpers for PIBot graph."""

from __future__ import annotations

import os
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langgraph.types import StreamWriter

from orchestrator.data import flow_data

from ..state import (
    AgentState,
    _clone_entities,
    _emit_stream_chunk,
)

logger = logging.getLogger(__name__)


def _extract_year(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else None


def _map_frequency_param(frequency: Optional[str]) -> Optional[str]:
    normalized = str(frequency or "").strip().lower()
    mapping = {
        "a": "ANNUAL",
        "annual": "ANNUAL",
        "anual": "ANNUAL",
        "q": "QUARTERLY",
        "quarterly": "QUARTERLY",
        "trimestral": "QUARTERLY",
        "m": "MONTHLY",
        "monthly": "MONTHLY",
        "mensual": "MONTHLY",
    }
    return mapping.get(normalized)


def _map_calc_param(calc_mode: Optional[str]) -> Optional[str]:
    normalized = str(calc_mode or "").strip().lower()
    mapping = {
        "yoy": "YTYPCT",
        "prev_period": "PCT",
    }
    return mapping.get(normalized)


def _build_target_series_url(
    *,
    source_url: Optional[str],
    series_id: Optional[str],
    period: Optional[List[Any]],
    frequency: Optional[str],
    calc_mode: Optional[str],
) -> Optional[str]:
    if not source_url or not series_id:
        return None

    period_values = period or []
    end_year = _extract_year(str(period_values[-1])) if period_values else None
    start_year = None
    if end_year is not None:
        try:
            start_year = str(int(end_year) - 10)
        except Exception:
            start_year = None

    frequency_param = _map_frequency_param(frequency)
    calc_param = _map_calc_param(calc_mode)

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


def _resolve_url_period_from_data(
    *,
    requested_period: Optional[List[Any]],
    req_form: Optional[str],
    observations: Optional[List[Dict[str, Any]]],
) -> Optional[List[Any]]:
    requested = requested_period if isinstance(requested_period, list) else None
    req = str(req_form or "").strip().lower()
    rows = [row for row in (observations or []) if isinstance(row, dict) and row.get("date")]
    if not rows:
        return requested

    data_start = str(rows[0].get("date"))
    data_end = str(rows[-1].get("date"))

    if req == "latest":
        return [data_end, data_end]

    requested_end = _extract_year(str(requested[-1])) if requested else None
    data_end_year = _extract_year(data_end)
    if requested_end and data_end_year and requested_end != data_end_year:
        return [data_end, data_end]

    requested_start = _extract_year(str(requested[0])) if requested else None
    data_start_year = _extract_year(data_start)
    if requested_start and data_start_year and requested_start != data_start_year and req in {"point", "specific_point"}:
        return [data_end, data_end]

    return requested


def _load_series_observations(
    *,
    series_id: Optional[str],
    firstdate: Optional[str],
    lastdate: Optional[str],
    target_frequency: Optional[str],
    agg_mode: str,
    calc_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from orchestrator.data.get_data_serie import get_series_from_redis

    series_data = get_series_from_redis(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=target_frequency,
        agg=agg_mode,
        use_fallback=True,
    )
    observations = [
        obs for obs in ((series_data or {}).get("observations") or []) if isinstance(obs, dict)
    ]

    mode = str(calc_mode or "").strip().lower()
    if mode not in {"prev_period", "yoy"}:
        return observations

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

    return adapted

def _fetch_series_by_req_form(
    *,
    series_id: Optional[str],
    req_form: Optional[str],
    frequency: Optional[str],
    indicator: Optional[str],
    firstdate: Optional[str],
    lastdate: Optional[str],
    calc_mode: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    req = str(req_form or "").strip().lower()
    target_frequency = str(frequency or "").upper() or None
    agg_mode = "sum" # if str(indicator or "").strip().lower() == "pib" else "avg"

    if req == "latest":
        observations = _load_series_observations(
            series_id=series_id,
            firstdate=None,
            lastdate=None,
            target_frequency=target_frequency,
            agg_mode=agg_mode,
            calc_mode=calc_mode,
        )
        latest_obs = observations[-1] if observations else None
        return observations, latest_obs

    # point/range/etc: usar rango entregado
    observations = _load_series_observations(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=target_frequency,
        agg_mode=agg_mode,
        calc_mode=calc_mode,
    )
    latest_obs = observations[-1] if observations else None

    if req in {"point", "range"} and not observations:
        fallback_obs = _load_series_observations(
            series_id=series_id,
            firstdate=None,
            lastdate=None,
            target_frequency=target_frequency,
            agg_mode=agg_mode,
            calc_mode=calc_mode,
        )
        if fallback_obs:
            latest_obs = fallback_obs[-1]

    return observations, latest_obs


def make_data_node(memory_adapter: Any):
    def data_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        question = state.get("question", "")
        session_id = state.get("session_id", "")
        entities = _clone_entities(state.get("entities"))

        classification = state.get("classification")
        predict_raw = getattr(classification, "predict_raw", None) if classification else None
        predict_raw = predict_raw if isinstance(predict_raw, dict) else {} # BORRAR !!
        
        calc_mode_cls = getattr(classification, "calc_mode", None) or {}
        activity_cls = getattr(classification, "activity", None) or {}
        region_cls = getattr(classification, "region", None) or {}
        investment_cls = getattr(classification, "investment", None) or {}
        req_form_cls = getattr(classification, "req_form", None) or {}

        entities = getattr(classification, "entities", None) or {}
        normalized = getattr(classification, "normalized", None) or {}
        
        indicator_ent = (normalized.get("indicator") or [None])[0]
        seasonality_ent = (normalized.get("seasonality") or [None])[0]
        frequency_ent = (normalized.get("frequency") or [None])[0]
        activity_ent = (normalized.get("activity") or [None])[0]
        region_ent = (normalized.get("region") or [None])[0]
        investment_ent = (normalized.get("investment") or [None])[0]
        period_ent = normalized.get("period")
        
        # REGLAS DE NEGOCIO PARA TRASLADAR !
        activity_cls_resolved = activity_cls
        activity_ent_resolved = activity_ent
        if indicator_ent == "imacec" and activity_ent is None:
            activity_cls_resolved = "specific"
            activity_ent_resolved = "imacec"
            
        price = None
        if indicator_ent == "pib" and activity_cls == "none" and region_cls == "none" and investment_cls == "none":
            price = None
        else:
            price = "enc"

        logger.info("[DATA_NODE !!!] =========================================================")
        logger.info("[DATA_NODE !!!] calc_mode=%s", calc_mode_cls)
        logger.info("[DATA_NODE !!!] activity=%s", activity_cls_resolved)
        logger.info("[DATA_NODE !!!] region=%s", region_cls)
        logger.info("[DATA_NODE !!!] investment=%s", investment_cls)
        logger.info("[DATA_NODE !!!] req_form=%s", req_form_cls)
        logger.info("[DATA_NODE !!!] entities=%s", entities)
        logger.info("[DATA_NODE !!!] normalized=%s", normalized)
        
        logger.info("[DATA_NODE !!!] indicator=%s", indicator_ent)
        logger.info("[DATA_NODE !!!] seasonality=%s", seasonality_ent)
        logger.info("[DATA_NODE !!!] frequency=%s", frequency_ent)
        logger.info("[DATA_NODE !!!] activity=%s", activity_ent_resolved)
        logger.info("[DATA_NODE !!!] region=%s", region_ent)
        logger.info("[DATA_NODE !!!] investment=%s", investment_ent)
        logger.info("[DATA_NODE !!!] price=%s", price)
        logger.info("[DATA_NODE !!!] period=%s", period_ent)
        logger.info("[DATA_NODE !!!] =========================================================")
        
        
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

        # Buscar una sola familia de series en el catalogo agrupado
        family_dict = find_family_by_classification(
            "orchestrator/catalog/catalog.json",
            indicator=indicator_ent,
            activity_value=activity_ent_resolved if activity_ent_resolved is not None else activity_cls_resolved,
            region_value=region_ent if region_ent is not None else region_cls,
            investment_value=investment_ent if investment_ent is not None else investment_cls,
            calc_mode=calc_mode_cls if calc_mode_cls == "contribution" else "original",
            price=family_price,
            seasonality=seasonality_ent,
            frequency=family_frequency,
        )
        family_series = family_to_series_rows(family_dict) if isinstance(family_dict, dict) else []
        source_family_series = family_dict.get("source_url") if isinstance(family_dict, dict) else None
        family_name = family_dict.get("family_name") if isinstance(family_dict, dict) else None
        
        logger.info(
            "[DATA_NODE !!! REFACTORING !!!] family_name=%s",
            family_name,
        )
        logger.info(
            "[DATA_NODE !!! REFACTORING !!!] family_source_url=%s",
            source_family_series,
        )
        logger.info("[DATA_NODE !!!] =========================================================")

        # Buscar serie especifica a partir de la familia de series
        series_eq = {
            "indicator": indicator_ent,
            "calc_mode": calc_mode_cls if calc_mode_cls == "contribution" else "original",
            "seasonality": seasonality_ent,
            "activity": activity_ent_resolved,
            "region": region_ent,
            "investment": investment_ent,
        }
        if calc_mode_cls == "contribution":
            series_eq["frequency"] = frequency_ent

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
            "[DATA_NODE !!! REFACTORING !!!] target_series_id=%s",
            target_series_id,
        )
        logger.info(
            "[DATA_NODE !!! REFACTORING !!!] target_series_title=%s",
            target_series_title,
        )
        logger.info(
            "[DATA_NODE !!! REFACTORING !!!] target_series_url=%s",
            target_series_url,
        )
        logger.info("[DATA_NODE !!!] =========================================================")
        
        
        ## Obtener data 
        ####################
        
        firstdate = str(period_ent[0])
        lastdate = str(period_ent[-1])
        observations: List[Dict[str, Any]] = []
        observations_all: List[Dict[str, Any]] = []
        latest_obs: Optional[Dict[str, Any]] = None
        used_latest_fallback_for_point = False
        
        if calc_mode_cls == "contribution":
            # recorrer family_series y obtener data de cada serie para luego agregar info a data_params y metadata_response
            for series in family_series:
                series_id = series.get("id")
                if not series_id:
                    continue
                series_observations, _ = _fetch_series_by_req_form(
                    series_id=series_id,
                    req_form=req_form_cls,
                    frequency=frequency_ent,
                    indicator=indicator_ent,
                    firstdate=firstdate,
                    lastdate=lastdate,
                )
                if not series_observations:
                    continue

                latest_series_obs = series_observations[-1]
                if not isinstance(latest_series_obs, dict):
                    continue

                series_title = str(
                    series.get("short_title") or series_id
                ).strip()
                series_classification = series.get("classification") if isinstance(series, dict) else None
                series_activity = None
                if isinstance(series_classification, dict):
                    series_activity = series_classification.get("activity")
                series_activity_normalized = (
                    str(series_activity).strip().lower() if series_activity not in (None, "") else None
                )

                observations.append(
                    {
                        "series_id": series_id,
                        "title": series_title,
                        "activity": series_activity_normalized,
                        "date": latest_series_obs.get("date"),
                        "value": latest_series_obs.get("value"),
                    }
                )
            observations_all = list(observations)

            if activity_cls == "general" and activity_ent is None:
                aggregate_row = next(
                    (
                        row
                        for row in observations
                        if str(row.get("title") or "").strip().lower() in {"pib", "imacec"}
                    ),
                    None,
                )
                if aggregate_row is None:
                    aggregate_row = next(
                        (
                            row
                            for row in observations
                            if row.get("activity") in (None, "", "total")
                        ),
                        None,
                    )

                if isinstance(aggregate_row, dict):
                    target_series_id = aggregate_row.get("series_id")
                    target_series_title = str(
                        aggregate_row.get("title") or family_name or ""
                    ).strip()
                    target_series_url = _build_target_series_url(
                        source_url=source_family_series,
                        series_id=target_series_id,
                        period=period_ent if isinstance(period_ent, list) else None,
                        frequency=frequency_ent,
                        calc_mode=calc_mode_cls,
                    )
                    observations = [aggregate_row]

        elif activity_cls_resolved in ("specific", "none") and region_cls in ("specific", "none") and investment_cls in ("specific", "none"):

            observations, latest_obs = _fetch_series_by_req_form(
                series_id=target_series_id,
                req_form=req_form_cls,
                frequency=frequency_ent,
                indicator=indicator_ent,
                firstdate=firstdate,
                lastdate=lastdate,
                calc_mode=calc_mode_cls if calc_mode_cls in {"prev_period", "yoy"} else "yoy",
            )
            if str(req_form_cls or "").strip().lower() == "point" and not observations and isinstance(latest_obs, dict):
                observations = [latest_obs]
                used_latest_fallback_for_point = True
        
        observed_rows_for_url = observations if observations else observations_all
        target_series_url_period = _resolve_url_period_from_data(
            requested_period=period_ent if isinstance(period_ent, list) else None,
            req_form=req_form_cls,
            observations=observed_rows_for_url,
        )
        target_series_url = _build_target_series_url(
            source_url=source_family_series,
            series_id=target_series_id,
            period=target_series_url_period,
            frequency=frequency_ent,
            calc_mode=calc_mode_cls,
        )

        logger.info("[DATA_NODE !!! REFACTORING !!!] observations_count=%s", len(observations or observations_all))
        logger.info("[DATA_NODE !!! REFACTORING !!!] observations_last_5=%s", (observations or observations_all)[-5:])
        logger.info("[DATA_NODE !!! REFACTORING !!!] target_series_url_period=%s", target_series_url_period)
        logger.info("[DATA_NODE !!! REFACTORING !!!] target_series_url_effective=%s", target_series_url)
        
        logger.info("[DATA_NODE !!!] =========================================================")
     
        if observations is not None or observations_all is not None: 
            
            payload = {
                "intent": "value",
                "classification": {
                    "indicator": indicator_ent,
                    "seasonality": seasonality_ent,
                    "frequency": frequency_ent,
                    "period": period_ent,
                    "calc_mode_cls": calc_mode_cls,
                    "activity_cls": activity_cls,
                    "region_cls": region_cls,
                    "investment_cls": investment_cls,
                    "req_form_cls": req_form_cls,
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
                    "history": None
                },
                "series": target_series_id,
                "series_title": target_series_title or family_name or None,
                "parsed_point": str(period_ent[-1]) if req_form_cls != "range" else None,
                "parsed_range": (str(period_ent[0]), str(period_ent[-1])),
                "reference_period": str(period_ent[-1] or period_ent[0]),
                "used_latest_fallback_for_point": used_latest_fallback_for_point,
                "result": observations,
                "all_series_data": observations_all or None,
                "source_url": target_series_url,
            }

            collected: List[str] = []
            try:
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
