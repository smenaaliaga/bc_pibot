"""Data node and supporting helpers for PIBot graph."""

from __future__ import annotations

import os
import logging
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from langgraph.types import StreamWriter

from orchestrator.data import flow_data

from ..state import (
    AgentState,
    _clone_entities,
    _emit_stream_chunk,
)

logger = logging.getLogger(__name__)


def _first_non_empty(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            if item not in (None, "", [], {}):
                return item
        return None
    if value in ("", [], {}):
        return None
    return value


def _period_bounds(period_value: Any) -> Tuple[Optional[str], Optional[str]]:
    if period_value is None:
        return None, None
    if isinstance(period_value, (list, tuple)):
        values = [str(item) for item in period_value if item not in (None, "")]
        if not values:
            return None, None
        if len(values) == 1:
            return values[0], values[0]
        return values[0], values[-1]
    text = str(period_value).strip()
    if not text:
        return None, None
    return text, text


def _normalize_price_value(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text or text in {"none", "null", "nan"}:
        return None
    if text in {"co", "corriente", "corrientes", "current"}:
        return "co"
    if text in {"enc", "en", "encadenado", "encadenados", "chained"}:
        return "enc"
    return text


def _infer_price_from_question(question: str) -> Optional[str]:
    normalized = unicodedata.normalize("NFD", str(question or "").lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    if "corriente" in normalized or "corrientes" in normalized:
        return "co"
    if "encadenad" in normalized:
        return "enc"
    return None


def _infer_indicator_from_question(question: str) -> Optional[str]:
    normalized = unicodedata.normalize("NFD", str(question or "").lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    has_pib = "pib" in normalized or "producto interno bruto" in normalized
    has_imacec = "imacec" in normalized
    if has_pib and not has_imacec:
        return "pib"
    if has_imacec and not has_pib:
        return "imacec"
    return None


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
    req = str(req_form or "latest").strip().lower() or "latest"
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
        
        calc_mode_cls = getattr(classification, "calc_mode", None)
        activity_cls = getattr(classification, "activity", None)
        region_cls = getattr(classification, "region", None)
        investment_cls = getattr(classification, "investment", None)
        req_form_cls = getattr(classification, "req_form", None)

        entities = getattr(classification, "entities", None) or {}
        normalized = getattr(classification, "normalized", None) or {}
        
        indicator_ent = _first_non_empty(normalized.get("indicator"))
        inferred_indicator = _infer_indicator_from_question(question)
        if inferred_indicator and str(indicator_ent or "").strip().lower() != inferred_indicator:
            indicator_ent = inferred_indicator
        seasonality_ent = _first_non_empty(normalized.get("seasonality"))
        frequency_ent = _first_non_empty(normalized.get("frequency"))
        activity_ent = _first_non_empty(normalized.get("activity"))
        region_ent = _first_non_empty(normalized.get("region"))
        investment_ent = _first_non_empty(normalized.get("investment"))
        price_ent = _normalize_price_value(_first_non_empty(normalized.get("price")))
        if price_ent is None:
            price_ent = _infer_price_from_question(question)
        period_ent = normalized.get("period")
        firstdate, lastdate = _period_bounds(period_ent)
        
        # REGLAS DE NEGOCIO PARA TRASLADAR !
        activity_cls_resolved = activity_cls
        activity_ent_resolved = activity_ent
        if indicator_ent == "imacec" and activity_ent is None:
            activity_cls_resolved = "specific"
            activity_ent_resolved = "imacec"

        calc_mode_value = str(calc_mode_cls or "").strip().lower()
        activity_cls_value = str(activity_cls_resolved or "").strip().lower()
        region_cls_value = str(region_cls or "").strip().lower()
        investment_cls_value = str(investment_cls or "").strip().lower()
        req_form_value = str(req_form_cls or "latest").strip().lower() or "latest"

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
        logger.info("[DATA_NODE !!!] price=%s", price_ent)
        logger.info("[DATA_NODE !!!] period=%s", period_ent)
        logger.info("[DATA_NODE !!!] =========================================================")
        
        
        ## Obtener ID Series
        #####################
        
        from orchestrator.catalog.series_search import (
            family_to_series_rows,
            find_family_by_classification,
            select_target_series_by_classification,
        )

        # Buscar una sola familia de series en el catalogo agrupado
        family_dict = find_family_by_classification(
            "orchestrator/catalog/catalog.json",
            indicator=indicator_ent,
            activity_value=activity_ent_resolved,
            region_value=region_ent,
            investment_value=investment_ent,
            calc_mode=calc_mode_value if calc_mode_value == "contribution" else "original",
            seasonality=seasonality_ent,
            price=price_ent,
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
        target_series = select_target_series_by_classification(
            family_series,
            eq={
                "indicator": indicator_ent,
                "calc_mode": calc_mode_value if calc_mode_value == "contribution" else "original",
                "seasonality": seasonality_ent,
                "frequency": frequency_ent,
                "activity": activity_ent_resolved,
                "region": region_ent,
                "investment": investment_ent,
                "price": price_ent,
            },
            fallback_to_first=True,
        )
        
        target_series_id = target_series.get("id") if isinstance(target_series, dict) else None
        target_series_title = target_series.get("title") if isinstance(target_series, dict) else None
        target_series_url = None
        if source_family_series and target_series_id:
            separator = "&" if "?" in str(source_family_series) else "?"
            target_series_url = (
                f"{source_family_series}{separator}id5=SI&idSerie={target_series_id}"
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
        
        observations: List[Dict[str, Any]] = []
        observations_all: List[Dict[str, Any]] = []
        latest_obs: Optional[Dict[str, Any]] = None
        
        if activity_cls_value in {"", "specific", "none"} and region_cls_value in {"", "specific", "none"} and investment_cls_value in {"", "specific", "none"}:
        
            observations, latest_obs = _fetch_series_by_req_form(
                series_id=target_series_id,
                req_form=req_form_value,
                frequency=frequency_ent,
                indicator=indicator_ent,
                firstdate=firstdate,
                lastdate=lastdate,
                calc_mode=calc_mode_value if calc_mode_value in {"prev_period", "yoy"} else "yoy",
            )
            
        elif calc_mode_value == "contribution":
            # recorrer family_series y obtener data de cada serie para luego agregar info a data_params y metadata_response
            for series in family_series:
                series_id = series.get("id")
                if not series_id:
                    continue
                series_observations, _ = _fetch_series_by_req_form(
                    series_id=series_id,
                    req_form=req_form_value,
                    frequency=frequency_ent,
                    indicator=indicator_ent,
                    firstdate=firstdate,
                    lastdate=lastdate,
                )
                if not series_observations:
                    continue

                latest_series_obs = series_observations[-1]

                series_title = str(series.get("title") or series_id).strip()
                observations.append(
                    {
                        "series_id": series_id,
                        "title": series_title,
                        "activity": None,
                        "date": latest_series_obs.get("date"),
                        "value": latest_series_obs.get("value"),
                    }
                )
                observations_all.extend(series_observations)
        else:
            observations, latest_obs = _fetch_series_by_req_form(
                series_id=target_series_id,
                req_form=req_form_value,
                frequency=frequency_ent,
                indicator=indicator_ent,
                firstdate=firstdate,
                lastdate=lastdate,
                calc_mode=calc_mode_value if calc_mode_value in {"prev_period", "yoy"} else "yoy",
            )

        observations_payload = observations or observations_all
        logger.info("[DATA_NODE !!! REFACTORING !!!] observations_count=%s", len(observations_payload))
        logger.info("[DATA_NODE !!! REFACTORING !!!] observations_last_5=%s", observations_payload[-5:] if observations_payload else [])
        logger.info("[DATA_NODE !!!] =========================================================")
     
        if observations is not None or observations_all is not None: 
            parsed_range = (firstdate, lastdate) if firstdate and lastdate else None
            reference_period = lastdate or firstdate
            
            payload = {
                "intent": "value",
                "question": question,
                "history_text": state.get("history_text"),
                "classification": {
                    "indicator": indicator_ent,
                    "seasonality": seasonality_ent,
                    "frequency": frequency_ent,
                    "period": period_ent,
                    "calc_mode_cls": calc_mode_value,
                    "activity_cls": activity_cls_value,
                    "region_cls": region_cls_value,
                    "investment_cls": investment_cls_value,
                    "req_form_cls": req_form_value,
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
                    "price": price_ent,
                    "history": None
                },
                "series": target_series_id,
                "series_title": family_name or None,
                "parsed_point": reference_period if req_form_value != "range" else None,
                "parsed_range": parsed_range,
                "reference_period": reference_period,
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
