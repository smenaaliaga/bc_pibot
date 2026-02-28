"""Data node and supporting helpers for PIBot graph."""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import data
from langgraph.types import StreamWriter

from orchestrator.data import flow_data

from ..state import (
    AgentState,
    _clone_entities,
    _emit_stream_chunk,
    _ensure_entity_slot,
)

logger = logging.getLogger(__name__)

_METADATA_LOOKUP: Optional[Dict[str, Dict[str, Any]]] = None


def _empty_to_none_token(value: Any) -> Any:
    if value in (None, "", [], {}):
        return "none"
    return value


def _first_or_none(value: Any) -> Any:
    if isinstance(value, list):
        return next((item for item in value if item not in (None, "")), None)
    return value


def _extract_label(payload: Dict[str, Any], key: str, fallback: Any = None) -> Any:
    value = payload.get(key)
    if isinstance(value, dict):
        return value.get("label", fallback)
    if value not in (None, ""):
        return value
    return fallback


def _extract_predict_sections(
    predict_raw: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    # Normaliza el payload del clasificador para soportar dos formatos:
    # 1) datos bajo predict_raw["interpretation"], 2) datos en la raíz.
    interpretation_root = (
        predict_raw.get("interpretation")
        if isinstance(predict_raw.get("interpretation"), dict)
        else predict_raw
    )
    interpretation_root = interpretation_root if isinstance(interpretation_root, dict) else {}

    interpretation_intents = (
        interpretation_root.get("intents")
        if isinstance(interpretation_root.get("intents"), dict)
        else {}
    )
    entities_normalized = (
        interpretation_root.get("entities_normalized")
        if isinstance(interpretation_root.get("entities_normalized"), dict)
        else {}
    )
    routing = predict_raw.get("routing") if isinstance(predict_raw.get("routing"), dict) else {}

    return interpretation_root, interpretation_intents, entities_normalized, routing


def _metadata_key_from_params(data_params: Dict[str, Any]) -> str:
    # Orden estable de campos para construir la key de lookup en metadata_q.json.
    order = [
        "activity_cls",
        "frequency", # frequency no define a una serie especifica
        "calc_mode_cls", # no deberia identificar una serie específica
        "region_cls",
        "investment_cls",
        "req_form_cls", # no deberia identificar una serie específica
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
    return "::".join(str(data_params.get(key)) for key in order)


def _load_metadata_lookup() -> Dict[str, Dict[str, Any]]:
    global _METADATA_LOOKUP
    if _METADATA_LOOKUP is not None:
        # Cache en memoria para no releer el archivo en cada request.
        return _METADATA_LOOKUP

    metadata_path = Path(__file__).resolve().parents[2] / "catalog" / "metadata_q.json"
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries = data.get("data") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            _METADATA_LOOKUP = {}
            return _METADATA_LOOKUP
        _METADATA_LOOKUP = {
            entry.get("key"): entry
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("key"), str)
        }
    except Exception:
        logger.exception("[DATA_NODE] Failed to load metadata_q.json")
        _METADATA_LOOKUP = {}
    return _METADATA_LOOKUP


def _build_metadata_response(data_params: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _load_metadata_lookup()
    if "__error__" in metadata:
        return {"error": metadata["__error__"]}
    key = _metadata_key_from_params(data_params)
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
                candidate_key = _metadata_key_from_params(params_fallback)
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

def _build_all_series_data(
    *,
    metadata_series: Any,
    period: Any,
    target_frequency: Optional[str],
    calc_mode_value: str,
    req_form_value: str,
) -> List[Dict[str, Any]]:
    if not isinstance(metadata_series, dict) or not metadata_series:
        return []

    from orchestrator.data.get_data_serie import get_series_from_redis

    firstdate: Optional[str] = None
    lastdate: Optional[str] = None
    if isinstance(period, list) and period:
        firstdate = str(period[0])
        lastdate = str(period[-1])

    rows: List[Dict[str, Any]] = []
    for serie_info in metadata_series.values():
        if not isinstance(serie_info, dict):
            continue
        serie_id = str(serie_info.get("id") or "").strip()
        if not serie_id or serie_id.lower() == "none":
            continue
        serie_title = str(serie_info.get("title") or serie_id).strip()

        agg_mode = "avg"
        if str(target_frequency or "").lower() == "a" and ".FLU." in serie_id.upper():
            agg_mode = "sum"

        try:
            series_data = get_series_from_redis(
                series_id=serie_id,
                firstdate=firstdate,
                lastdate=lastdate,
                target_frequency=target_frequency,
                agg=agg_mode,
                use_fallback=True,
            )
            observations = (series_data or {}).get("observations") or []

            if req_form_value == "latest" and not observations:
                fallback_data = get_series_from_redis(
                    series_id=serie_id,
                    firstdate=None,
                    lastdate=None,
                    target_frequency=target_frequency,
                    agg=agg_mode,
                    use_fallback=True,
                )
                observations = (fallback_data or {}).get("observations") or []

            observations = [obs for obs in observations if isinstance(obs, dict)]
            if not observations:
                continue

            chosen_obs = observations[-1]

            calc_mode_norm = str(calc_mode_value or "").strip().lower()
            if calc_mode_norm == "contribution":
                metric_value = chosen_obs.get("value")
            elif calc_mode_norm == "prev_period":
                metric_value = chosen_obs.get("pct")
                if metric_value is None:
                    metric_value = chosen_obs.get("yoy_pct")
                if metric_value is None:
                    metric_value = chosen_obs.get("value")
            else:
                metric_value = chosen_obs.get("yoy_pct")
                if metric_value is None:
                    metric_value = chosen_obs.get("pct")
                if metric_value is None:
                    metric_value = chosen_obs.get("value")

            rows.append(
                {
                    "series_id": serie_id,
                    "title": serie_title,
                    "activity": "",
                    "date": chosen_obs.get("date"),
                    "value": metric_value,
                }
            )
        except Exception:
            logger.exception(
                "[=====DATA_NODE=====] Error obteniendo serie de contribución | serie_id=%s",
                serie_id,
            )

    return rows


def make_data_node(memory_adapter: Any):
    def data_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        # Contexto base del turno y entidad principal del estado conversacional.
        session_id = state.get("session_id", "")
        entities = _clone_entities(state.get("entities"))
        primary_entity = _ensure_entity_slot(entities, 0)

        classification = state.get("classification")
        intent_payload = state.get("intent") if isinstance(state.get("intent"), dict) else {}
        predict_raw = getattr(classification, "predict_raw", None) if classification else None
        predict_raw = predict_raw if isinstance(predict_raw, dict) else {}

        _, interpretation_intents, entities_normalized, routing = _extract_predict_sections(
            predict_raw
        )

        # Prioridad de extracción: entidades normalizadas -> estado previo (primary_entity).
        indicator_candidate = (
            _first_or_none(entities_normalized.get("indicator"))
            or primary_entity.get("indicator")
        )
        seasonality_candidate = (
            _first_or_none(entities_normalized.get("seasonality"))
            or primary_entity.get("seasonality")
        )
        frequency_candidate = (
            _first_or_none(entities_normalized.get("frequency"))
            or primary_entity.get("frequency")
        )
        activity_candidate = (
            _first_or_none(entities_normalized.get("activity"))
            or primary_entity.get("activity")
        )
        region_candidate = (
            _first_or_none(entities_normalized.get("region"))
            or primary_entity.get("region")
        )
        period_norm = entities_normalized.get("period")
        if isinstance(period_norm, list) and period_norm:
            period_candidate = period_norm
        else:
            period_candidate = primary_entity.get("period")

        calc_mode_cls = _extract_label(interpretation_intents, "calc_mode")
        activity_cls = _extract_label(interpretation_intents, "activity")
        region_cls = _extract_label(interpretation_intents, "region")
        investment_cls = _extract_label(interpretation_intents, "investment")
        req_form_cls = _extract_label(interpretation_intents, "req_form")
        intent_cls = _extract_label(routing, "intent", intent_payload.get("intent_cls"))
        

        data_params = {
            "indicator": indicator_candidate,
            "seasonality": seasonality_candidate,
            "frequency": frequency_candidate,
            "period": period_candidate,
            "calc_mode_cls": calc_mode_cls,
            "activity_cls": activity_cls,
            "region_cls": region_cls,
            "investment_cls": investment_cls,
            "req_form_cls": req_form_cls,
            "intent_cls": intent_cls,
            "activity_value": activity_candidate,
            "sub_activity_value": predict_raw.get("sub_activity_value"),
            "region_value": region_candidate,
            "investment_value": predict_raw.get("investment_value"),
            "gasto_value": predict_raw.get("gasto_value"),
            "price": _first_or_none(entities_normalized.get("price")) or primary_entity.get("price") or "co",
            "history": _first_or_none(entities_normalized.get("history")) or primary_entity.get("history") or "2018",
        }
        # Normaliza vacíos para construir metadata key consistente con catálogo.
        data_params = {key: _empty_to_none_token(value) for key, value in data_params.items()}

        metadata_response = _build_metadata_response(data_params)
        
        metadata_key = metadata_response.get("key") if isinstance(metadata_response, dict) else None
        if isinstance(metadata_response, dict):
            logger.info(
                "[DATA_NODE] metadata_summary key=%s serie=%s source=%s latest_update=%s",
                metadata_key,
                metadata_response.get("serie_default"),
                "PRESENT" if metadata_response.get("sources_url") else "MISSING",
                metadata_response.get("latest_update"),
            )

        data_params_status = {
            key: "PRESENT" if value not in (None, "", [], {}) else "MISSING"
            for key, value in data_params.items()
        }
        logger.info(
            "[DATA_NODE] predict_payload_status intents=%s entities_normalized=%s indicator=%s req_form=%s calc_mode=%s",
            "PRESENT" if interpretation_intents else "MISSING",
            "PRESENT" if entities_normalized else "MISSING",
            data_params.get("indicator"),
            data_params.get("req_form_cls"),
            data_params.get("calc_mode_cls"),
        )

        source_url: Optional[Any] = None
        if isinstance(metadata_response, dict):
            sources_url = metadata_response.get("sources_url")
            if sources_url:
                source_url = sources_url

        series_fetch_result = None
        serie_default = metadata_response.get("serie_default") if isinstance(metadata_response, dict) else None
        payload_series = serie_default
        reference_period: Optional[str] = None
        all_series_data: List[Dict[str, Any]] = []
        has_metadata_series = serie_default is not None and str(serie_default).strip() != ""
        has_specific_series = has_metadata_series and str(serie_default).strip().lower() != "none"
        
        
        if has_specific_series:
            try:
                from orchestrator.data.get_data_serie import get_series_from_redis

                period = data_params.get("period")
                if isinstance(period, list) and period:
                    firstdate = str(period[0])
                    lastdate = str(period[-1])
                    reference_period = str(period[-1] or period[0])
                else:
                    firstdate = None
                    lastdate = None

                req_form_value = str(data_params.get("req_form_cls") or "").lower()
                target_frequency = str(data_params.get("frequency") or "").lower() or None
                
                metadata_series = metadata_response.get("series") if isinstance(metadata_response, dict) else None
                if calc_mode_cls == "contribution" and isinstance(metadata_series, dict) and metadata_series:
                    all_series_data = _build_all_series_data(
                        metadata_series=metadata_series,
                        period=period,
                        target_frequency=target_frequency,
                        calc_mode_value=calc_mode_cls,
                        req_form_value=str(req_form_value or "").lower(),
                    )

                if (
                    (not payload_series or str(payload_series).strip().lower() == "none")
                    and all_series_data
                ):
                    payload_series = all_series_data[0].get("series_id")
                
                # Obtener valores de la serie
                series_data = get_series_from_redis(
                    series_id=serie_default,
                    firstdate=firstdate,
                    lastdate=lastdate,
                    target_frequency=target_frequency,
                    req_form=req_form_value,
                    use_fallback=True,
                )

                observations_log = (series_data or {}).get("observations") or []
                logger.info(
                    "[DATA_NODE] series_data_observations rows=%s values=%s",
                    len(observations_log),
                    observations_log,
                )
                
                # Obtener valores
                observations = (series_data or {}).get("observations") or []

                latest_obs = observations[-1] if observations else None
                
                series_fetch_result = {
                    "rows": len(observations),
                    "latest": latest_obs,
                }
            except Exception as exc:
                series_fetch_result = {"error": str(exc)}

        try:
            logs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "logs")
            os.makedirs(logs_dir, exist_ok=True)
            fixed = os.getenv("RUN_MAIN_LOG", "").strip() or "run_main.log"
            log_file_path = os.path.join(logs_dir, fixed)
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write("[CLASSIFIER_FILE] METADATA_RESPONSE=%s\n" % metadata_response)
        except Exception:
            pass
        
        def _build_trace_info(payload_obj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            return {
                "parsed_point": payload_obj.get("parsed_point") if isinstance(payload_obj, dict) else None,
                "parsed_range": payload_obj.get("parsed_range") if isinstance(payload_obj, dict) else None,
                "series": payload_obj.get("series") if isinstance(payload_obj, dict) else None,
                "data_classification": payload_obj.get("classification") if isinstance(payload_obj, dict) else None,
                "data_params": data_params,
                "data_params_status": data_params_status,
                "metadata_response": metadata_response,
                "metadata_key": metadata_key,
                "series_fetch_result": series_fetch_result,
            }

        # Camino principal: metadata disponible + serie obtenida (o error controlado)
        latest_obs = (series_fetch_result or {}).get("latest") if isinstance(series_fetch_result, dict) else None
        if has_metadata_series:
            period = data_params.get("period")
            period_list = period if isinstance(period, list) else []

            latest_obs_dict = latest_obs if isinstance(latest_obs, dict) else {}

            output_dict: Dict[str, Any] = {
                "date": latest_obs_dict.get("date"),
                "value": latest_obs_dict.get("value"),
                "prev_period": latest_obs_dict.get("pct") if latest_obs_dict.get("pct") is not None else None,
                "yoy": latest_obs_dict.get("yoy_pct") if latest_obs_dict.get("yoy_pct") is not None else None,
                "serie": serie_default,
            }

            payload = {
                "intent": data_params.get("intent_cls") or "value",
                "classification": {
                    "indicator": indicator_candidate,
                    "metric_type": calc_mode_cls,
                    "seasonality": seasonality_candidate,
                    "activity": activity_candidate,
                    "frequency": frequency_candidate,
                    "req_form": req_form_cls,
                    "price": data_params.get("price"),
                    "history": data_params.get("history"),
                },
                "series": payload_series,
                "series_title": metadata_response.get("title_serie_default") if isinstance(metadata_response, dict) else None,
                "parsed_point": period_list[-1] if period_list else None,
                "parsed_range": period_list if period_list else None,
                "reference_period": reference_period,
                "result": output_dict,
                "all_series_data": all_series_data or None,
                "source_url": source_url,
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

            trace_info = _build_trace_info(payload)
            return {"output": "".join(collected), "entities": entities, **trace_info}

        text = "[GRAPH] No se recibió clasificación para el nodo DATA."
        logger.warning("[=====DATA_NODE=====] %s", text)
        _emit_stream_chunk(text, writer)
        return {
            "output": text,
            "entities": entities,
            "data_params": data_params,
            "data_params_status": data_params_status,
            "metadata_response": metadata_response,
            "metadata_key": metadata_key,
            "series_fetch_result": series_fetch_result,
        }
        
    return data_node
      
__all__ = ["make_data_node"]
