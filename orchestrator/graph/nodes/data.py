"""Data node and supporting helpers for PIBot graph."""

from __future__ import annotations

import datetime
import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langgraph.types import StreamWriter

from orchestrator.classifier.classifier_agent import predict_with_interpreter
from orchestrator.data import flow_data

from ..state import (
    AgentState,
    _build_timeseries_map,
    _clone_entities,
    _emit_stream_chunk,
    _ensure_entity_slot,
    _merge_entity_fields,
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
    # Convierte data_params en una respuesta de metadata compacta para trazas/payload.
    key = _metadata_key_from_params(data_params)
    entry = _load_metadata_lookup().get(key)
    if not isinstance(entry, dict):
        return {"key": key, "match": None}
    return {
        "key": key,
        "label": entry.get("label"),
        "serie_default": entry.get("serie_default"),
        "title_serie_default": entry.get("title_serie_default"),
        "sources_url": entry.get("sources_url"),
        "latest_update": entry.get("latest_update"),
    }


def _normalize_region_value(value: Any) -> Optional[str]:
    # Estandariza región a string en minúsculas; vacíos quedan como None real.
    if value in (None, "", [], {}):
        return None
    if isinstance(value, list):
        first = next((item for item in value if item not in (None, "")), None)
        return str(first).strip().lower() if first not in (None, "") else None
    return str(value).strip().lower() or None


def _normalize_calc_mode(
    calc_mode_label: Optional[str],
) -> str:
    # Restringe modos válidos para cálculo de variación; por defecto prev_period.
    raw = str(calc_mode_label or "").strip().lower()
    if raw in {"prev_period", "yoy"}:
        return raw
    return "prev_period"


def make_data_node(memory_adapter: Any):
    def data_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        # Contexto base del turno y entidad principal del estado conversacional.
        question = state.get("question", "")
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
        period_candidate = (
            _first_or_none(entities_normalized.get("period"))
            or primary_entity.get("period")
        )
        

        data_params = {
            "indicator": indicator_candidate,
            "seasonality": seasonality_candidate,
            "frequency": frequency_candidate,
            "period": period_candidate,
            "calc_mode_cls": _extract_label(interpretation_intents, "calc_mode"),
            "activity_cls": _extract_label(interpretation_intents, "activity"),
            "region_cls": _extract_label(interpretation_intents, "region"),
            "investment_cls": _extract_label(interpretation_intents, "investment"),
            "req_form_cls": _extract_label(interpretation_intents, "req_form"),
            "macro_cls": _extract_label(routing, "macro", intent_payload.get("macro_cls")),
            "intent_cls": _extract_label(routing, "intent", intent_payload.get("intent_cls")),
            "context_cls": _extract_label(routing, "context", intent_payload.get("context_cls")),
            "enable": predict_raw.get("enable"),
            "enable_all": predict_raw.get("enable_all"),
            "activity_value": activity_candidate,
            "sub_activity_value": predict_raw.get("sub_activity_value"),
            "region_value": _normalize_region_value(region_candidate),
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

        series_fetch_args = None
        series_fetch_result = None
        series_fetch_observations: List[Dict[str, Any]] = []
        serie_default = metadata_response.get("serie_default") if isinstance(metadata_response, dict) else None
        has_metadata_series = serie_default is not None and str(serie_default).strip() != ""
        has_specific_series = has_metadata_series and str(serie_default).strip().lower() != "none"
        if has_specific_series:
            try:
                from orchestrator.data.get_data_serie import get_series_from_redis

                period = data_params.get("period")
                if isinstance(period, list) and period:
                    lastdate = period[-1]
                    firstdate = period[0]
                else:
                    lastdate = None
                    firstdate = None

                req_form_value = str(data_params.get("req_form_cls") or "").lower()
                target_frequency = str(data_params.get("frequency") or "").lower() or None
                raw_series_id = str(serie_default or "").strip().upper()
                series_native_frequency = None
                if raw_series_id:
                    token = raw_series_id.split(".")[-1]
                    series_native_frequency = {
                        "M": "m",
                        "T": "q",
                        "Q": "q",
                        "A": "a",
                        "D": "d",
                    }.get(token)
                if series_native_frequency and (
                    req_form_value in {"point", "range"}
                    or target_frequency != series_native_frequency
                ):
                    target_frequency = series_native_frequency

                if target_frequency == "a":
                    reference = str(lastdate or firstdate or "").strip()
                    if len(reference) >= 4 and reference[:4].isdigit():
                        year = reference[:4]
                        firstdate, lastdate = f"{year}-01-01", f"{year}-12-31"
                series_fetch_args = {
                    "series_id": serie_default,
                    "firstdate": firstdate,
                    "lastdate": lastdate,
                    "target_frequency": target_frequency,
                    "agg": "avg",
                }
                series_data = get_series_from_redis(
                    series_id=serie_default,
                    firstdate=firstdate,
                    lastdate=lastdate,
                    target_frequency=target_frequency,
                    agg="avg",
                    use_fallback=True,
                )
                observations = (series_data or {}).get("observations") or []
                if (firstdate or lastdate) and observations:
                    filtered_observations: List[Dict[str, Any]] = []
                    for obs in observations:
                        if not isinstance(obs, dict):
                            continue
                        obs_date = str(obs.get("date") or "")
                        if not obs_date:
                            continue
                        if firstdate and obs_date < str(firstdate):
                            continue
                        if lastdate and obs_date > str(lastdate):
                            continue
                        filtered_observations.append(obs)
                    observations = filtered_observations

                if req_form_value == "latest" and not observations:
                    # Fallback: para latest, si no hay observaciones acotadas,
                    # consulta sin rango y usa el último dato disponible.
                    fallback_data = get_series_from_redis(
                        series_id=serie_default,
                        firstdate=None,
                        lastdate=None,
                        target_frequency=target_frequency,
                        agg="avg",
                        use_fallback=True,
                    )
                    fallback_obs = (fallback_data or {}).get("observations") or []
                    observations = [obs for obs in fallback_obs if isinstance(obs, dict)]
                    if isinstance(series_fetch_args, dict):
                        series_fetch_args["firstdate"] = None
                        series_fetch_args["lastdate"] = None
                        series_fetch_args["fallback_unbounded"] = True

                series_fetch_observations = observations
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

        result = predict_with_interpreter(question)

        logger.info(
            "[=====DATA_NODE=====] PIBOT_SERIES_INTERPRETER | %s",
            vars(result) if hasattr(result, "__dict__") else result,
        )

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
                "series_fetch_args": series_fetch_args,
                "series_fetch_result": series_fetch_result,
            }

        if not result:
            # Camino de fallback cuando el intérprete legacy no devuelve resultado,
            # pero sí existe metadata suficiente para responder.
            serie_default = metadata_response.get("serie_default") if isinstance(metadata_response, dict) else None
            has_metadata_series = serie_default is not None and str(serie_default).strip() != ""
            has_specific_series = has_metadata_series and str(serie_default).strip().lower() != "none"
            latest_obs = (series_fetch_result or {}).get("latest") if isinstance(series_fetch_result, dict) else None
            if has_metadata_series:
                req_form_value = data_params.get("req_form_cls") or "latest"
                period = data_params.get("period")
                parsed_point = None
                parsed_range = None
                if req_form_value == "range" and isinstance(period, list) and period:
                    parsed_range = (str(period[0]), str(period[-1]))
                if req_form_value == "point" and isinstance(period, list) and period:
                    parsed_point = str(period[-1])

                effective_req_form = req_form_value

                output_dict: Dict[str, Any] = {
                    "date": None,
                    "value": None,
                    "serie": serie_default,
                }
                calc_mode_value = str(data_params.get("calc_mode_cls") or "").lower()
                effective_frequency = str(data_params.get("frequency") or "").lower() or None
                if isinstance(series_fetch_args, dict):
                    args_frequency = series_fetch_args.get("target_frequency")
                    if args_frequency:
                        effective_frequency = str(args_frequency).lower()
                range_rows: List[Dict[str, Any]] = []
                if has_specific_series and isinstance(latest_obs, dict):
                    output_dict["date"] = latest_obs.get("date")
                    output_dict["value"] = latest_obs.get("value")
                    if calc_mode_value == "prev_period":
                        if latest_obs.get("pct") is not None:
                            output_dict["prev_period"] = latest_obs.get("pct")
                        elif latest_obs.get("yoy_pct") is not None:
                            output_dict["yoy"] = latest_obs.get("yoy_pct")
                    else:
                        if latest_obs.get("yoy_pct") is not None:
                            output_dict["yoy"] = latest_obs.get("yoy_pct")
                        elif latest_obs.get("pct") is not None:
                            output_dict["prev_period"] = latest_obs.get("pct")

                if has_specific_series and effective_req_form == "range":
                    for obs in series_fetch_observations:
                        if not isinstance(obs, dict):
                            continue
                        date_val = obs.get("date")
                        value_val = obs.get("value")
                        if date_val in (None, ""):
                            continue
                        row: Dict[str, Any] = {
                            "date": str(date_val),
                            "value": value_val,
                        }
                        if calc_mode_value == "prev_period":
                            if obs.get("pct") is not None:
                                row["prev_period"] = obs.get("pct")
                        else:
                            if obs.get("yoy_pct") is not None:
                                row["yoy"] = obs.get("yoy_pct")
                            elif obs.get("pct") is not None:
                                row["prev_period"] = obs.get("pct")
                        range_rows.append(row)

                payload = {
                    "intent": data_params.get("intent_cls") or "value",
                    "classification": {
                        "indicator": data_params.get("indicator"),
                        "metric_type": data_params.get("metric_type") or "index",
                        "seasonality": data_params.get("seasonality"),
                        "activity": data_params.get("activity_value"),
                        "frequency": effective_frequency or data_params.get("frequency"),
                        "calc_mode": data_params.get("calc_mode_cls"),
                        "req_form": effective_req_form,
                        "calc_mode_cls": data_params.get("calc_mode_cls"),
                        "frequency_cls": effective_frequency or data_params.get("frequency"),
                        "activity_cls": data_params.get("activity_cls"),
                        "region_cls": data_params.get("region_cls"),
                        "investment_cls": data_params.get("investment_cls"),
                        "req_form_cls": req_form_value,
                        "price": data_params.get("price"),
                        "history": data_params.get("history"),
                    },
                    "series": serie_default,
                    "series_title": metadata_response.get("title_serie_default") if isinstance(metadata_response, dict) else None,
                    "parsed_point": parsed_point if effective_req_form != "range" else None,
                    "parsed_range": parsed_range,
                    "result": range_rows if effective_req_form == "range" else output_dict,
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
                "series_fetch_args": series_fetch_args,
                "series_fetch_result": series_fetch_result,
            }

        legacy_indicator = getattr(getattr(result, "indicator", None), "label", None)
        legacy_activity = getattr(getattr(result, "activity", None), "label", None)
        legacy_seasonality = getattr(getattr(result, "seasonality", None), "label", None)
        legacy_metric_type = getattr(getattr(result, "metric_type", None), "label", None)

        calc_mode_cls = (
            getattr(getattr(result, "calc_mode_cls", None), "label", None)
            or getattr(getattr(result, "calc_mode", None), "label", None)
        )
        req_form_cls = (
            getattr(getattr(result, "req_form_cls", None), "label", None)
            or getattr(getattr(result, "req_form", None), "label", None)
        )
        frequency_cls = (
            getattr(getattr(result, "frequency_cls", None), "label", None)
            or getattr(getattr(result, "frequency", None), "label", None)
        )
        activity_cls = (
            getattr(getattr(result, "activity_cls", None), "label", None)
            or (legacy_activity if legacy_activity in {"general", "specific", "none"} else None)
        )
        region_cls = (
            getattr(getattr(result, "region_cls", None), "label", None)
            or getattr(getattr(result, "region", None), "label", None)
        )

        _merge_entity_fields(
            primary_entity,
            {
                "indicator": legacy_indicator,
                "activity": legacy_activity if legacy_activity not in {"general", "specific", "none"} else None,
                "seasonality": legacy_seasonality,
            },
            overwrite=True,
        )
        _merge_entity_fields(
            primary_entity,
            {
                "metric_type_cls": legacy_metric_type,
                "calc_mode_cls": calc_mode_cls,
                "req_form_cls": req_form_cls,
                "frequency_cls": frequency_cls,
                "activity_cls": activity_cls,
                "region_cls": region_cls,
            },
            overwrite=True,
        )
        # En este punto primary_entity queda enriquecida y sirve como fuente
        # estable para selección de serie y contexto de los próximos nodos.

        collected: List[str] = []

        from orchestrator.data.date_parser import parse_point_date, parse_range
        from registry import get_bde_client, get_catalog_service

        parsed_point: Optional[str] = None
        parsed_range: Optional[Tuple[str, str]] = None

        req_form_label = getattr(getattr(result, "req_form", None), "label", None)
        req_form_cls_label = getattr(getattr(result, "req_form_cls", None), "label", None)
        frequency_label = getattr(getattr(result, "frequency", None), "label", None) or frequency_cls

        if not req_form_label:
            if req_form_cls_label == "specific":
                parsed_range = parse_range(question, frequency_label)
                if parsed_range:
                    req_form_label = "range"
                else:
                    parsed_point = parse_point_date(question, frequency_label)
                    req_form_label = "point" if parsed_point else "latest"
            elif req_form_cls_label == "general":
                req_form_label = "latest"
            else:
                req_form_label = "latest"

        if req_form_label == "point":
            parsed_point = parsed_point or parse_point_date(question, frequency_label)
            if not parsed_point:
                logger.warning(
                    "[=====DATA_NODE=====] parse_point_date falló | question=%s | frequency=%s",
                    question,
                    frequency_label,
                )
            logger.info(
                "[=====DATA_NODE=====] parse_point_date exitoso | parsed_point=%s | frequency=%s",
                parsed_point,
                frequency_label,
            )
        elif req_form_label == "range":
            parsed_range = parsed_range or parse_range(question, frequency_label)
            if not parsed_range:
                logger.warning(
                    "[=====DATA_NODE=====] parse_range falló | question=%s | frequency=%s",
                    question,
                    frequency_label,
                )
            logger.info(
                "[=====DATA_NODE=====] parse_range exitoso | parsed_range=%s | frequency=%s",
                parsed_range,
                frequency_label,
            )

        catalog = get_catalog_service(catalog_path="catalog/series_catalog.json")
        bde = get_bde_client()

        indicator_label = legacy_indicator or primary_entity.get("indicator")
        metric_type_label = legacy_metric_type
        if not metric_type_label and calc_mode_cls == "contribution":
            metric_type_label = "contribution"
        if not metric_type_label:
            metric_type_label = "index"
        seasonality_label = legacy_seasonality or primary_entity.get("seasonality")
        activity_label = primary_entity.get("activity") or legacy_activity
        if not activity_label:
            if activity_cls in {"general", "none"}:
                activity_label = "total"
        frequency_label = frequency_label or "m"

        match = catalog.find_series(
            indicator=indicator_label,
            metric_type=metric_type_label,
            seasonality=seasonality_label,
            activity=activity_label,
            frequency=frequency_label,
        )

        logger.info(
            "[=====DATA_NODE=====] catalog.find_series | match=%s",
            match if match else "(no encontrado)",
        )

        from orchestrator.data.common import normalize_series_obs

        obs = bde.fetch_series(match["id"])
        normalized = normalize_series_obs(obs, frequency_label)
        timeseries_map = _build_timeseries_map(normalized)
        if timeseries_map:
            primary_entity["timeseries"] = timeseries_map

        from orchestrator.data.variations import (
            compute_variations_for_range,
            prev_period as compute_prev,
            yoy as compute_yoy,
        )

        payload: Dict[str, Any]
        calc_mode_label = getattr(getattr(result, "calc_mode", None), "label", None) or calc_mode_cls
        calc_mode_label = _normalize_calc_mode(calc_mode_label)

        def _legacy_classification_payload(req_form: str) -> Dict[str, Any]:
            return {
                "indicator": indicator_label,
                "metric_type": metric_type_label,
                "seasonality": seasonality_label,
                "activity": activity_label,
                "frequency": frequency_label,
                "calc_mode": calc_mode_label,
                "req_form": req_form,
                "calc_mode_cls": calc_mode_cls,
                "frequency_cls": frequency_cls,
                "activity_cls": activity_cls,
                "region_cls": region_cls,
                "req_form_cls": req_form_cls,
            }

        if req_form_label == "latest":
            d, v = normalized[-1]
            output_dict = {"date": d.strftime("%d-%m-%Y"), "value": v}
            if calc_mode_label == "yoy":
                var = compute_yoy(normalized, d, v)
                if var is not None:
                    output_dict["yoy"] = var
            elif calc_mode_label == "prev_period":
                var = compute_prev(normalized, d, v)
                if var is not None:
                    output_dict["prev_period"] = var
            payload = {
                "intent": "value",
                "classification": _legacy_classification_payload(req_form_label),
                "series": match["id"],
                "parsed_point": None,
                "parsed_range": None,
                "result": output_dict,
                "source_url": source_url,
            }
            logger.info(f"[=====DATA_NODE=====] Result (latest): {payload}")

        elif req_form_label == "point":
            target_dt = datetime.datetime.strptime(parsed_point, "%d-%m-%Y")  # type: ignore[arg-type]
            d_target = None
            for d, v in normalized:
                if d == target_dt:
                    d_target = d
                    v_target = v
                    break
            if d_target is None:
                return {"error": f"No existe observación para la fecha {parsed_point}.", "entities": entities}
            output_dict = {"date": d_target.strftime("%d-%m-%Y"), "value": v_target}
            if calc_mode_label == "yoy":
                var = compute_yoy(normalized, d_target, v_target)
                if var is not None:
                    output_dict["yoy"] = var
            elif calc_mode_label == "prev_period":
                var = compute_prev(normalized, d_target, v_target)
                if var is not None:
                    output_dict["prev_period"] = var
            payload = {
                "intent": "value",
                "classification": _legacy_classification_payload(req_form_label),
                "series": match["id"],
                "parsed_point": parsed_point,
                "parsed_range": None,
                "result": output_dict,
                "source_url": source_url,
            }
            logger.info(f"[=====DATA_NODE=====] Result (point): {payload}")

        else:  # range
            start_str, end_str = parsed_range  # type: ignore[misc]
            from_dt = None
            to_dt = None

            logger.info(
                f"[=====DATA_NODE=====] Attempting to parse range | start_str={start_str} end_str={end_str}"
            )

            try:
                from_dt = datetime.datetime.strptime(start_str, "%d-%m-%Y")
                to_dt = datetime.datetime.strptime(end_str, "%d-%m-%Y")
                logger.info(
                    f"[=====DATA_NODE=====] Successfully parsed range dates | from_dt={from_dt} to_dt={to_dt}"
                )
            except Exception as exc:
                logger.error(
                    "[=====DATA_NODE=====] Failed to parse range dates | start_str=%s end_str=%s error=%s: %s",
                    start_str,
                    end_str,
                    type(exc).__name__,
                    exc,
                )
                from_dt = normalized[0][0]
                to_dt = normalized[-1][0]
                logger.info(
                    f"[=====DATA_NODE=====] Using fallback range (full history) | from_dt={from_dt} to_dt={to_dt}"
                )

            logger.info(
                f"[=====DATA_NODE=====] Calling compute_variations_for_range | normalized_size={len(normalized)} from_dt={from_dt} to_dt={to_dt}"
            )
            values = compute_variations_for_range(normalized, calc_mode_label, from_dt, to_dt)
            logger.info(
                f"[=====DATA_NODE=====] Computed range values | count={len(values)} first_date={values[0]['date'] if values else 'N/A'} last_date={values[-1]['date'] if values else 'N/A'}"
            )

            payload = {
                "intent": "value",
                "classification": _legacy_classification_payload(req_form_label),
                "series": match["id"],
                "parsed_point": None,
                "parsed_range": parsed_range,
                "result": values,
                "source_url": source_url,
            }
            logger.info(f"[=====DATA_NODE=====] Result (range): payload with {len(values)} observations")

        trace_info = _build_trace_info(payload)

        try:
            # Streaming incremental de respuesta al cliente/graph writer.
            stream = flow_data.stream_data_flow(
                payload,
                session_id=session_id,
            )
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
            return {"output": "".join(collected), "entities": entities, **trace_info}

        logger.info("[DATA_NODE] Completado | chunks=%d", len(collected))
        return {"output": "".join(collected), "entities": entities, **trace_info}

    return data_node


__all__ = ["make_data_node"]
