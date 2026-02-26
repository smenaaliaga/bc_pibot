"""Data node and supporting helpers for PIBot graph."""

from __future__ import annotations

import datetime
import os
import logging
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


def make_data_node(memory_adapter: Any):
    def data_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        question = state.get("question", "")
        session_id = state.get("session_id", "")
        entities = _clone_entities(state.get("entities"))
        primary_entity = _ensure_entity_slot(entities, 0)

        classification = state.get("classification")
        predict_raw = getattr(classification, "predict_raw", None) if classification else None
        predict_raw = predict_raw if isinstance(predict_raw, dict) else {}

        interpretation = predict_raw.get("interpretation") or {}
        interpretation_intents = interpretation.get("intents") or {}
        entities_normalized = interpretation.get("entities_normalized") or {}
        entities_raw = interpretation.get("entities") or {}
        routing = predict_raw.get("routing") or {}
        fallback_normalized = getattr(classification, "normalized", None) or {}
        fallback_entities = getattr(classification, "entities", None) or {}

        def _first_or_none(value: Any) -> Any:
            if isinstance(value, list):
                return next((item for item in value if item not in (None, "")), None)
            return value

        def _intent_label(key: str) -> Any:
            intent_payload = interpretation_intents.get(key)
            if isinstance(intent_payload, dict):
                return intent_payload.get("label")
            return intent_payload

        def _empty_to_none(value: Any) -> Any:
            if value in (None, "", [], {}):
                return "none"
            return value

        from rules.business_rule import resolve_region_value

        region_candidate = (
            entities_normalized.get("region")
            or entities_raw.get("region")
            or fallback_normalized.get("region")
            or fallback_entities.get("region")
            or predict_raw.get("region_value")
        )

        data_params = {
            "indicator": _first_or_none(entities_normalized.get("indicator"))
            or _first_or_none(fallback_normalized.get("indicator"))
            or _first_or_none(fallback_entities.get("indicator")),
            "seasonality": _first_or_none(entities_normalized.get("seasonality"))
            or _first_or_none(fallback_normalized.get("seasonality"))
            or _first_or_none(fallback_entities.get("seasonality")),
            "frequency": _first_or_none(entities_normalized.get("frequency"))
            or _first_or_none(fallback_normalized.get("frequency"))
            or _first_or_none(fallback_entities.get("frequency")),
            "period": entities_normalized.get("period")
            or fallback_normalized.get("period")
            or fallback_entities.get("period"),
            "calc_mode_cls": _intent_label("calc_mode"),
            "activity_cls": _intent_label("activity"),
            "region_cls": _intent_label("region"),
            "investment_cls": _intent_label("investment"),
            "req_form_cls": _intent_label("req_form"),
            "macro_cls": (routing.get("macro") or {}).get("label"),
            "intent_cls": (routing.get("intent") or {}).get("label"),
            "context_cls": (routing.get("context") or {}).get("label"),
            "enable": predict_raw.get("enable"),
            "enable_all": predict_raw.get("enable_all"),
            "activity_value": _first_or_none(entities_normalized.get("activity"))
            or _first_or_none(entities_raw.get("activity"))
            or _first_or_none(fallback_normalized.get("activity"))
            or _first_or_none(fallback_entities.get("activity"))
            or predict_raw.get("activity_value"),
            "sub_activity_value": predict_raw.get("sub_activity_value"),
            "region_value": resolve_region_value(region_candidate),
            "investment_value": predict_raw.get("investment_value"),
            "gasto_value": predict_raw.get("gasto_value"),
            "price": None,
            "history": None,
        }
        from rules.business_rule import resolve_calc_mode_cls

        data_params["calc_mode_cls"] = resolve_calc_mode_cls(
            question=question,
            calc_mode_cls=data_params.get("calc_mode_cls"),
            intent_cls=data_params.get("intent_cls"),
            req_form_cls=data_params.get("req_form_cls"),
            frequency=data_params.get("frequency"),
        )
        from rules.business_rule import classify_headers
        ## se aplican las reglas de negocio
        business_headers = classify_headers(
            question,
            predict_raw,
            enabled={"seasonality": True, "price": True, "history": True},
        )
        for key in ("seasonality", "price", "history"):
            if business_headers.get(key) and not data_params.get(key):
                data_params[key] = business_headers.get(key)
        data_params = {key: _empty_to_none(value) for key, value in data_params.items()}

        from rules.business_rule import (
            build_metadata_response,
            apply_latest_update_period,
            resolve_pib_annual_validity,
        )

        metadata_response = build_metadata_response(data_params)
        metadata_key = metadata_response.get("key") if isinstance(metadata_response, dict) else None

        period_override = apply_latest_update_period(data_params, metadata_response)
        if period_override:
            data_params["period"] = period_override

        annual_validation = resolve_pib_annual_validity(data_params, metadata_response)
        resolved_period = annual_validation.get("resolved_period") if isinstance(annual_validation, dict) else None
        if isinstance(resolved_period, list) and resolved_period:
            data_params["period"] = resolved_period

        source_url: Optional[Any] = None
        if isinstance(metadata_response, dict):
            sources_url = metadata_response.get("sources_url")
            if sources_url:
                source_url = sources_url

        data_params_status = {
            key: "PRESENT" if value not in (None, "", [], {}) else "MISSING"
            for key, value in data_params.items()
        }

        if isinstance(annual_validation, dict) and annual_validation.get("applies") and not annual_validation.get("is_valid"):
            requested_year = annual_validation.get("requested_year")
            max_valid_year = annual_validation.get("max_valid_year")
            max_valid_year_text = str(max_valid_year) if max_valid_year not in (None, "", "none") else "no disponible"

            if requested_year:
                base_text = (
                    f"El PIB anual correspondiente al año {requested_year} aún no se encuentra publicado. "
                    f"El último dato disponible con frecuencia anual corresponde al año {max_valid_year_text}"
                )
            else:
                base_text = (
                    "El PIB anual consultado aún no se encuentra publicado. "
                    f"El último dato disponible con frecuencia anual corresponde al año {max_valid_year_text}"
                )
            source_ref = None
            if isinstance(source_url, dict):
                source_ref = next(
                    (str(v).strip() for v in source_url.values() if v and str(v).strip().lower() != "none"),
                    None,
                )
            elif isinstance(source_url, list):
                source_ref = next(
                    (str(v).strip() for v in source_url if v and str(v).strip().lower() != "none"),
                    None,
                )
            elif isinstance(source_url, str) and source_url.strip() and source_url.strip().lower() != "none":
                source_ref = source_url.strip()

            if source_ref:
                text = f"{base_text} y puede consultarse en el siguiente enlace: {source_ref}"
            else:
                text = base_text
            _emit_stream_chunk(text, writer)
            return {
                "output": text,
                "entities": entities,
                "data_params": data_params,
                "data_params_status": data_params_status,
                "metadata_response": metadata_response,
                "metadata_key": metadata_key,
                "series_fetch_args": None,
                "series_fetch_result": None,
                "annual_validation": annual_validation,
            }

        def _infer_frequency_from_series_id(series_id: Optional[str]) -> Optional[str]:
            if not series_id:
                return None
            raw = str(series_id).strip().upper()
            if not raw:
                return None
            token = raw.split(".")[-1]
            mapping = {
                "M": "m",
                "T": "q",
                "Q": "q",
                "A": "a",
                "D": "d",
            }
            return mapping.get(token)

        def _normalize_period_for_frequency(
            firstdate: Optional[str],
            lastdate: Optional[str],
            target_frequency: Optional[str],
            req_form: Optional[str],
        ) -> Tuple[Optional[str], Optional[str]]:
            if target_frequency != "a":
                return firstdate, lastdate

            req_form_value = str(req_form or "").strip().lower()
            if req_form_value == "range":
                return firstdate, lastdate

            reference = str(lastdate or firstdate or "").strip()
            if len(reference) < 4 or not reference[:4].isdigit():
                return firstdate, lastdate

            year = reference[:4]
            return f"{year}-01-01", f"{year}-12-31"

        def _normalize_activity_key_from_title(title: Optional[str]) -> str:
            text = str(title or "").strip().lower()
            if not text:
                return "unknown"
            normalized = (
                text.replace("á", "a")
                .replace("é", "e")
                .replace("í", "i")
                .replace("ó", "o")
                .replace("ú", "u")
            )
            mapping = {
                "imacec": "total",
                "imacec no minero": "no_minero",
                "produccion de bienes": "bienes",
                "mineria": "minero",
                "industria": "industria",
                "resto de bienes": "resto_bienes",
                "comercio": "comercio",
                "servicios": "servicios",
                "impuestos sobre los productos": "impuestos sobre los productos",
            }
            return mapping.get(normalized, normalized.replace(" ", "_"))

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

            firstdate, lastdate = _normalize_period_for_frequency(
                firstdate,
                lastdate,
                target_frequency,
                req_form_value,
            )

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
                            "activity": _normalize_activity_key_from_title(serie_title),
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
                series_native_frequency = _infer_frequency_from_series_id(str(serie_default))
                if series_native_frequency and target_frequency != "a" and (
                    req_form_value in {"point", "range"}
                    or target_frequency != series_native_frequency
                ):
                    target_frequency = series_native_frequency

                agg_mode = "avg"
                series_id_text = str(serie_default or "").upper()
                if target_frequency == "a" and ".FLU." in series_id_text:
                    agg_mode = "sum"

                firstdate, lastdate = _normalize_period_for_frequency(
                    firstdate,
                    lastdate,
                    target_frequency,
                    req_form_value,
                )
                series_fetch_args = {
                    "series_id": serie_default,
                    "firstdate": firstdate,
                    "lastdate": lastdate,
                    "target_frequency": target_frequency,
                    "agg": agg_mode,
                }
                series_data = get_series_from_redis(
                    series_id=serie_default,
                    firstdate=firstdate,
                    lastdate=lastdate,
                    target_frequency=target_frequency,
                    agg=agg_mode,
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
                    fallback_data = get_series_from_redis(
                        series_id=serie_default,
                        firstdate=None,
                        lastdate=None,
                        target_frequency=target_frequency,
                        agg=agg_mode,
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

        log_path = None
        try:
            logs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "logs")
            os.makedirs(logs_dir, exist_ok=True)
            fixed = os.getenv("RUN_MAIN_LOG", "").strip() or "run_main.log"
            log_path = os.path.join(logs_dir, fixed)
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write("[CLASSIFIER_FILE] METADATA_RESPONSE=%s\n" % metadata_response)
        except Exception:
            pass

        result = predict_with_interpreter(question)

        logger.info(
            "[=====DATA_NODE=====] PIBOT_SERIES_INTERPRETER | %s",
            vars(result) if hasattr(result, "__dict__") else result,
        )

        if not result:
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
                reference_period = None
                if isinstance(period, list) and period:
                    reference_period = str(period[-1] or period[0])

                effective_req_form = req_form_value

                output_dict: Dict[str, Any] = {
                    "date": None,
                    "value": None,
                    "serie": serie_default,
                }
                calc_mode_value = str(data_params.get("calc_mode_cls") or "").lower()
                metric_type_value = "contribution" if calc_mode_value == "contribution" else "index"
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

                all_series_data: List[Dict[str, Any]] = []
                metadata_series = metadata_response.get("series") if isinstance(metadata_response, dict) else None
                if metric_type_value == "contribution" and isinstance(metadata_series, dict) and metadata_series:
                    all_series_data = _build_all_series_data(
                        metadata_series=metadata_series,
                        period=period,
                        target_frequency=effective_frequency,
                        calc_mode_value=calc_mode_value,
                        req_form_value=str(req_form_value or "").lower(),
                    )

                payload_series = serie_default
                if (
                    (not payload_series or str(payload_series).strip().lower() == "none")
                    and all_series_data
                ):
                    payload_series = all_series_data[0].get("series_id")

                payload = {
                    "intent": data_params.get("intent_cls") or "value",
                    "classification": {
                        "indicator": data_params.get("indicator"),
                        "metric_type": metric_type_value,
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
                    "series": payload_series,
                    "series_title": metadata_response.get("title_serie_default") if isinstance(metadata_response, dict) else None,
                    "parsed_point": parsed_point if effective_req_form != "range" else None,
                    "parsed_range": parsed_range,
                    "reference_period": reference_period,
                    "result": range_rows if effective_req_form == "range" else output_dict,
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

                trace_info = {
                    "parsed_point": payload.get("parsed_point"),
                    "parsed_range": payload.get("parsed_range"),
                    "series": payload.get("series"),
                    "data_classification": payload.get("classification"),
                    "data_params": data_params,
                    "data_params_status": data_params_status,
                    "metadata_response": metadata_response,
                    "metadata_key": metadata_key,
                    "series_fetch_args": series_fetch_args,
                    "series_fetch_result": series_fetch_result,
                    "annual_validation": annual_validation,
                }
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
                "annual_validation": annual_validation,
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
        calc_mode_label = _normalize_calc_mode(
            calc_mode_label,
            indicator_label,
            req_form_label,
            region_cls,
            activity_cls,
            getattr(getattr(result, "investment_cls", None), "label", None),
        )

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
                "classification": {
                    "indicator": indicator_label,
                    "metric_type": metric_type_label,
                    "seasonality": seasonality_label,
                    "activity": activity_label,
                    "frequency": frequency_label,
                    "calc_mode": calc_mode_label,
                    "req_form": req_form_label,
                    "calc_mode_cls": calc_mode_cls,
                    "frequency_cls": frequency_cls,
                    "activity_cls": activity_cls,
                    "region_cls": region_cls,
                    "req_form_cls": req_form_cls,
                },
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
                "classification": {
                    "indicator": indicator_label,
                    "metric_type": metric_type_label,
                    "seasonality": seasonality_label,
                    "activity": activity_label,
                    "frequency": frequency_label,
                    "calc_mode": calc_mode_label,
                    "req_form": req_form_label,
                    "calc_mode_cls": calc_mode_cls,
                    "frequency_cls": frequency_cls,
                    "activity_cls": activity_cls,
                    "region_cls": region_cls,
                    "req_form_cls": req_form_cls,
                },
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
                "classification": {
                    "indicator": indicator_label,
                    "metric_type": metric_type_label,
                    "seasonality": seasonality_label,
                    "activity": activity_label,
                    "frequency": frequency_label,
                    "calc_mode": calc_mode_label,
                    "req_form": req_form_label,
                    "calc_mode_cls": calc_mode_cls,
                    "frequency_cls": frequency_cls,
                    "activity_cls": activity_cls,
                    "region_cls": region_cls,
                    "req_form_cls": req_form_cls,
                },
                "series": match["id"],
                "parsed_point": None,
                "parsed_range": parsed_range,
                "reference_period": None,
                "result": values,
                "source_url": source_url,
            }
            logger.info(f"[=====DATA_NODE=====] Result (range): payload with {len(values)} observations")

        if req_form_label != "range":
            period_value = data_params.get("period")
            if isinstance(period_value, list) and period_value:
                payload["reference_period"] = str(period_value[-1] or period_value[0])
            elif payload.get("result") and isinstance(payload.get("result"), dict):
                result_date = payload.get("result", {}).get("date")
                if result_date:
                    payload["reference_period"] = str(result_date)

        all_series_data: Optional[List[Dict[str, Any]]] = None
        if (
            isinstance(metric_type_label, str)
            and metric_type_label.lower() == "contribution"
            and isinstance(metadata_response, dict)
        ):
            metadata_series = metadata_response.get("series")
            if isinstance(metadata_series, dict) and metadata_series:
                all_series_data = _build_all_series_data(
                    metadata_series=metadata_series,
                    period=data_params.get("period"),
                    target_frequency=str(frequency_label or "").lower() or None,
                    calc_mode_value=str(calc_mode_label or "").lower(),
                    req_form_value=str(req_form_label or "").lower(),
                )

        if all_series_data:
            payload["all_series_data"] = all_series_data
            series_payload = payload.get("series")
            if not series_payload or str(series_payload).strip().lower() == "none":
                payload["series"] = all_series_data[0].get("series_id")

        trace_info = {
            "parsed_point": payload.get("parsed_point"),
            "parsed_range": payload.get("parsed_range"),
            "series": payload.get("series"),
            "data_classification": payload.get("classification"),
            "data_params": data_params,
            "data_params_status": data_params_status,
            "metadata_response": metadata_response,
            "metadata_key": metadata_key,
            "series_fetch_args": series_fetch_args,
            "series_fetch_result": series_fetch_result,
            "annual_validation": annual_validation,
        }

        try:
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
