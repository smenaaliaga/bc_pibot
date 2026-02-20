"""Data node and supporting helpers for PIBot graph."""

from __future__ import annotations

import datetime
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

        result = predict_with_interpreter(question)

        logger.info(
            "[=====DATA_NODE=====] PIBOT_SERIES_INTERPRETER | %s",
            vars(result) if hasattr(result, "__dict__") else result,
        )

        if not result:
            text = "[GRAPH] No se recibió clasificación para el nodo DATA."
            logger.warning("[=====DATA_NODE=====] %s", text)
            _emit_stream_chunk(text, writer)
            return {"output": text, "entities": entities}

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
                "result": values,
            }
            logger.info(f"[=====DATA_NODE=====] Result (range): payload with {len(values)} observations")

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
            return {"output": "".join(collected), "entities": entities}

        logger.info("[DATA_NODE] Completado | chunks=%d", len(collected))
        return {"output": "".join(collected), "entities": entities}

    return data_node


__all__ = ["make_data_node"]
