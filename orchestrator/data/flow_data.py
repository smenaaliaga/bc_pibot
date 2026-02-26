from __future__ import annotations
import logging
from typing import Any, Dict, Iterable, Optional

from orchestrator.data.response import (
    general_response,
    normalize_sources,
    specific_point_response,
    specific_response,
    format_period_labels,
)
    
logger = logging.getLogger(__name__)


def stream_data_flow(
    payload: Dict[str, Any],
    session_id: Optional[str] = None,
) -> Iterable[str]:
    """Fetch de datos y tabla desde payload precalculado.
    
    Args:
        payload: Diccionario con estructura:
            {
                'intent': 'value',
                'classification': {...},
                'series': 'F032.IMC.IND.Z.Z.EP18.Z.Z.0.M',
                'parsed_point': 'DD-MM-YYYY' or None,  # fecha específica para point
                'parsed_range': ('DD-MM-YYYY', 'DD-MM-YYYY') or None,  # rango para range
                'result': [...],  # datos ya procesados (dict para latest/point, lista para range)
                'source_url': 'https://...'  # URL de la fuente
            }
        session_id: Identificador de sesión (opcional)
    """
    
    # Extraer datos del payload
    series_id = payload.get("series")
    classification_dict = payload.get("classification", {})
    result_data = payload.get("result", [])
    intent = payload.get("intent", "")
    req_form = classification_dict.get("req_form", "latest")
    parsed_point = payload.get("parsed_point")  # DD-MM-YYYY o None
    parsed_range = payload.get("parsed_range")  # Tupla (DD-MM-YYYY, DD-MM-YYYY) o None
    reference_period = payload.get("reference_period")
    all_series_data = payload.get("all_series_data")  # Lista con todas las series (para contribución con activity=none)
    source_url = payload.get("source_url")  # URL de la fuente
    series_title = payload.get("series_title")
    source_urls = normalize_sources(source_url)
    intro_llm_temperature = payload.get("intro_llm_temperature", 0.7)
    try:
        intro_llm_temperature_value = float(intro_llm_temperature)
    except (TypeError, ValueError):
        intro_llm_temperature_value = 0.7

    if not series_id or str(series_id).lower() == "none":
        for chunk in general_response(source_urls, series_id=series_id):
            yield chunk
        return
    
    # Normalizar result_data: puede ser dict (latest/point) o lista (range)
    if isinstance(result_data, dict):
        obs = [result_data]  # Convertir dict a lista de 1 elemento
        logger.debug("[STREAM_DATA_FLOW] result_data es dict, normalizado a lista")
    elif isinstance(result_data, list):
        obs = result_data
        logger.debug(f"[STREAM_DATA_FLOW] result_data es lista con {len(obs)} elementos")
    else:
        logger.error("[STREAM_DATA_FLOW] result_data tiene tipo inesperado: %s", type(result_data))
        return
    
    if not obs:
        logger.error("[STREAM_DATA_FLOW] No hay observaciones en result_data")
        return
    
    # Extraer valores de clasificación
    indicator_context_val = classification_dict.get("indicator")
    component_context_val = classification_dict.get("activity")  # o usar otro campo si aplica
    seasonality_context_val = classification_dict.get("seasonality")
    metric_type_val = classification_dict.get("metric_type")
    freq = classification_dict.get("frequency", "M").upper()
    
    logger.info(
        "[STREAM_DATA_FLOW] Procesando payload | series=%s | indicator=%s | freq=%s | req_form=%s",
        series_id,
        indicator_context_val,
        freq,
        req_form,
    )

    # Para range/specific_point: mostrar todas las observaciones. Para latest/point: solo la última
    if req_form in {"range", "specific_point"}:
        obs_to_show = obs
    else:
        chosen_row = max(obs, key=lambda o: o.get("date", "")) if obs else None
        if not chosen_row:
            logger.error("[STREAM_DATA_FLOW] No hay observaciones en result_data")
            return
        obs_to_show = [chosen_row]
    
    # Nombre final del indicador: indicador + componente (si aplica) + estacionalidad (si aplica)
    indicator_parts: list[str] = []
    if isinstance(indicator_context_val, str) and indicator_context_val.strip():
        indicator_parts.append(indicator_context_val.upper().strip())
    # Excluir "total" del componente/activity
    if isinstance(component_context_val, str) and component_context_val.strip() and component_context_val.lower() != "total":
        indicator_parts.append(component_context_val.strip())
    # Para estacionalidad: reemplazar "SA" por "desestacionalizado", excluir "nsa"
    if isinstance(seasonality_context_val, str) and seasonality_context_val.strip():
        if seasonality_context_val.lower() == "sa":
            indicator_parts.append("desestacionalizado")
        elif seasonality_context_val.lower() != "nsa":
            indicator_parts.append(seasonality_context_val.strip())
    final_indicator_name = " ".join(indicator_parts).strip() if indicator_parts else "indicador"
    
    # Determinar si component_context_val es una actividad concreta o no
    is_specific_activity = (
        isinstance(component_context_val, str)
        and component_context_val.strip()
        and component_context_val.lower() not in {"none", "total", ""}
    )

    # Flag para lógica especial de contribución
    is_contribution = (
        isinstance(metric_type_val, str)
        and metric_type_val.lower() == "contribution"
        and isinstance(indicator_context_val, str)
        and indicator_context_val.strip()
    )

    if not reference_period and isinstance(all_series_data, list) and all_series_data:
        date_candidates = [
            str(item.get("date"))
            for item in all_series_data
            if isinstance(item, dict) and item.get("date")
        ]
        if date_candidates:
            reference_period = max(date_candidates)
    
    # Para range/specific_point: mostrar rango de fechas usando parsed_range si está disponible; para latest/point: usar parsed_point o fecha de observación
    if req_form in {"range", "specific_point"}:
        if parsed_range:
            start_str, end_str = parsed_range
            start_labels = format_period_labels(start_str, freq)
            end_labels = format_period_labels(end_str, freq)
            date_range_label = f"desde {start_labels[0]} hasta {end_labels[0]}"
            logger.info("[STREAM_DATA_FLOW] Rango (parsed) | %s | %d observaciones", date_range_label, len(obs_to_show))
        else:
            first_date = obs_to_show[0].get("date", "") if obs_to_show else ""
            last_date = obs_to_show[-1].get("date", "") if obs_to_show else ""
            first_labels = format_period_labels(first_date, freq)
            last_labels = format_period_labels(last_date, freq)
            date_range_label = f"desde {first_labels[0]} hasta {last_labels[0]}"
            logger.info("[STREAM_DATA_FLOW] Rango (observaciones) | %s | %d observaciones", date_range_label, len(obs_to_show))
    else:
        if parsed_point:
            period_labels = format_period_labels(parsed_point, freq)
            date_range_label = period_labels[0]
            logger.info("[STREAM_DATA_FLOW] Renderizando (parsed_point) | date=%s", parsed_point)
        elif req_form == "latest" and is_contribution and reference_period:
            period_labels = format_period_labels(str(reference_period), freq)
            date_range_label = period_labels[0]
            logger.info("[STREAM_DATA_FLOW] Renderizando (reference_period) | date=%s", reference_period)
        else:
            date_raw = obs_to_show[0].get("date", "") if obs_to_show else ""
            period_labels = format_period_labels(date_raw, freq)
            date_range_label = period_labels[0]
            logger.info("[STREAM_DATA_FLOW] Renderizando (observación) | date=%s", date_raw)

    # Etiqueta para mostrar en prompts: para latest usar texto genérico
    if req_form == "latest":
        display_period_label = date_range_label if (is_contribution and date_range_label != "--") else "el último período disponible"
    else:
        display_period_label = date_range_label

    response_stream = (
        specific_point_response(
            series_id=series_id,
            series_title=series_title,
            req_form=req_form,
            obs_to_show=obs_to_show,
            parsed_point=parsed_point,
            parsed_range=parsed_range,
            final_indicator_name=final_indicator_name,
            indicator_context_val=indicator_context_val,
            component_context_val=component_context_val,
            seasonality_context_val=seasonality_context_val,
            metric_type_val=metric_type_val,
            calc_mode_cls=classification_dict.get("calc_mode_cls"),
            intent_cls=intent,
            freq=freq,
            display_period_label=display_period_label,
            date_range_label=date_range_label,
            reference_period=str(reference_period) if reference_period else None,
            is_contribution=is_contribution,
            is_specific_activity=is_specific_activity,
            all_series_data=all_series_data,
            source_urls=source_urls,
            intro_llm_temperature=intro_llm_temperature_value,
        )
        if req_form == "specific_point"
        else specific_response(
            series_id=series_id,
            series_title=series_title,
            req_form=req_form,
            obs_to_show=obs_to_show,
            parsed_point=parsed_point,
            parsed_range=parsed_range,
            final_indicator_name=final_indicator_name,
            indicator_context_val=indicator_context_val,
            component_context_val=component_context_val,
            seasonality_context_val=seasonality_context_val,
            metric_type_val=metric_type_val,
            calc_mode_cls=classification_dict.get("calc_mode_cls"),
            intent_cls=intent,
            freq=freq,
            display_period_label=display_period_label,
            date_range_label=date_range_label,
            reference_period=str(reference_period) if reference_period else None,
            is_contribution=is_contribution,
            is_specific_activity=is_specific_activity,
            all_series_data=all_series_data,
            source_urls=source_urls,
            intro_llm_temperature=intro_llm_temperature_value,
        )
    )

    for chunk in response_stream:
        yield chunk