from __future__ import annotations
import logging
from datetime import date
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
    req_form = (
        classification_dict.get("req_form")
        or classification_dict.get("req_form_cls")
        or "latest"
    )
    effective_req_form = str(req_form or "").strip().lower()
    parsed_point = payload.get("parsed_point")  # DD-MM-YYYY o None
    parsed_range = payload.get("parsed_range")  # Tupla (DD-MM-YYYY, DD-MM-YYYY) o None
    reference_period = payload.get("reference_period")
    all_series_data = payload.get("all_series_data")  # Lista con todas las series (para contribución con activity=none)
    source_url = payload.get("source_url")  # URL de la fuente
    series_title = payload.get("series_title")
    user_question = payload.get("question") or ""
    source_urls = normalize_sources(source_url)
    used_latest_fallback_for_point = bool(payload.get("used_latest_fallback_for_point"))
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
    component_context_val = (
        classification_dict.get("activity")
        or classification_dict.get("activity_value")
    )
    seasonality_context_val = classification_dict.get("seasonality")
    region_context_val = (
        classification_dict.get("region_value")
        or classification_dict.get("region")
    )
    metric_type_val = classification_dict.get("metric_type") or classification_dict.get("calc_mode_cls")
    freq = classification_dict.get("frequency", "M").upper()
    
    logger.info(
        "[STREAM_DATA_FLOW] Procesando payload | series=%s | indicator=%s | freq=%s | req_form=%s",
        series_id,
        indicator_context_val,
        freq,
        req_form,
    )

    def _parse_iso_date(value: Any) -> Optional[date]:
        try:
            text = str(value or "").strip()
            if not text:
                return None
            return date.fromisoformat(text[:10])
        except Exception:
            return None

    def _same_requested_period(requested: Optional[date], observed: Optional[date], freq_value: Any) -> bool:
        if requested is None or observed is None:
            return False
        freq_norm = str(freq_value or "").strip().lower()
        if freq_norm in {"a", "annual", "anual"}:
            return requested.year == observed.year
        if freq_norm in {"q", "t", "quarterly", "trimestral"}:
            requested_quarter = ((requested.month - 1) // 3) + 1
            observed_quarter = ((observed.month - 1) // 3) + 1
            return requested.year == observed.year and requested_quarter == observed_quarter
        return requested.year == observed.year and requested.month == observed.month

    if effective_req_form == "range":
        freq_norm = str(freq or "").strip().lower()
        if freq_norm in {"a", "annual", "anual"}:
            start_date: Optional[date] = None
            end_date: Optional[date] = None
            indicator_norm = str(indicator_context_val or "").strip().lower()

            if isinstance(parsed_range, (tuple, list)) and len(parsed_range) == 2:
                start_date = _parse_iso_date(parsed_range[0])
                end_date = _parse_iso_date(parsed_range[1])

            if start_date is None or end_date is None:
                observed_dates = [
                    _parse_iso_date(row.get("date"))
                    for row in obs
                    if isinstance(row, dict)
                ]
                valid_observed_dates = [value for value in observed_dates if value is not None]
                if valid_observed_dates:
                    start_date = min(valid_observed_dates)
                    end_date = max(valid_observed_dates)

            if (
                start_date is not None
                and end_date is not None
                and start_date.year == end_date.year
                and indicator_norm != "imacec"
            ):
                effective_req_form = "point"
                if not parsed_point:
                    parsed_point = end_date.isoformat()

    # Para range/specific_point: mostrar todas las observaciones.
    # Para latest: última observación disponible.
    # Para point: observación alineada al período consultado (parsed_point).
    if effective_req_form in {"range", "specific_point"}:
        obs_to_show = obs
    elif effective_req_form == "point":
        requested_date = _parse_iso_date(parsed_point)
        dated_obs = [
            (row, _parse_iso_date(row.get("date")))
            for row in obs
            if isinstance(row, dict)
        ]

        chosen_row = None
        if requested_date:
            same_date = [row for row, row_date in dated_obs if row_date == requested_date]
            if same_date:
                chosen_row = same_date[-1]
            else:
                previous_or_equal = [
                    (row, row_date)
                    for row, row_date in dated_obs
                    if row_date is not None and row_date <= requested_date
                ]
                if previous_or_equal:
                    chosen_row = max(previous_or_equal, key=lambda item: item[1])[0]
                else:
                    valid_dated = [
                        (row, row_date)
                        for row, row_date in dated_obs
                        if row_date is not None
                    ]
                    if valid_dated:
                        chosen_row = min(
                            valid_dated,
                            key=lambda item: abs((item[1] - requested_date).days),
                        )[0]

        if chosen_row is None:
            chosen_row = max(obs, key=lambda o: o.get("date", "")) if obs else None

        if not chosen_row:
            logger.error("[STREAM_DATA_FLOW] No hay observaciones en result_data")
            return
        obs_to_show = [chosen_row]
        observed_date = _parse_iso_date(chosen_row.get("date")) if isinstance(chosen_row, dict) else None
        if requested_date and not _same_requested_period(requested_date, observed_date, freq):
            used_latest_fallback_for_point = True
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
    if (
        isinstance(component_context_val, str)
        and component_context_val.strip()
        and component_context_val.lower() not in {"total", "general", "none", "specific"}
    ):
        indicator_parts.append(component_context_val.strip())
    # Para estacionalidad: reemplazar "SA" por "desestacionalizado", excluir "nsa"
    if isinstance(seasonality_context_val, str) and seasonality_context_val.strip():
        if seasonality_context_val.lower() == "sa":
            indicator_parts.append("desestacionalizado")
        elif seasonality_context_val.lower() != "nsa":
            indicator_parts.append(seasonality_context_val.strip())
    final_indicator_name = " ".join(indicator_parts).strip() if indicator_parts else "indicador"

    region_context_norm = str(region_context_val or "").strip()
    region_context_key = region_context_norm.lower().replace("_", " ")
    has_specific_region = region_context_key not in {"", "none", "null", "general", "specific", "total"}
    if str(indicator_context_val or "").strip().lower() == "pib" and has_specific_region:
        region_display = " ".join(word.capitalize() for word in region_context_key.split())
        if "region" not in final_indicator_name.lower() and "región" not in final_indicator_name.lower():
            final_indicator_name = f"{final_indicator_name} de la región de {region_display}".strip()
    
    # Determinar si component_context_val es una actividad concreta o no
    is_specific_activity = (
        isinstance(component_context_val, str)
        and component_context_val.strip()
        and component_context_val.lower() not in {"none", "total", "general", "specific", ""}
    )

    # Flag para lógica especial de contribución
    metric_type_norm = str(metric_type_val or "").strip().lower()
    is_contribution = (
        metric_type_norm == "contribution"
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
    if effective_req_form in {"range", "specific_point"}:
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
        if effective_req_form == "latest" and is_contribution:
            if reference_period:
                period_labels = format_period_labels(str(reference_period), freq)
                date_range_label = period_labels[0]
                logger.info("[STREAM_DATA_FLOW] Renderizando (reference_period) | date=%s", reference_period)
            else:
                date_raw = obs_to_show[0].get("date", "") if obs_to_show else ""
                period_labels = format_period_labels(date_raw, freq)
                date_range_label = period_labels[0]
                logger.info("[STREAM_DATA_FLOW] Renderizando (observación/latest_contrib) | date=%s", date_raw)
        elif parsed_point:
            period_labels = format_period_labels(parsed_point, freq)
            date_range_label = period_labels[0]
            logger.info("[STREAM_DATA_FLOW] Renderizando (parsed_point) | date=%s", parsed_point)
        else:
            date_raw = obs_to_show[0].get("date", "") if obs_to_show else ""
            period_labels = format_period_labels(date_raw, freq)
            date_range_label = period_labels[0]
            logger.info("[STREAM_DATA_FLOW] Renderizando (observación) | date=%s", date_raw)

    # Etiqueta para mostrar en prompts: para latest usar texto genérico
    if effective_req_form == "latest":
        display_period_label = date_range_label if (is_contribution and date_range_label != "--") else "el último período disponible"
    else:
        display_period_label = date_range_label

    response_stream = (
        specific_point_response(
            series_id=series_id,
            series_title=series_title,
            req_form=effective_req_form,
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
            used_latest_fallback_for_point=used_latest_fallback_for_point,
            source_urls=source_urls,
            intro_llm_temperature=intro_llm_temperature_value,
            question=user_question,
        )
        if effective_req_form == "specific_point"
        else specific_response(
            series_id=series_id,
            series_title=series_title,
            req_form=effective_req_form,
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
            used_latest_fallback_for_point=used_latest_fallback_for_point,
            source_urls=source_urls,
            intro_llm_temperature=intro_llm_temperature_value,
            question=user_question,
        )
    )

    for chunk in response_stream:
        yield chunk