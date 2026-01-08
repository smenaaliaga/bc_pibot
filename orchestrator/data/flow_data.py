from __future__ import annotations
import logging
import time
import json
from typing import Any, Dict, Iterable, Optional

from orchestrator.memory.memory_adapter import MemoryAdapter
from orchestrator.data.get_series import detect_series_code 
from orchestrator.data.get_data_serie import get_series_api_rest_bcch
from orchestrator.data.templates import select_template, render_template
from orchestrator.llm.llm_adapter import build_llm
    
logger = logging.getLogger(__name__)


def _get_context(
    *,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Cargar memoria/historial desde Redis."""
    ctx: Dict[str, Any] = {
        "session_id": session_id,
        "facts": None,
        "history": None,
    }

    # Obtener memoria/historial desde Redis (MemoryAdapter) si hay session_id
    if session_id:
        try:
            mem = MemoryAdapter()
            ctx["facts"] = mem.get_facts(session_id)
            ctx["history"] = mem.get_history_for_llm(session_id)
        except Exception:
            ctx["facts_error"] = "memory_unavailable"

    return ctx


def _as_mapping(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        try:
            return dict(payload.__dict__)
        except Exception:
            return {}
    return {}


def _resolve_period_context(normalized: Dict[str, Any], facts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Resuelve período: primero clasificación (nuevo formato: granularity + target_date), luego Redis."""
    period_obj = normalized.get("period")
    # Nuevo formato simplificado: granularity + target_date
    if isinstance(period_obj, dict) and period_obj.get("granularity") and period_obj.get("target_date"):
        try:
            logger.debug(
                f"Periodo desde clasificación: granularity={period_obj.get('granularity')} target_date={period_obj.get('target_date')}"
            )
        except Exception:
            pass
        return period_obj
    # Formato legacy: firstdate + lastdate (para compatibilidad con Redis)
    if isinstance(period_obj, dict) and period_obj.get("firstdate") and period_obj.get("lastdate"):
        try:
            logger.debug(
                f"Periodo desde clasificación (legacy): firstdate={period_obj.get('firstdate')} lastdate={period_obj.get('lastdate')}"
            )
        except Exception:
            pass
        return period_obj
    if isinstance(facts, dict):
        facts_period = facts.get("period")
        logger.debug(f"Verificando facts.period: {facts_period}")
        # Si viene como string JSON desde memoria, intentar parsearlo
        if isinstance(facts_period, str):
            try:
                parsed = json.loads(facts_period)
                if isinstance(parsed, dict):
                    facts_period = parsed
                    logger.debug("facts.period parseado desde JSON correctamente")
            except Exception:
                logger.debug("No se pudo parsear facts.period como JSON", exc_info=False)
        # Nuevo formato
        if isinstance(facts_period, dict) and facts_period.get("granularity") and facts_period.get("target_date"):
            try:
                logger.debug(
                    f"Periodo desde memoria: granularity={facts_period.get('granularity')} target_date={facts_period.get('target_date')}"
                )
            except Exception:
                pass
            return facts_period
        # Formato legacy
        if isinstance(facts_period, dict) and facts_period.get("firstdate") and facts_period.get("lastdate"):
            try:
                logger.debug(
                    f"Periodo desde memoria (legacy): firstdate={facts_period.get('firstdate')} lastdate={facts_period.get('lastdate')}"
                )
            except Exception:
                pass
            return facts_period
        if facts_period is not None:
            logger.debug(f"facts.period existe pero no tiene el formato esperado: {facts_period}")
    else:
        logger.debug("facts es None o no es dict")
    logger.debug("Periodo no determinado en clasificación/memoria; se utilizará ventana automática (None)")
    return None


def _resolve_indicator_context(normalized: Dict[str, Any], facts: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resuelve indicador: primero clasificación, luego Redis."""
    indicator_obj = normalized.get("indicator")
    if isinstance(indicator_obj, dict):
        val = indicator_obj.get("normalized")
        if isinstance(val, str) and val.strip():
            logger.debug(f"Indicador desde clasificación: {val}")
            return val
    if isinstance(facts, dict):
        ind = facts.get("indicator")
        if isinstance(ind, str) and ind.strip():
            logger.debug(f"Indicador desde memoria: {ind.strip()}")
            return ind.strip()
    logger.debug("Indicador no determinado (clasificación/memoria)")
    return None


def _resolve_component_context(normalized: Dict[str, Any], facts: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resuelve componente: primero clasificación, luego Redis."""
    comp_obj = normalized.get("component")
    if isinstance(comp_obj, dict):
        val = comp_obj.get("normalized")
        if isinstance(val, str) and val.strip():
            logger.debug(f"Componente desde clasificación: {val}")
            return val
    if isinstance(facts, dict):
        comp = facts.get("component")
        if isinstance(comp, str) and comp.strip():
            logger.debug(f"Componente desde memoria: {comp.strip()}")
            return comp.strip()
    logger.debug("Componente no determinado (clasificación/memoria)")
    return None


def _resolve_seasonality_context(normalized: Dict[str, Any], facts: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resuelve estacionalidad: primero clasificación, luego Redis."""
    seasonality_obj = normalized.get("seasonality")
    if isinstance(seasonality_obj, dict):
        val = seasonality_obj.get("normalized") or seasonality_obj.get("label")
        if isinstance(val, str) and val.strip():
            logger.debug(f"Estacionalidad desde clasificación: {val.strip()}")
            return val.strip()
    elif isinstance(seasonality_obj, str) and seasonality_obj.strip():
        logger.debug(f"Estacionalidad desde clasificación (texto): {seasonality_obj.strip()}")
        return seasonality_obj.strip()
    if isinstance(facts, dict):
        seas = facts.get("seasonality")
        if isinstance(seas, str) and seas.strip():
            logger.debug(f"Estacionalidad desde memoria: {seas.strip()}")
            return seas.strip()
    logger.debug("Estacionalidad no determinada (clasificación/memoria)")
    return None


def _match_observation(
    observations: list,
    target_date_str: Optional[str],
    granularity: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Match observation using target_date and granularity.
    - target_date formats: "YYYY-MM-DD" (month), "YYYY-QN" (quarter), "YYYY" (year)
    - Monthly: match by YYYY-MM
    - Quarterly: map to quarter-end month (03, 06, 09, 12)
    - Year: match by YYYY (choose last observation in year)
    """
    if not observations or not target_date_str:
        return None

    try:
        gran = (granularity or "").lower()
        
        # Monthly: match by year-month (YYYY-MM-DD -> YYYY-MM)
        if gran == "month":
            ym_target = target_date_str[:7]
            logger.debug(f"Matching month: buscando {ym_target}")
            for obs in observations:
                ds = obs.get("date")
                if ds and ds[:7] == ym_target:
                    return obs

        # Quarterly: handle either symbolic quarter (YYYY-QN) or explicit date
        elif gran == "quarter":
            # Observations are quarter-end (e.g., 1996-03-31, 1996-06-30, ...)
            # Convert target to last month of quarter for matching
            import re
            m = re.match(r"(\d{4})-Q([1-4])", target_date_str)
            if m:
                year = int(m.group(1))
                quarter = int(m.group(2))
                last_month = quarter * 3
                ym_target = f"{year:04d}-{last_month:02d}"
                logger.debug(f"Matching quarter: {target_date_str}, buscando mes de cierre {ym_target}")
                for obs in observations:
                    ds = obs.get("date")
                    if ds and ds[:7] == ym_target:
                        logger.debug(f"Encontrado: {ds}")
                        return obs
                logger.debug(f"No se encontró observación para {ym_target}")
            else:
                # If target_date_str already is a date, map month to quarter-end month
                m2 = re.match(r"^(\d{4})-(0[1-9]|1[0-2])(-\d{2})?$", target_date_str)
                if m2:
                    year = int(m2.group(1))
                    month = int(m2.group(2))
                    quarter = ((month - 1) // 3) + 1
                    last_month = quarter * 3
                    ym_target = f"{year:04d}-{last_month:02d}"
                    logger.debug(f"Matching quarter (fecha explícita): buscando mes de cierre {ym_target}")
                    for obs in observations:
                        ds = obs.get("date")
                        if ds and ds[:7] == ym_target:
                            logger.debug(f"Encontrado: {ds}")
                            return obs

        # Year: target_date_str is "YYYY"
        elif gran == "year":
            y_target = target_date_str[:4]
            logger.debug(f"Matching year: buscando {y_target}")
            # Find last observation in that year
            year_obs = [o for o in observations if o.get("date", "")[:4] == y_target]
            if year_obs:
                return max(year_obs, key=lambda o: o.get("date", ""))

        # Fallback: try year-month match if target is in YYYY-MM-DD format
        if "-" in target_date_str and len(target_date_str) >= 7:
            ym_target = target_date_str[:7]
            logger.debug(f"Fallback: buscando {ym_target}")
            for obs in observations:
                ds = obs.get("date")
                if ds and ds[:7] == ym_target:
                    return obs
    except Exception as e:
        logger.warning(f"Error en _match_observation: {e}")
    return None


def _format_period_labels(date_str: Optional[str], freq: str) -> list[str]:
    """Return both long and short period labels as a list [long, short].
    - Quarterly: ["el 3er trimestre del 2025", "3T 2025"]
    - Monthly:   ["Marzo 2025", "Mar 2025"]
    """
    if not date_str:
        return ["--", "--"]
    try:
        y = int(date_str[:4])
        m = int(date_str[5:7])
        if freq in {"Q", "T"}:
            q = ((m - 1) // 3) + 1
            ordinal = {1: "1er", 2: "2do", 3: "3er", 4: "4to"}.get(q, f"{q}º")
            long_label = f"el {ordinal} trimestre del {y}"
            short_label = f"{q}T {y}"
            return [long_label, short_label]
        else:
            meses_es = [
                "", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
            ]
            meses_abrev = [
                "", "Ene", "Feb", "Mar", "Abr", "May", "Jun",
                "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"
            ]
            mes_nombre = meses_es[m] if 1 <= m <= 12 else str(m)
            mes_abrev = meses_abrev[m] if 1 <= m <= 12 else str(m)
            return [f"{mes_nombre} {y}", f"{mes_abrev} {y}"]
    except Exception:
        return [date_str or "--", date_str or "--"]


def _calculate_variation(
    obs: list,
    target_row: Optional[Dict[str, Any]],
    show_qoq: bool,
    freq: str,
) -> tuple[Optional[float], str, str]:
    """Calculate variation (QoQ if desestacionalizado, else YoY). Returns (value, label, key)."""
    if not target_row:
        return None, "Variación anual", "yoy_pct"
    
    freq_label = {"Q": "trimestral", "T": "trimestral", "M": "mensual"}.get(freq, "anual")
    
    if show_qoq:
        obs_sorted = sorted(obs, key=lambda o: o.get("date", ""))
        idx = next((i for i, o in enumerate(obs_sorted) if o.get("date") == target_row.get("date")), None)
        if idx is not None and idx > 0:
            prev = obs_sorted[idx - 1]
            try:
                v_now = float(target_row.get("value"))
                v_prev = float(prev.get("value"))
                if v_prev != 0:
                    var_value = 100.0 * (v_now - v_prev) / abs(v_prev)
                    return var_value, f"Variación {freq_label}", "qoq_pct"
            except Exception:
                pass
    
    yoy_val = target_row.get("yoy_pct")
    return yoy_val, f"Variación {freq_label if show_qoq else 'anual'}", "yoy_pct"


def _format_value(value: Any) -> str:
    """Format numeric value with thousand separators."""
    try:
        return f"{float(value):,.0f}".replace(",", "_").replace("_", ".")
    except Exception:
        return "--"


def _format_percentage(value: Any) -> str:
    """Format percentage value."""
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "--"


def _generate_csv_marker(
    row: Dict[str, Any],
    series_id: str,
    var_value: Optional[float],
    var_label: str,
    var_key: str,
) -> Iterable[str]:
    """Generate CSV download marker for export."""
    try:
        import pandas as _pd
        import tempfile
        
        export_map = {
            "date": "Periodo",
            "value": "Valor",
            var_key: var_label
        }
        export_row = {
            export_map[c]: row.get(c) if c != var_key else var_value
            for c in export_map if c in row or c == var_key
        }
        df_export = _pd.DataFrame([export_row])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="serie_", mode="w", encoding="utf-8") as tmp:
            df_export.to_csv(tmp, index=False)
            tmp_path = tmp.name
        
        filename = f"serie_{series_id}.csv"
        yield "##CSV_DOWNLOAD_START\n"
        yield f"path={tmp_path}\n"
        yield f"filename={filename}\n"
        yield "label=Descargar CSV\n"
        yield "mimetype=text/csv\n"
        yield "##CSV_DOWNLOAD_END\n"
    except Exception as e:
        logger.warning(f"No se pudo generar CSV para descarga: {e}")


def stream_data_flow(
    classification: Any,
    session_id: Optional[str] = None,
) -> Iterable[str]:
    """Fetch de datos y tabla."""

    # Debug: inspeccionar classification recibido
    logger.debug(f"[FLOW_DATA] classification type: {type(classification)}")
    logger.debug(f"[FLOW_DATA] classification hasattr 'intent': {hasattr(classification, 'intent')}")
    if hasattr(classification, "intent"):
        logger.debug(f"[FLOW_DATA] classification.intent = {getattr(classification, 'intent', None)}")
    if hasattr(classification, "__dict__"):
        logger.debug(f"[FLOW_DATA] classification.__dict__ = {classification.__dict__}")

    # Obtiene memoria de Redis
    redis_ctx = _get_context(session_id=session_id)
    ctx_map = _as_mapping(redis_ctx)
    facts = _as_mapping(ctx_map.get("facts"))
    try:
        logger.debug(
            "Facts cargados desde memoria: %s",
            list(facts.keys()) if isinstance(facts, dict) else facts,
        )
    except Exception:
        pass
    
    # Extraer entidades normalizados desde classification
    normalized = {}
    if classification is not None:
        normalized_raw = getattr(classification, "normalized", None)
        normalized = _as_mapping(normalized_raw)
    
    # Regla: si la consulta cambia el indicador, no usar memoria para llenar entidades
    normalized_indicator_key = None
    ind_obj = normalized.get("indicator")
    if isinstance(ind_obj, dict):
        normalized_indicator_key = str(
            (ind_obj.get("normalized") or ind_obj.get("label") or "")
        ).strip().lower()
    elif isinstance(ind_obj, str):
        normalized_indicator_key = ind_obj.strip().lower()

    facts_indicator_key = None
    if isinstance(facts, dict):
        facts_indicator_key = str(facts.get("indicator") or "").strip().lower()

    ignore_memory = (
        (
            normalized_indicator_key == "pib"
            and isinstance(facts_indicator_key, str)
            and ("imacec" in facts_indicator_key)
        )
        or (
            isinstance(normalized_indicator_key, str)
            and ("imacec" in normalized_indicator_key)
            and isinstance(facts_indicator_key, str)
            and ("pib" in facts_indicator_key)
        )
    )
    if ignore_memory:
        logger.debug("Ignorando memoria para entidades (cambio entre IMACEC↔PIB)")
    # Solicitud: no usar memoria para resolver la serie; solo la clasificación actual
    facts_for_resolvers = None

    # Resolver entidades desde clasificación y (opcional) Redis
    period_context = _resolve_period_context(normalized, facts_for_resolvers)
    indicator_context_val = _resolve_indicator_context(normalized, facts_for_resolvers)
    component_in_classification = False
    comp_obj = normalized.get("component")
    if isinstance(comp_obj, dict):
        comp_val = comp_obj.get("normalized") or comp_obj.get("label")
        if isinstance(comp_val, str) and comp_val.strip():
            component_in_classification = True
    elif isinstance(comp_obj, str) and comp_obj.strip():
        component_in_classification = True

    # No usar memoria para componente si falta en la clasificación
    facts_for_component = facts_for_resolvers if component_in_classification else None

    component_context_val = _resolve_component_context(normalized, facts_for_component)
    seasonality_context_val = _resolve_seasonality_context(normalized, facts_for_resolvers)

    # Detectar código de serie
    detection_result = detect_series_code(
        indicator=indicator_context_val,
        component=component_context_val,
        seasonality=seasonality_context_val,
    )
    
    series_id = detection_result.get("series_code")
    metadata = detection_result.get("metadata", {})

    if not series_id:
        logger.error("No se pudo determinar el código de serie")
        return

    # Determinar frecuencia objetivo desde el período normalizado
    freq = "M"  # default mensual
    if period_context and isinstance(period_context, dict):
        granularity = period_context.get("granularity")
        if granularity == "quarter":
            freq = "Q"
        elif granularity == "year":
            freq = "A"

    target_date = period_context.get("target_date") if period_context else None

    # Obtener datos de la serie 
    data = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            # yield f"Intento {attempt + 1}/{max_retries + 1}...\n\n"
            data = get_series_api_rest_bcch(
                series_id=series_id,
                target_date=target_date,
                target_frequency=freq,
                agg="avg"
            )
            # Exito
            break
        except Exception as e:
            logger.warning(f"Error en intento {attempt + 1}: {type(e).__name__}: {e}")
            if attempt < max_retries:
                delay = 1.0 * (2 ** attempt)
                # yield f"Reintentando en {delay}s...\n\n"
                time.sleep(delay)
            else:
                logger.error(f"Falló después de {max_retries + 1} intentos")
                return
    
    if not data:
        logger.warning("La API no devolvió datos")
        return

    api_meta = data.get("meta", {}) or {}
    
    # Obtener observaciones
    obs = data.get("observations", [])

    if obs:
        
        # Si hay target_date, buscar observación que coincida
        target_date_str = period_context.get("target_date") if period_context else None
        granularity = period_context.get("granularity") if period_context else None
        
        logger.debug(f"Buscando observación con target_date={target_date_str}, granularity={granularity}")
        obs_match = _match_observation(obs, target_date_str, granularity)
        obs_latest = max(obs, key=lambda o: o.get("date", "")) if obs else None
        
        if obs_match:
            logger.debug(f"Observación encontrada: {obs_match.get('date')}")
        else:
            logger.debug(f"No se encontró observación para target_date, usando último período: {obs_latest.get('date') if obs_latest else None}")
        
        chosen_row = obs_match or obs_latest
        
        show_qoq = seasonality_context_val and 'desestacionalizado' in seasonality_context_val.lower()

        # Calcular variación
        var_value, var_label, var_key = _calculate_variation(obs, chosen_row, show_qoq, freq)

        # Seleccionar y renderizar plantilla de mensaje
        chosen_date = chosen_row.get("date") if chosen_row else None
        freq_label = {"Q": "trimestral", "T": "trimestral", "M": "mensual"}.get(freq, "anual")
        
        # Datos finales
        date_raw = chosen_row.get("date")
        value = chosen_row.get("value")
        
        # Obtiene el formato de mostrar el periodo
        # TODO: normalizar classification.frequency, y usar si es dada por el usuario
        if (indicator_context_val or "").strip().lower() == "pib":
            period_labels = _format_period_labels(date_raw, "Q")
        else:
            period_labels = _format_period_labels(date_raw, freq)
            
        # Nombre final del indicador: indicador + componente (si aplica) + estacionalidad (si aplica)
        indicator_parts: list[str] = []
        if isinstance(indicator_context_val, str) and indicator_context_val.strip():
            indicator_parts.append(indicator_context_val.upper().strip())
        if isinstance(component_context_val, str) and component_context_val.strip():
            indicator_parts.append(component_context_val.strip())
        if isinstance(seasonality_context_val, str) and seasonality_context_val.strip():
            indicator_parts.append(seasonality_context_val.strip())
        final_indicator_name = " ".join(indicator_parts) if indicator_parts else "indicador"
        
        # Flags sobre la posición de lastdate respecto al rango disponible
        position = str(api_meta.get("lastdate_position") or "").strip().lower()
        last_available_date = api_meta.get("last_available_date")
        first_available_date = api_meta.get("first_available_date")

        # Etiquetas legibles para límites de la serie
        if (indicator_context_val or "").strip().lower() == "pib":
            last_available_labels = _format_period_labels(last_available_date, "Q") if last_available_date else ["--", "--"]
            first_available_labels = _format_period_labels(first_available_date, "Q") if first_available_date else ["--", "--"]
        else:
            last_available_labels = _format_period_labels(last_available_date, freq) if last_available_date else ["--", "--"]
            first_available_labels = _format_period_labels(first_available_date, freq) if first_available_date else ["--", "--"]

        # Construir prompt para el LLM con los datos disponibles
        llm_prompt_parts = []
        
        # Contexto específico según la posición del período consultado
        if position == "gt_latest":
            # Período consultado está después del último dato disponible
            llm_prompt_parts.append("SITUACIÓN: El usuario preguntó por un período que aún no tiene datos disponibles.")
            llm_prompt_parts.append(f"Redacta una respuesta breve (máximo 2 oraciones) que:")
            llm_prompt_parts.append(f"1. Indique claramente que ese período no tiene datos disponibles todavía")
            llm_prompt_parts.append(f"2. Mencione que los datos más recientes del {final_indicator_name} son de {last_available_labels[0]}")
            if var_value is not None:
                llm_prompt_parts.append(f"3. Opcionalmente menciona que en ese último período registró una variación {freq_label if show_qoq else 'anual'} de {var_value:.1f}% (esto es VARIACIÓN PORCENTUAL, no el valor del índice)")
        
        elif position == "lt_first":
            # Período consultado está antes del primer dato disponible
            llm_prompt_parts.append("SITUACIÓN: El usuario preguntó por un período anterior al inicio de la serie.")
            llm_prompt_parts.append(f"Redacta una respuesta breve (máximo 2 oraciones) que:")
            llm_prompt_parts.append(f"1. Indique que no hay datos para ese período porque es anterior al inicio de la serie")
            llm_prompt_parts.append(f"2. Mencione que la serie del {final_indicator_name} comienza en {first_available_labels[0]}")
            if var_value is not None:
                llm_prompt_parts.append(f"3. Opcionalmente menciona que en ese primer período registró una variación {freq_label if show_qoq else 'anual'} de {var_value:.1f}% (esto es VARIACIÓN PORCENTUAL, no el valor del índice)")
        
        elif position == "eq_latest":
            # Período consultado es exactamente el último dato disponible
            llm_prompt_parts.append("SITUACIÓN: El usuario preguntó por el período más reciente disponible.")
            llm_prompt_parts.append(f"Redacta una respuesta breve (máximo 2 oraciones) informando:")
            llm_prompt_parts.append(f"- Indicador: {final_indicator_name}")
            llm_prompt_parts.append(f"- Período: {period_labels[0]}")
            if var_value is not None:
                llm_prompt_parts.append(f"- Variación {freq_label if show_qoq else 'anual'}: {var_value:.1f}%")
                llm_prompt_parts.append(f"IMPORTANTE: {var_value:.1f}% es la VARIACIÓN PORCENTUAL (crecimiento/caída), NO menciones el valor absoluto del índice")
            llm_prompt_parts.append(f"- Opcional: menciona que este es el dato más reciente")
        
        elif not var_value:
            # Caso especial: hay valor pero no hay variación (primer período de la serie)
            llm_prompt_parts.append("SITUACIÓN: El usuario preguntó por un período que tiene datos pero no tiene variación calculable.")
            llm_prompt_parts.append(f"Redacta una respuesta breve (máximo 2 oraciones) que:")
            llm_prompt_parts.append(f"1. Mencione que en {period_labels[0]} el {final_indicator_name} registró un índice de {_format_value(value)} puntos")
            llm_prompt_parts.append(f"2. Explique brevemente que no hay variación porcentual disponible porque no existe un período anterior para comparar")
        
        else:
            # Caso normal: período dentro del rango disponible
            llm_prompt_parts.append("SITUACIÓN: El usuario preguntó por un período que tiene datos disponibles.")
            llm_prompt_parts.append(f"Redacta una respuesta breve (máximo 2 oraciones) informando:")
            llm_prompt_parts.append(f"- Indicador: {final_indicator_name}")
            llm_prompt_parts.append(f"- Período: {period_labels[0]}")
            if var_value is not None:
                llm_prompt_parts.append(f"- Variación {freq_label if show_qoq else 'anual'}: {var_value:.1f}%")
                llm_prompt_parts.append(f"IMPORTANTE: {var_value:.1f}% es la VARIACIÓN PORCENTUAL (crecimiento/caída respecto al período anterior), NO es el valor absoluto del índice")
        
        llm_prompt_parts.append("\nREQUISITOS DE ESTILO:")
        llm_prompt_parts.append("- Usa un tono conversacional y directo, como si hablaras con un colega")
        llm_prompt_parts.append("- Puedes mencionar 'Base de Datos Estadísticos' o 'BDE' o 'Banco Central', pero varía la forma y no lo uses siempre")
        llm_prompt_parts.append("- Varía la estructura: no empieces siempre igual (alterna 'En [período]...', 'Durante [período]...', 'Los datos de [período]...', etc.)")
        llm_prompt_parts.append("- No des opiniones, análisis económico ni explicaciones extras")
        llm_prompt_parts.append("- Sé conciso: máximo 2 oraciones")
        
        llm_prompt = "\n".join(llm_prompt_parts)
        
        # Generar respuesta con el LLM
        try:
            llm = build_llm(streaming=True, temperature=0.7, mode="fallback")
            for chunk in llm.stream(llm_prompt, history=[], intent_info=None):
                yield chunk
            yield "\n"
        except Exception as e:
            logger.warning(f"Error generando con LLM, usando plantilla: {e}")
            
            # Inicializar payload con todos los componentes desde el principio
            payload = {
                "indicator": final_indicator_name,
                "value": _format_value(value),
                "var_label": freq_label if show_qoq else "anual",
                "var_value": float(var_value) if var_value is not None else None,
                "period_label": period_labels[0],
                "last_available_label": last_available_labels[0], # Flags de posición
                "first_available_label": first_available_labels[0], # Flags de posición
            }
        
            # Fallback a plantillas si el LLM falla
            tmpl_ctx = {
                "has_indicator": bool(final_indicator_name),
                "has_value": value is not None,
                "has_var_value": var_value is not None,
                "has_period": bool(chosen_date),
                "has_seasonality": bool(seasonality_context_val),
                "no_data": False,
                "lastdate_position": position,
                "value": value,
            }
            message = render_template(select_template(tmpl_ctx), payload)
            yield f"{message}\n"
        
        # Tabla markdown
        yield f"Periodo | Valor | {var_label}\n"
        yield f"--------|-------|{'-'*len(var_label)}\n"
        yield f"{period_labels[1]} | {_format_value(value)} * | {_format_percentage(var_value)}\n"
        # yield "-- | -- | --\n"
        yield "\n"
        
        # Fuente (link corto)
        yield r"\* _Índice_" + "\n\n" if indicator_context_val == "imacec" else r"\* _Miles de millones de pesos_" + "\n\n"
        yield f"**Fuente:** Banco Central de Chile (BDE) — [Ver serie en la BDE]({detection_result.get('metadata', {}).get('source_url')})"
        # yield "\n\n" + api_meta.get("descripEsp", "")

        # CSV download marker
        if chosen_row:
            yield from _generate_csv_marker(chosen_row, series_id, var_value, var_label, var_key)
        
        # Sugerencias se generan globalmente en memory_node para todas las rutas
    
    return
