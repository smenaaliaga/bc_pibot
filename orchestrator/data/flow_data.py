from __future__ import annotations
import logging
import time
from typing import Any, Dict, Iterable, Optional

from orchestrator.memory.memory_adapter import MemoryAdapter
from orchestrator.data.get_series import detect_series_code 
from orchestrator.data.get_data_serie import get_series_api_rest_bcch
from orchestrator.data.templates import select_template, render_template
    
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
    """Resuelve período: primero clasificación, luego Redis (requiere firstdate/lastdate)."""
    period_obj = normalized.get("period")
    if isinstance(period_obj, dict) and period_obj.get("firstdate") and period_obj.get("lastdate"):
        return period_obj
    if isinstance(facts, dict):
        facts_period = facts.get("period")
        if isinstance(facts_period, dict) and facts_period.get("firstdate") and facts_period.get("lastdate"):
            return facts_period
        if isinstance(facts_period, dict):
            logger.debug("facts.period presente pero sin 'firstdate/lastdate'; usando ventana auto")
    return None


def _resolve_indicator_context(normalized: Dict[str, Any], facts: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resuelve indicador: primero clasificación, luego Redis."""
    indicator_obj = normalized.get("indicator")
    if isinstance(indicator_obj, dict):
        val = indicator_obj.get("normalized")
        if isinstance(val, str) and val.strip():
            return val
    if isinstance(facts, dict):
        ind = facts.get("indicator")
        if isinstance(ind, str) and ind.strip():
            return ind.strip()
    return None


def _resolve_component_context(normalized: Dict[str, Any], facts: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resuelve componente: primero clasificación, luego Redis."""
    comp_obj = normalized.get("component")
    if isinstance(comp_obj, dict):
        val = comp_obj.get("normalized")
        if isinstance(val, str) and val.strip():
            return val
    if isinstance(facts, dict):
        comp = facts.get("component")
        if isinstance(comp, str) and comp.strip():
            return comp.strip()
    return None


def _resolve_seasonality_context(normalized: Dict[str, Any], facts: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resuelve estacionalidad: primero clasificación, luego Redis."""
    seasonality_obj = normalized.get("seasonality")
    if isinstance(seasonality_obj, dict):
        val = seasonality_obj.get("normalized") or seasonality_obj.get("label")
        if isinstance(val, str) and val.strip():
            return val.strip()
    elif isinstance(seasonality_obj, str) and seasonality_obj.strip():
        return seasonality_obj.strip()
    if isinstance(facts, dict):
        seas = facts.get("seasonality")
        if isinstance(seas, str) and seas.strip():
            return seas.strip()
    return None


def _match_observation(observations: list, candidates: list[str]) -> Optional[Dict[str, Any]]:
    if not observations or not candidates:
        return None
    for candidate in candidates:
        for obs in observations:
            date_str = obs.get("date")
            if not date_str:
                continue
            if date_str == candidate:
                return obs
            if date_str[:7] == candidate[:7]:
                return obs
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
        return f"{float(value):,.2f}".replace(",", "_").replace("_", ".")
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

    # Obtiene memoria de Redis
    redis_ctx = _get_context(session_id=session_id)
    ctx_map = _as_mapping(redis_ctx)
    facts = _as_mapping(ctx_map.get("facts"))
    
    # Extraer entidades normalizados desde classification
    normalized = {}
    if classification is not None:
        normalized_raw = getattr(classification, "normalized", None)
        normalized = _as_mapping(normalized_raw)
    
    # Resolver entidades desde clasificación y Redis (helpers)
    period_context = _resolve_period_context(normalized, facts)
    indicator_context_val = _resolve_indicator_context(normalized, facts)
    component_context_val = _resolve_component_context(normalized, facts)
    seasonality_context_val = _resolve_seasonality_context(normalized, facts)

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
        granularity = str(period_context.get("granularity") or period_context.get("period_type") or "").lower()
        if granularity in ("quarter", "q", "t", "trimestre", "trimestral"):
            freq = "Q"
        elif granularity in ("year", "y", "anual", "annual"):
            freq = "A"

    if period_context and period_context.get("firstdate") and period_context.get("lastdate"):
        firstdate = period_context["firstdate"]
        lastdate = period_context["lastdate"]
    else:
        # Sin período explícito: dejar "auto" en la API (None)
        firstdate = None
        lastdate = None

    # Obtener datos de la serie 
    data = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            # yield f"Intento {attempt + 1}/{max_retries + 1}...\n\n"
            data = get_series_api_rest_bcch(
                series_id=series_id,
                firstdate=firstdate,
                lastdate=lastdate,
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
        
        candidates = period_context["match_candidates"] if period_context else []
        obs_match = _match_observation(obs, candidates)
        obs_latest = max(obs, key=lambda o: o.get("date", "")) if obs else None
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
        
        payload = {
            "indicator": (indicator_context_val or "indicador").upper(),
            "var_label": freq_label if show_qoq else "anual",
            "var_value": float(var_value) if var_value is not None else None,
            "period_label": period_labels[0]
        }
        tmpl_ctx = {
            "has_indicator": bool(indicator_context_val),
            "has_value": var_value is not None,
            "has_period": bool(chosen_date),
            "has_seasonality": bool(seasonality_context_val),
            "no_data": False,
        }
        message = render_template(select_template(tmpl_ctx), payload)
        yield f"{message}\n"
        
        # Tabla markdown
        yield f"Periodo | Valor | {var_label}\n"
        yield f"--------|-------|{'-'*len(var_label)}\n"
        yield f"{period_labels[1]} | {_format_value(value)} | {_format_percentage(var_value)}\n"
        # yield "-- | -- | --\n"
        yield "\n"
        
        # Fuente (link corto)
        yield f"**Fuente:** Banco Central de Chile (BDE) — [Ver serie en la BDE]({detection_result.get('metadata', {}).get('source_url')})"

        # CSV download marker
        if chosen_row:
            yield from _generate_csv_marker(chosen_row, series_id, var_value, var_label, var_key)
    
    return
