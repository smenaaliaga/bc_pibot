from __future__ import annotations
import calendar
import logging
import re
import time
from datetime import date, datetime
from typing import Any, Dict, Iterable, Optional

from config import get_settings

try:
    from orchestrator.data.get_series import detect_series_code 
except Exception:
    detect_series_code = None 
    
logger = logging.getLogger(__name__)
_settings = get_settings()


def _coerce_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text[: len(fmt)], fmt)
            if fmt == "%Y":
                return date(parsed.year, 1, 1)
            if fmt == "%Y-%m":
                return date(parsed.year, parsed.month, 1)
            return parsed.date()
        except Exception:
            continue
    return None


def _end_of_period(value: date, granularity: str) -> date:
    gran = (granularity or "").lower()
    if gran == "month":
        last_day = calendar.monthrange(value.year, value.month)[1]
        return value.replace(day=last_day)
    if gran == "quarter":
        quarter = ((value.month - 1) // 3) + 1
        last_month = quarter * 3
        last_day = calendar.monthrange(value.year, last_month)[1]
        return value.replace(month=last_month, day=last_day)
    if gran == "year":
        return value.replace(month=12, day=31)
    return value


def _subtract_months(value: date, months: int) -> date:
    total_months = value.year * 12 + (value.month - 1) - max(months, 0)
    year = total_months // 12
    month = total_months % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(value.day, last_day)
    return value.replace(year=year, month=month, day=day)


def _format_date(value: Optional[date]) -> Optional[str]:
    return value.strftime("%Y-%m-%d") if value else None


def _date_from_period_key(period_key: Any) -> Optional[str]:
    if not period_key:
        return None
    key = str(period_key).strip()
    if re.match(r"^\d{4}-\d{2}$", key):
        try:
            dt = datetime.strptime(key, "%Y-%m").date()
            return _format_date(dt)
        except Exception:
            return None
    if re.match(r"^\d{4}-Q[1-4]$", key.upper()):
        year = int(key[:4])
        quarter = int(key[-1])
        month = quarter * 3
        last_day = calendar.monthrange(year, month)[1]
        return f"{year:04d}-{month:02d}-{last_day:02d}"
    return None


def _build_period_context(normalized: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    period_obj: Any = None
    if isinstance(normalized, dict):
        period_obj = normalized.get("period")
    if not period_obj:
        return None
    period_dict = period_obj if isinstance(period_obj, dict) else getattr(period_obj, "__dict__", {})
    raw_start = period_dict.get("start_date") or period_dict.get("startDate")
    raw_end = period_dict.get("end_date") or period_dict.get("endDate")
    gran = str(period_dict.get("granularity") or period_dict.get("period_type") or "").lower()
    start = _coerce_date(raw_start)
    end = _coerce_date(raw_end)
    if not start and end:
        start = end
    if start and not end:
        end = _end_of_period(start, gran)
    if not start and not end:
        return None
    target = end or start
    buffer_months = {"month": 15, "quarter": 24, "year": 60}.get(gran, 18)
    first_candidate = _subtract_months(target, buffer_months)
    firstdate = _format_date(first_candidate)
    lastdate = _format_date(target)
    candidates = []
    for dt in (start, end):
        formatted = _format_date(dt)
        if formatted:
            candidates.append(formatted)
    key_date = _date_from_period_key(period_dict.get("period_key"))
    if key_date:
        candidates.append(key_date)
    label = period_dict.get("period_key") or period_dict.get("label") or lastdate or firstdate
    return {
        "start": start,
        "end": end,
        "granularity": gran,
        "firstdate": firstdate,
        "lastdate": lastdate,
        "match_candidates": [c for c in candidates if c],
        "label": label,
    }


def _as_mapping(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "__dict__"):
        try:
            return dict(payload.__dict__)
        except Exception:
            return {}
    return {}


def _coerce_meta_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("standard_name", "normalized", "original", "text_normalized", "label"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _has_indicator_info(normalized: Dict[str, Any], entities: Dict[str, Any]) -> bool:
    indicator = _coerce_meta_value(normalized.get("indicator"))
    if indicator:
        return True
    indicator = _coerce_meta_value(entities.get("indicator"))
    return bool(indicator)


def _has_sector_info(normalized: Dict[str, Any]) -> bool:
    if _coerce_meta_value(normalized.get("sector")):
        return True
    if _coerce_meta_value(normalized.get("component")):
        return True
    return False


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

def stream_data_flow(
    classification: Any,
    question: str,
    history_text: str,
    *,
    indicator_context: Optional[Dict[str, str]] = None,
) -> Iterable[str]:
    """Fetch de datos y tabla."""
    
    # Extraer info de JointBERT si está disponible
    entities_raw = getattr(classification, "entities", None)
    normalized_raw = getattr(classification, "normalized", None)
    entities = entities_raw if isinstance(entities_raw, dict) else {}
    normalized = normalized_raw if isinstance(normalized_raw, dict) else _as_mapping(normalized_raw)
    period_context = _build_period_context(normalized)

    year = None
    if period_context and period_context.get("start"):
        year = period_context["start"].year  # type: ignore[index]
    if year is None:
        try:
            year = int(time.strftime("%Y"))
        except Exception:
            logger.error("No se pudo determinar el año")
            return

    # 2. Detectar serie usando el módulo series_detector
    if detect_series_code is None:
        logger.error("No se pudo importar detect_series_code")
        return
    
    indicator_override = None
    sector_override = None
    if indicator_context:
        if not _has_indicator_info(normalized, entities):
            indicator_override = indicator_context.get("indicator")
        if not _has_sector_info(normalized):
            sector_override = indicator_context.get("sector") or indicator_context.get("component")

    detection_result = detect_series_code(
        normalized=normalized,
        entities=entities,
        indicator=indicator_override,
        sector=sector_override,
        component=sector_override,
    )
    
    series_id = detection_result.get("series_code")
    metadata = detection_result.get("metadata", {})

    if not series_id:
        logger.error("No se pudo determinar el código de serie")
        return
    
    # Determinar frecuencia (por ahora mensual por defecto)
    freq = "M"

    if period_context and period_context.get("firstdate") and period_context.get("lastdate"):
        firstdate = period_context["firstdate"]
        lastdate = period_context["lastdate"]
    else:
        firstdate = f"{year-1}-01-01"
        lastdate = f"{year}-12-31"

    # Llamar a get_series_api_rest_bcch
    try:
        from orchestrator.data.get_data_serie import get_series_api_rest_bcch
    except Exception as e:
        logger.error(f"No se pudo importar get_series_api_rest_bcch: {e}")
        return
    
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
            # Si llegamos aquí, tuvo éxito
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
    
    # Obtener observaciones
    obs = data.get("observations", [])

    if obs:
        candidates = period_context["match_candidates"] if period_context else []
        obs_match = _match_observation(obs, candidates)
        obs_latest = max(obs, key=lambda o: o.get("date", "")) if obs else None

        mensaje_emitido = False
        if period_context and obs_match:
            date = obs_match.get("date")
            yoy_pct = obs_match.get("yoy_pct")
            if date and yoy_pct is not None:
                try:
                    mes = int(date[5:7])
                    anio = int(date[:4])
                    mes_nombre = calendar.month_name[mes].capitalize()
                except Exception:
                    mes_nombre = date[5:7]
                    anio = date[:4]
                yield f"La variación anual para {mes_nombre} de {anio} fue {float(yoy_pct):.1f}%\n"
                mensaje_emitido = True
        if not mensaje_emitido and obs_latest:
            date = obs_latest.get("date")
            yoy_pct = obs_latest.get("yoy_pct")
            if date and yoy_pct is not None:
                yield f"La última variación anual disponible es {float(yoy_pct):.1f}% (período {date}).\n"

        yield "Periodo | Valor | Variación anual\n"
        yield "--------|-------|-----------------\n"
        target_row = obs_match or obs_latest
        if target_row:
            date = target_row.get("date")
            value = target_row.get("value")
            yoy_pct = target_row.get("yoy_pct")
            val_fmt = f"{float(value):,.2f}".replace(",", "_").replace("_", ".") if value is not None else "--"
            yoy_fmt = f"{float(yoy_pct):.1f}%" if yoy_pct is not None else "--"
            yield f"{date} | {val_fmt} | {yoy_fmt}\n"
        else:
            yield "-- | -- | --\n"
        yield "\n"

    # Mostrar metadatos de la serie
    if metadata:
        yield f"- Código: `{series_id}`\n"
        yield f"- Título: {metadata.get('title', '')}\n"
        # Frecuencia: buscar en varios campos posibles
        freq = metadata.get('freq_effective') or metadata.get('default_frequency') or metadata.get('original_frequency') or metadata.get('frequency') or ''
        yield f"- Frecuencia: {freq}\n"
        yield f"- Unidad: {metadata.get('unit', '')}\n"

        # Gráfico: mostrar siempre el campo, aunque esté vacío
        grafico_url = metadata.get('source_url', '')
        yield f"- Gráfico: {grafico_url}\n"

        # Metodología: mostrar siempre el campo, aunque esté vacío
        metodologia = ''
        notes = metadata.get('notes', {})
        if isinstance(notes, dict):
            metodologia = notes.get('metodologia', '')
            
        if metodologia:
            if isinstance(metodologia, str) and metodologia.startswith('http'):
                metodologia_str = f"[Ver documento]({metodologia})"
            else:
                metodologia_str = metodologia
        else:
            metodologia_str = ''
        yield f"- Metodología: {metodologia_str}\n"
        yield "\n"

    
    return
