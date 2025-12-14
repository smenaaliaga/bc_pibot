from __future__ import annotations
import logging
import re
import time
from typing import Any, Iterable, Optional

from config import get_settings

try:
    from orchestrator.data.get_series import detect_series_code 
except Exception:
    detect_series_code = None 
    
logger = logging.getLogger(__name__)
_settings = get_settings()

def stream_data_flow(
    classification: Any,
    question: str,
    history_text: str,
) -> Iterable[str]:
    """Fetch de datos y tabla."""
    
    # Extraer info de JointBERT si está disponible
    entities = getattr(classification, "entities", None)
    normalized = getattr(classification, "normalized", None)
    intent_value = getattr(classification, "intent", None)
    
    # Extraer año desde normalized.period 
    year = None
    if normalized and isinstance(normalized, dict):
        period_norm = normalized.get('period')
        if period_norm and isinstance(period_norm, dict):
            start_date = period_norm.get('start_date')
            if start_date:
                try:
                    if hasattr(start_date, 'year'):
                        year = start_date.year
                    else:
                        year = int(str(start_date)[:4])
                    # yield f"**Año extraído de normalized:** {year}\n\n"
                except Exception as e:
                    logger.warning(f"Error extrayendo year: {e}")
    
    # Fallback: año actual
    if year is None:
        try:
            year = int(time.strftime("%Y"))
            # yield f"**Año por defecto (actual):** {year}\n\n"
        except Exception:
            logger.error("No se pudo determinar el año")
            return
    
    # 2. Detectar serie usando el módulo series_detector
    if detect_series_code is None:
        logger.error("No se pudo importar detect_series_code")
        return
    
    detection_result = detect_series_code(
        normalized=normalized,
        entities=entities
    )
    
    series_id = detection_result.get("series_code")
    metadata = detection_result.get("metadata", {})

    if not series_id:
        logger.error("No se pudo determinar el código de serie")
        return
    
    # Determinar frecuencia (por ahora mensual por defecto)
    freq = "M"
    
    # 4. Calcular rango usando el período normalizado
    # Extraer start_date del período normalizado
    start_date_raw = None
    if normalized and isinstance(normalized, dict):
        period_norm = normalized.get('period')
        if period_norm and isinstance(period_norm, dict):
            start_date_raw = period_norm.get('start_date')
    
    # Calcular lastdate y firstdate
    if start_date_raw:
        # Usar start_date como lastdate
        from datetime import datetime, timedelta
        def subtract_years_months(dt, years=0, months=0):
            # Resta años y meses a un datetime.date o datetime.datetime
            year = dt.year - years
            month = dt.month - months
            while month <= 0:
                month += 12
                year -= 1
            # Ajustar día si el mes resultante no tiene ese día
            day = min(dt.day, [31,29 if year%4==0 and (year%100!=0 or year%400==0) else 28,31,30,31,30,31,31,30,31,30,31][month-1])
            return dt.replace(year=year, month=month, day=day)

        if hasattr(start_date_raw, 'strftime'):
            lastdate = start_date_raw.strftime('%Y-%m-%d')
            try:
                firstdate_dt = subtract_years_months(start_date_raw, years=1, months=3)
                firstdate = firstdate_dt.strftime('%Y-%m-%d')
            except Exception:
                # Fallback: solo restar año
                year_minus_one = start_date_raw.year - 1
                firstdate = start_date_raw.replace(year=year_minus_one).strftime('%Y-%m-%d')
        else:
            # Si es string, parsearlo
            lastdate = str(start_date_raw)
            try:
                dt = datetime.strptime(str(start_date_raw)[:10], '%Y-%m-%d')
                firstdate_dt = subtract_years_months(dt, years=1, months=3)
                firstdate = firstdate_dt.strftime('%Y-%m-%d')
            except Exception:
                # Fallback: usar año extraído anteriormente
                firstdate = f"{year-1}-01-01"
                lastdate = f"{year}-12-31"
    else:
        # Fallback: usar año extraído anteriormente
        firstdate = f"{year-1}-01-01"
        lastdate = f"{year}-12-31"

    # Llamar a get_series_api_rest_bcch
    try:
        from orchestrator.data.get_series import get_series_api_rest_bcch
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
        # Mostrar solo el periodo igual a start_date, o el máximo disponible si no existe
        # Buscar start_date como string (puede ser datetime o string)
        start_date_str = None
        if 'start_date' in locals():
            if hasattr(start_date, 'strftime'):
                start_date_str = start_date.strftime('%Y-%m-%d')
            else:
                start_date_str = str(start_date)
        else:
            start_date_str = None

        # Buscar observación con periodo igual a start_date
        obs_match = None
        if start_date_str:
            for o in obs:
                if o.get('date') == start_date_str:
                    obs_match = o
                    break
        # Si no existe, usar el máximo periodo disponible
        if not obs_match and obs:
            obs_match = max(obs, key=lambda o: o.get('date', ''))

        # Encontrar el máximo periodo disponible
        obs_max = max(obs, key=lambda o: o.get('date', '')) if obs else None
        max_date = obs_max.get('date') if obs_max else None

        # Mensaje resumen según condición
        resumen_emitido = False
        if max_date and start_date_str:
            if max_date < start_date_str:
                # Usar el máximo periodo (no hay dato para start_date)
                yoy_pct = obs_max.get('yoy_pct') if obs_max else None
                date = obs_max.get('date') if obs_max else None
                if yoy_pct is not None and date:
                    yield f"La última variación anual disponible es {float(yoy_pct):.1f}% (período {date}).\n"
                    resumen_emitido = True
            elif obs_match and max_date == start_date_str:
                # Si el máximo periodo es exactamente el start_date, mostrar el mensaje específico
                yoy_pct = obs_match.get('yoy_pct')
                date = obs_match.get('date')
                if yoy_pct is not None and date:
                    try:
                        mes = int(date[5:7])
                        anio = int(date[:4])
                        meses_es = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
                        mes_nombre = meses_es[mes-1].capitalize()
                    except Exception:
                        mes_nombre = date[5:7]
                        anio = date[:4]
                    yield f"La variacion anual para el mes de {mes_nombre} del {anio} fue {float(yoy_pct):.1f}%\n"
                    resumen_emitido = True
        elif obs_match:
            yoy_pct = obs_match.get('yoy_pct')
            date = obs_match.get('date')
            if yoy_pct is not None and date:
                try:
                    mes = int(date[5:7])
                    anio = int(date[:4])
                    import calendar
                    mes_nombre = calendar.month_name[mes].capitalize()
                except Exception:
                    mes_nombre = date[5:7]
                    anio = date[:4]
                yield f"La variacion anual para el mes de {mes_nombre} del {anio} fue {float(yoy_pct):.1f}%\n"
                resumen_emitido = True

        # Mostrar la tabla de dato solicitado
        yield "Periodo | Valor | Variación anual\n"
        yield "--------|-------|-----------------\n"
        if obs_match:
            date = obs_match.get('date')
            value = obs_match.get('value')
            yoy_pct = obs_match.get('yoy_pct')
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
