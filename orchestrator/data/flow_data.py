from __future__ import annotations
import logging
from typing import Any, Dict, Iterable, Optional

from orchestrator.llm.llm_adapter import build_llm
    
logger = logging.getLogger(__name__)


def _format_period_labels(date_str: Optional[str], freq: str) -> list[str]:
    """Return both long and short period labels as a list [long, short].
    - Quarterly: ["el 3er trimestre del 2025", "3T 2025"]
    - Monthly:   ["Marzo 2025", "Mar 2025"]
    
    Expects date_str in format "DD-MM-YYYY" or "YYYY-MM-DD"
    """
    if not date_str:
        return ["--", "--"]
    try:
        # Detectar formato: si comienza con dos dígitos menores a 32, es DD-MM-YYYY
        parts = date_str.split("-")
        if len(parts) == 3:
            if int(parts[0]) > 31:  # YYYY-MM-DD format
                y = int(parts[0])
                m = int(parts[1])
            else:  # DD-MM-YYYY format
                d = int(parts[0])
                m = int(parts[1])
                y = int(parts[2])
        else:
            return [date_str or "--", date_str or "--"]
        
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
    all_series_data = payload.get("all_series_data")  # Lista con todas las series (para contribución con activity=none)
    source_url = payload.get("source_url")  # URL de la fuente
    
    if not series_id:
        logger.error("[STREAM_DATA_FLOW] No se encontró 'series' en payload")
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

    # Para range: mostrar todas las observaciones. Para latest/point: solo la última
    if req_form == "range":
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
    
    # Para range: mostrar rango de fechas usando parsed_range si está disponible; para latest: usar parsed_point o fecha de observación
    if req_form == "range":
        if parsed_range:
            start_str, end_str = parsed_range
            start_labels = _format_period_labels(start_str, freq)
            end_labels = _format_period_labels(end_str, freq)
            date_range_label = f"desde {start_labels[0]} hasta {end_labels[0]}"
            logger.info("[STREAM_DATA_FLOW] Rango (parsed) | %s | %d observaciones", date_range_label, len(obs_to_show))
        else:
            first_date = obs_to_show[0].get("date", "") if obs_to_show else ""
            last_date = obs_to_show[-1].get("date", "") if obs_to_show else ""
            first_labels = _format_period_labels(first_date, freq)
            last_labels = _format_period_labels(last_date, freq)
            date_range_label = f"desde {first_labels[0]} hasta {last_labels[0]}"
            logger.info("[STREAM_DATA_FLOW] Rango (observaciones) | %s | %d observaciones", date_range_label, len(obs_to_show))
    else:
        if parsed_point:
            period_labels = _format_period_labels(parsed_point, freq)
            date_range_label = period_labels[0]
            logger.info("[STREAM_DATA_FLOW] Renderizando (parsed_point) | date=%s", parsed_point)
        else:
            date_raw = obs_to_show[0].get("date", "") if obs_to_show else ""
            period_labels = _format_period_labels(date_raw, freq)
            date_range_label = period_labels[0]
            logger.info("[STREAM_DATA_FLOW] Renderizando (observación) | date=%s", date_raw)

    # Etiqueta para mostrar en prompts: para latest usar texto genérico
    display_period_label = "el último período disponible" if req_form == "latest" else date_range_label
    
    # Construir prompt para el LLM
    if req_form == "range":
        llm_prompt_parts = []
        
        # Solo incluir la última variación (o contribución)
        if obs_to_show:
            last_row = obs_to_show[-1]
            last_date_str = last_row.get("date", "")
            seasonality_lower = (seasonality_context_val or "").lower()
            prefer_yoy = seasonality_lower == "nsa"

            # Seleccionar métrica de contribución o variación
            if is_contribution:
                last_var = last_row.get("value")
                contrib_label = "contribución interanual" if prefer_yoy else "contribución vs período anterior"
            else:
                contrib_value = None
                contrib_label = "contribución"
                if prefer_yoy and last_row.get("yoy") is not None:
                    contrib_value = last_row.get("yoy")
                    contrib_label = "contribución interanual"
                elif last_row.get("prev_period") is not None:
                    contrib_value = last_row.get("prev_period")
                    contrib_label = "contribución vs período anterior"
                elif last_row.get("yoy") is not None:
                    contrib_value = last_row.get("yoy")
                    contrib_label = "contribución interanual"
                last_var = contrib_value if contrib_value is not None else last_row.get("yoy") or last_row.get("prev_period")
            last_period_label = _format_period_labels(last_date_str, freq)[0]
            
            if is_contribution:
                llm_prompt_parts.append(f"El usuario preguntó por la contribución de {final_indicator_name} en el período: {display_period_label}.")
                llm_prompt_parts.append(f"Cierre: {last_period_label} registró {contrib_label} de {_format_percentage(last_var)} (1 decimal).")
            else:
                llm_prompt_parts.append(f"El usuario preguntó por {final_indicator_name} en el período: {display_period_label}.")
                llm_prompt_parts.append(f"Cierre: {last_period_label} registró una variación de {_format_percentage(last_var)}.")
        
        llm_prompt_parts.append("")
        if is_contribution:
            llm_prompt_parts.append(f"TAREA: Redacta una respuesta (máximo 2 oraciones) que MENCIONE el período ({display_period_label}) y cuánta fue la contribución del cierre (solo el porcentaje, 1 decimal). No menciones el valor del índice.")
            llm_prompt_parts.append("Termina con una frase que introduzca la tabla (ej: 'La evolución fue:', 'Los datos mes a mes:' o 'El comportamiento fue:'). Factual y neutral.")
        else:
            llm_prompt_parts.append(f"TAREA: Redacta una respuesta (máximo 2 oraciones) que MENCIONE el período ({display_period_label}) y la variación del cierre.")
            llm_prompt_parts.append("Termina con una frase que introduzca la tabla (ej: 'La evolución fue:', 'Los datos mes a mes:' o 'El comportamiento fue:'). Factual y neutral.")
        llm_prompt = "\n".join(llm_prompt_parts)
    else:
        llm_prompt_parts = []
        if is_contribution:
            row = obs_to_show[0]
            seasonality_lower = (seasonality_context_val or "").lower()
            prefer_yoy = seasonality_lower == "nsa"
            contrib_label = "contribución interanual" if prefer_yoy else "contribución vs período anterior"
            contrib_value = row.get("value")
            
            # Determinar si debemos mencionar la actividad específica (con all_series_data)
            if is_specific_activity and all_series_data:
                # Caso especial: tenemos todas las series de contribución
                # Buscar el valor del IMACEC total
                imacec_total_value = None
                for s in all_series_data:
                    if s.get("activity", "").lower() == "total":
                        imacec_total_value = s.get("value")
                        break
                
                # Nombre de la actividad ganadora (normalizar a lenguaje natural)
                activity_name_raw = component_context_val.strip().replace("_", " ")
                
                # Mapear actividades a nombres más naturales para la redacción
                activity_name_mapping = {
                    "no minero": "no minera",
                    "minero": "minera",
                    "bienes": "producción de bienes",
                    "industria": "industria manufacturera",
                    "resto bienes": "resto de bienes",
                    "comercio": "comercio",
                    "servicios": "servicios",
                    "impuestos sobre los productos": "impuestos sobre los productos",
                }
                activity_name = activity_name_mapping.get(activity_name_raw.lower(), activity_name_raw)
                
                # Construir prompt para frase natural siguiendo el patrón
                llm_prompt_parts.append(f"CONTEXTO:")
                llm_prompt_parts.append(f"- Indicador: {indicator_context_val.upper()}")
                llm_prompt_parts.append(f"- Período: {display_period_label}")
                llm_prompt_parts.append(f"- Variación total: {_format_percentage(imacec_total_value) if imacec_total_value else 'N/A'}")
                llm_prompt_parts.append(f"- Actividad con mayor contribución: {activity_name}")
                llm_prompt_parts.append(f"- Valor de contribución: {_format_percentage(contrib_value)}")
                llm_prompt_parts.append("")
                llm_prompt_parts.append("TAREA:")
                llm_prompt_parts.append("Redacta 2 oraciones siguiendo esta estructura (sin usar comillas):")
                llm_prompt_parts.append("")
                llm_prompt_parts.append("Primera oración:")
                llm_prompt_parts.append(f"- Inicia con: De acuerdo con la BDE, el {indicator_context_val.upper()} de [período] creció [variación] en comparación con igual mes del año anterior (ver tabla).")
                llm_prompt_parts.append("")
                llm_prompt_parts.append("Segunda oración:")
                llm_prompt_parts.append(f"- Explica: El resultado del {indicator_context_val.upper()} se explicó por el crecimiento de la actividad [nombre actividad] con una contribución de [valor].")
                llm_prompt_parts.append("")
                llm_prompt_parts.append("IMPORTANTE:")
                llm_prompt_parts.append("- NO uses comillas en el texto")
                llm_prompt_parts.append("- Redacta de forma natural y fluida")
                llm_prompt_parts.append(f"- Para el período usa: {display_period_label}")
                llm_prompt_parts.append(f"- Para la variación usa: {_format_percentage(imacec_total_value) if imacec_total_value else 'N/A'}")
                llm_prompt_parts.append(f"- Para la actividad usa una forma natural de: {activity_name}")
                llm_prompt_parts.append(f"- Para la contribución usa: {_format_percentage(contrib_value)}")
                llm_prompt_parts.append("- Si el valor es negativo, cambia 'creció' por 'decreció' o 'cayó'")
                llm_prompt_parts.append("- Ajusta el género de la actividad (ej: 'la actividad minera', 'el comercio', 'los servicios')")
            elif is_specific_activity:
                # Tenemos una actividad específica pero sin all_series_data
                activity_name = component_context_val.strip().replace("_", " ")
                llm_prompt_parts.append(f"El usuario preguntó cuál actividad contribuyó más al crecimiento del {indicator_context_val.upper()} en {display_period_label}.")
                llm_prompt_parts.append(f"La actividad con mayor contribución fue: {activity_name}, con {_format_percentage(contrib_value)} (1 decimal).")
                llm_prompt_parts.append("")
                llm_prompt_parts.append(f"TAREA: Responde indicando que {activity_name} fue la actividad que más contribuyó al crecimiento del {indicator_context_val.upper()} en {display_period_label}, con una contribución de {_format_percentage(contrib_value)}. Máximo 2 oraciones. Neutral y factual.")
            else:
                # No hay actividad específica: respuesta genérica
                llm_prompt_parts.append(f"El usuario preguntó por la contribución de {final_indicator_name} en {display_period_label}.")
                llm_prompt_parts.append(f"Cierre: {contrib_label} de {_format_percentage(contrib_value)} (1 decimal).")
                llm_prompt_parts.append("")
                llm_prompt_parts.append("TAREA: Reporta solo la contribución (porcentaje, 1 decimal). No menciones el valor absoluto del índice. Máximo 2 oraciones. Neutral y factual.")
        else:
            row = obs_to_show[0]
            var_value = row.get("yoy") if "yoy" in row else row.get("prev_period")
            var_label = "Variación anual" if "yoy" in row else "Variación período anterior"
            
            llm_prompt_parts.append("SITUACIÓN: El usuario preguntó por un dato económico específico.")
            llm_prompt_parts.append(f"Reporta solo el hecho (máximo 2 oraciones) informando:")
            llm_prompt_parts.append(f"- Indicador: {final_indicator_name}")
            llm_prompt_parts.append(f"- Período: {display_period_label}")
            llm_prompt_parts.append(f"- {var_label}: {_format_percentage(var_value)}")
            llm_prompt_parts.append(f"IMPORTANTE: {_format_percentage(var_value)} es la VARIACIÓN PORCENTUAL, NO menciones el valor absoluto del índice")

            llm_prompt_parts.append("\nREQUISITOS DE ESTILO:")
            llm_prompt_parts.append("- Usa un tono neutral y factual")
            llm_prompt_parts.append("- NO des opiniones, análisis, interpretaciones ni juicios sobre las cifras")
            llm_prompt_parts.append("- NO uses adjetivos que sugieran valoración (bueno, malo, preocupante, alentador, etc.)")
            llm_prompt_parts.append("- Sé conciso: máximo 2 oraciones")
        llm_prompt = "\n".join(llm_prompt_parts)
    
    # Generar respuesta con el LLM
    try:
        llm = build_llm(streaming=True, temperature=0.7, mode="fallback")
        for chunk in llm.stream(llm_prompt, history=[], intent_info=None):
            yield chunk
        yield "\n\n"
    except Exception as e:
        logger.warning(f"Error generando con LLM: {e}")
        if req_form == "range":
            yield f"{final_indicator_name} ({date_range_label}): {len(obs_to_show)} observaciones"
        else:
            yield f"{final_indicator_name} en {display_period_label}: {_format_value(obs_to_show[0].get('value'))}"
        yield "\n\n"
    
    # Tabla markdown - mostrar todas las filas para range, una para latest
    if is_contribution and all_series_data:
        # Construir tabla completa con todas las series de contribución
        yield f"Actividad | Contribución (a/a)\n"
        yield f"----------|-------------------\n"
        
        # Mapear nombres de actividades a nombres legibles y definir orden
        activity_display_names = {
            "total": "IMACEC",
            "bienes": "Producción de bienes",
            "minero": "  Minería",
            "industria": "  Industria manufacturera",
            "resto_bienes": "  Resto de bienes",
            "comercio": "Comercio",
            "servicios": "Servicios",
            "impuestos sobre los productos": "Impuestos sobre los productos",
            "no_minero": "IMACEC No Minero",
        }
        
        # Ordenar según el orden deseado (como en la imagen)
        activity_order = [
            "total", "bienes", "minero", "industria", "resto_bienes",
            "comercio", "servicios", "impuestos sobre los productos", "no_minero"
        ]
        
        # Crear un diccionario para acceso rápido por actividad
        series_by_activity = {s["activity"]: s for s in all_series_data}
        
        # Encontrar la actividad con el mayor valor (excluyendo "total")
        max_activity = None
        max_value = float("-inf")
        for s in all_series_data:
            activity = s.get("activity", "")
            value = s.get("value", 0)
            if activity and activity.lower() != "total" and value > max_value:
                max_value = value
                max_activity = activity
        
        # Mostrar en el orden correcto
        for activity_key in activity_order:
            if activity_key in series_by_activity:
                series_info = series_by_activity[activity_key]
                display_name = activity_display_names.get(activity_key, activity_key)
                value = series_info.get("value", 0)
                
                # Marcar en negrita si es la actividad con mayor contribución
                if activity_key == max_activity:
                    yield f"**{display_name}** | **{_format_percentage(value)}**\n"
                else:
                    yield f"{display_name} | {_format_percentage(value)}\n"
        
        # Agregar nota en itálica después de la tabla
        yield "\n"
        yield "_Tasa de variación porcentual_\n"
        
    elif is_contribution:
        # Tabla simple para contribución (cuando no hay all_series_data)
        yield f"Periodo | Contribución\n"
        yield f"--------|---------------\n"
        for row in obs_to_show:
            date_str = row.get("date", "")
            var_value = row.get("value")
            period_label = _format_period_labels(date_str, freq)[0]
            yield f"{period_label} | {_format_percentage(var_value)}\n"
    else:
        yield f"Periodo | Valor | Variación\n"
        yield f"--------|-------|----------\n"
        for row in obs_to_show:
            date_str = row.get("date", "")
            value = row.get("value")
            var_value = row.get("yoy") if "yoy" in row else row.get("prev_period")
            period_label = _format_period_labels(date_str, freq)[0]
            yield f"{period_label} | {_format_value(value)} | {_format_percentage(var_value)}\n"
    yield "\n"
    
    # Fuente y nota
    if is_contribution and all_series_data:
        # Para tabla de contribuciones, no mostrar nota de unidades
        bde_url = source_url if source_url else "https://si3.bcentral.cl/siete"
        yield f"**Fuente:** 🔗 [Base de Datos Estadísticos (BDE)]({bde_url}) del Banco Central de Chile."
    else:
        # Fuente con nota de unidades
        if indicator_context_val == "imacec":
            yield r"\* _Índice_" + "\n\n"
        else:
            yield r"\* _Miles de millones de pesos_" + "\n\n"
        
        bde_url = source_url if source_url else "https://si3.bcentral.cl/siete"
        yield f"**Fuente:** 🔗 [Base de Datos Estadísticos (BDE)]({bde_url}) del Banco Central de Chile."
    
    # CSV download marker solo si no es range y no es tabla de contribuciones completa
    if req_form != "range" and obs_to_show and not (is_contribution and all_series_data):
        first_row = obs_to_show[0]
        var_value = first_row.get("yoy") if "yoy" in first_row else first_row.get("prev_period")
        var_label = "Variación anual" if "yoy" in first_row else "Variación período anterior"
        var_key = "yoy" if "yoy" in first_row else "prev_period"
        yield from _generate_csv_marker(first_row, series_id, var_value, var_label, var_key)