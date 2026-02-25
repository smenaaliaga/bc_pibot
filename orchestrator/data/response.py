from __future__ import annotations

# === Tipos de respuesta y estructura ===
#
# response_specific
#   - introduction: texto generado (LLM) con contexto y variación
#   - table: tabla con periodos/valores/variaciones
#   - suggestions: sugerencias (placeholder, personalizable)
#   - references: fuentes (1..n cuadros)
#   - charts: gráficos (placeholder, reutilizable)
#   - attachments: CSV download marker
#
# response_general
#   - introduction: texto genérico + lista de cuadros
#   - table: vacío
#   - suggestions: vacío
#   - references: fuente BDE
#   - charts: vacío
#   - attachments: vacío

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from orchestrator.llm.llm_adapter import build_llm

logger = logging.getLogger(__name__)


# === Composición de secciones ===


@dataclass(frozen=True)
class ResponseSections:
    introduction: Iterable[str]
    metadata: Iterable[str]
    table: Iterable[str]
    suggestions: Iterable[str]
    references: Iterable[str]
    charts: Iterable[str]
    attachments: Iterable[str]


def compose_response(sections: ResponseSections) -> Iterable[str]:
    for part in (
        sections.introduction,
        sections.metadata,
        sections.table,
        sections.suggestions,
        sections.references,
        sections.charts,
        sections.attachments,
    ):
        for chunk in part:
            yield chunk


# === Utilidades comunes ===


def normalize_sources(source_url: Any) -> List[str]:
    if isinstance(source_url, dict):
        return [str(v) for v in source_url.values() if v and str(v).lower() != "none"]
    if isinstance(source_url, list):
        return [str(v) for v in source_url if v and str(v).lower() != "none"]
    if isinstance(source_url, str) and source_url.strip() and source_url.lower() != "none":
        return [source_url.strip()]
    return []


def format_period_labels(date_str: Optional[str], freq: str) -> list[str]:
    if not date_str:
        return ["--", "--"]
    try:
        parts = date_str.split("-")
        if len(parts) == 3:
            if int(parts[0]) > 31:
                y = int(parts[0])
                m = int(parts[1])
            else:
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
        meses_es = [
            "", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
            "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
        ]
        meses_abrev = [
            "", "Ene", "Feb", "Mar", "Abr", "May", "Jun",
            "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
        ]
        mes_nombre = meses_es[m] if 1 <= m <= 12 else str(m)
        mes_abrev = meses_abrev[m] if 1 <= m <= 12 else str(m)
        return [f"{mes_nombre} {y}", f"{mes_abrev} {y}"]
    except Exception:
        return [date_str or "--", date_str or "--"]


def format_value(value: Any) -> str:
    try:
        return f"{float(value):,.0f}".replace(",", "_").replace("_", ".")
    except Exception:
        return "--"


def format_percentage(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "--"


def generate_csv_marker(
    row: Dict[str, Any],
    series_id: str,
    var_value: Optional[float],
    var_label: str,
    var_key: str,
) -> Iterable[str]:
    try:
        import pandas as _pd
        import tempfile

        export_map = {
            "date": "Periodo",
            "value": "Valor",
            var_key: var_label,
        }
        export_row = {
            export_map[c]: row.get(c) if c != var_key else var_value
            for c in export_map if c in row or c == var_key
        }
        df_export = _pd.DataFrame([export_row])
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
            prefix="serie_",
            mode="w",
            encoding="utf-8",
        ) as tmp:
            df_export.to_csv(tmp, index=False)
            tmp_path = tmp.name

        filename = f"serie_{series_id}.csv"
        yield "##CSV_DOWNLOAD_START\n"
        yield f"path={tmp_path}\n"
        yield f"filename={filename}\n"
        yield "label=Descargar CSV\n"
        yield "mimetype=text/csv\n"
        yield "##CSV_DOWNLOAD_END\n"
    except Exception as exc:
        logger.warning("No se pudo generar CSV para descarga: %s", exc)


# === Tipos de respuesta ===


def general_response(source_urls: List[str], *, series_id: Optional[str] = None) -> Iterable[str]:
    sections = ResponseSections(
        introduction=_general_intro(source_urls),
        metadata=_metadata_block(series_id),
        table=(),
        suggestions=_general_suggestions(),
        references=_general_references(source_urls),
        charts=(),
        attachments=(),
    )
    return compose_response(sections)


def specific_response(
    *,
    series_id: str,
    series_title: Optional[str] = None,
    req_form: str,
    obs_to_show: List[Dict[str, Any]],
    parsed_point: Optional[str],
    parsed_range: Optional[Tuple[str, str]],
    final_indicator_name: str,
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    metric_type_val: Optional[str],
    intent_cls: Optional[str],
    display_period_label: str,
    freq: str,
    date_range_label: str,
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    source_urls: List[str],
) -> Iterable[str]:
    sections = ResponseSections(
        introduction=_specific_intro(
            req_form=req_form,
            obs_to_show=obs_to_show,
            parsed_point=parsed_point,
            parsed_range=parsed_range,
            final_indicator_name=final_indicator_name,
            indicator_context_val=indicator_context_val,
            component_context_val=component_context_val,
            seasonality_context_val=seasonality_context_val,
            metric_type_val=metric_type_val,
            freq=freq,
            display_period_label=display_period_label,
            date_range_label=date_range_label,
            is_contribution=is_contribution,
            is_specific_activity=is_specific_activity,
            all_series_data=all_series_data,
        ),
        metadata=_metadata_block(series_id, series_title),
        table=_specific_table(
            req_form=req_form,
            obs_to_show=obs_to_show,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
            freq=freq,
        ),
        suggestions=_specific_suggestions(
            indicator_context_val=indicator_context_val,
            component_context_val=component_context_val,
            seasonality_context_val=seasonality_context_val,
            final_indicator_name=final_indicator_name,
            metric_type_val=metric_type_val,
            intent_cls=intent_cls,
            req_form=req_form,
            display_period_label=display_period_label,
        ),
        references=_specific_references(
            indicator_context_val=indicator_context_val,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
            source_urls=source_urls,
        ),
        charts=_default_charts(),
        attachments=_specific_attachments(
            series_id=series_id,
            req_form=req_form,
            obs_to_show=obs_to_show,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
        ),
    )
    return compose_response(sections)


def specific_point_response(
    *,
    series_id: str,
    series_title: Optional[str] = None,
    req_form: str,
    obs_to_show: List[Dict[str, Any]],
    parsed_point: Optional[str],
    parsed_range: Optional[Tuple[str, str]],
    final_indicator_name: str,
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    metric_type_val: Optional[str],
    intent_cls: Optional[str],
    display_period_label: str,
    freq: str,
    date_range_label: str,
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    source_urls: List[str],
) -> Iterable[str]:
    sections = ResponseSections(
        introduction=_specific_intro(
            req_form="specific_point",
            obs_to_show=obs_to_show,
            parsed_point=parsed_point,
            parsed_range=parsed_range,
            final_indicator_name=final_indicator_name,
            indicator_context_val=indicator_context_val,
            component_context_val=component_context_val,
            seasonality_context_val=seasonality_context_val,
            metric_type_val=metric_type_val,
            freq=freq,
            display_period_label=display_period_label,
            date_range_label=date_range_label,
            is_contribution=is_contribution,
            is_specific_activity=is_specific_activity,
            all_series_data=all_series_data,
        ),
        metadata=_metadata_block(series_id, series_title),
        table=_specific_table(
            req_form="specific_point",
            obs_to_show=obs_to_show,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
            freq=freq,
        ),
        suggestions=_specific_suggestions(
            indicator_context_val=indicator_context_val,
            component_context_val=component_context_val,
            seasonality_context_val=seasonality_context_val,
            final_indicator_name=final_indicator_name,
            metric_type_val=metric_type_val,
            intent_cls=intent_cls,
            req_form="specific_point",
            display_period_label=display_period_label,
        ),
        references=_specific_references(
            indicator_context_val=indicator_context_val,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
            source_urls=source_urls,
        ),
        charts=_default_charts(),
        attachments=_specific_attachments(
            series_id=series_id,
            req_form="specific_point",
            obs_to_show=obs_to_show,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
        ),
    )
    return compose_response(sections)


# === Secciones: response_general ===


def _general_intro(source_urls: List[str]) -> Iterable[str]:
    yield "Para esta consulta, los datos están disponibles en el siguiente cuadro de la BDE:\n\n"
    if source_urls:
        for idx, url in enumerate(source_urls, start=1):
            yield f"- Cuadro {idx}: {url}\n"
    else:
        yield "- No hay cuadros disponibles para esta combinación de filtros.\n"


def _metadata_block(series_id: Optional[str], series_title: Optional[str] = None) -> Iterable[str]:
    series_code = str(series_id or "").strip()
    if not series_code or series_code.lower() == "none":
        return ()
    description = str(series_title or "").strip()
    if description and description.lower() != "none":
        return (
            f"**Código de serie:** {series_code}\n",
            f"**Descripción de la serie:** {description}\n\n",
        )
    return (f"**Código de serie:** {series_code}\n\n",)


def _general_references(source_urls: List[str]) -> Iterable[str]:
    bde_urls = source_urls or ["https://si3.bcentral.cl/siete"]
    if len(bde_urls) == 1:
        yield f"\n**Fuente:** 🔗 [Base de Datos Estadísticos (BDE)]({bde_urls[0]}) del Banco Central de Chile."
    else:
        yield "\n**Fuentes:**\n"
        for idx, url in enumerate(bde_urls, start=1):
            yield f"- 🔗 [Cuadro {idx}]({url})\n"
        yield "\nBanco Central de Chile."


# === Secciones: response_specific ===


def _specific_intro(
    *,
    req_form: str,
    obs_to_show: List[Dict[str, Any]],
    parsed_point: Optional[str],
    parsed_range: Optional[Tuple[str, str]],
    final_indicator_name: str,
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    metric_type_val: Optional[str],
    freq: str,
    display_period_label: str,
    date_range_label: str,
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
) -> Iterable[str]:
    llm_prompt = _build_specific_prompt(
        req_form=req_form,
        obs_to_show=obs_to_show,
        parsed_point=parsed_point,
        parsed_range=parsed_range,
        final_indicator_name=final_indicator_name,
        indicator_context_val=indicator_context_val,
        component_context_val=component_context_val,
        seasonality_context_val=seasonality_context_val,
        metric_type_val=metric_type_val,
        freq=freq,
        display_period_label=display_period_label,
        date_range_label=date_range_label,
        is_contribution=is_contribution,
        is_specific_activity=is_specific_activity,
        all_series_data=all_series_data,
    )
    yield from _stream_llm_or_fallback(
        llm_prompt=llm_prompt,
        req_form=req_form,
        final_indicator_name=final_indicator_name,
        date_range_label=date_range_label,
        display_period_label=display_period_label,
        obs_to_show=obs_to_show,
    )


def _build_specific_prompt(
    *,
    req_form: str,
    obs_to_show: List[Dict[str, Any]],
    parsed_point: Optional[str],
    parsed_range: Optional[Tuple[str, str]],
    final_indicator_name: str,
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    metric_type_val: Optional[str],
    freq: str,
    display_period_label: str,
    date_range_label: str,
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
) -> str:
    if req_form in {"range", "specific_point"}:
        return _build_range_prompt(
            obs_to_show=obs_to_show,
            final_indicator_name=final_indicator_name,
            seasonality_context_val=seasonality_context_val,
            freq=freq,
            display_period_label=display_period_label,
            is_contribution=is_contribution,
        )
    return _build_latest_prompt(
        obs_to_show=obs_to_show,
        final_indicator_name=final_indicator_name,
        indicator_context_val=indicator_context_val,
        component_context_val=component_context_val,
        seasonality_context_val=seasonality_context_val,
        freq=freq,
        display_period_label=display_period_label,
        is_contribution=is_contribution,
        is_specific_activity=is_specific_activity,
        all_series_data=all_series_data,
    )


def _build_range_prompt(
    *,
    obs_to_show: List[Dict[str, Any]],
    final_indicator_name: str,
    seasonality_context_val: Optional[str],
    freq: str,
    display_period_label: str,
    is_contribution: bool,
) -> str:
    llm_prompt_parts: List[str] = []
    if obs_to_show:
        last_row = obs_to_show[-1]
        last_date_str = last_row.get("date", "")
        seasonality_lower = (seasonality_context_val or "").lower()
        prefer_yoy = seasonality_lower == "nsa"

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
        last_period_label = format_period_labels(last_date_str, freq)[0]

        if is_contribution:
            llm_prompt_parts.append(
                f"El usuario preguntó por la contribución de {final_indicator_name} en el período: {display_period_label}."
            )
            llm_prompt_parts.append(
                f"Cierre: {last_period_label} registró {contrib_label} de {format_percentage(last_var)} (1 decimal)."
            )
        else:
            llm_prompt_parts.append(
                f"El usuario preguntó por {final_indicator_name} en el período: {display_period_label}."
            )
            llm_prompt_parts.append(
                f"Cierre: {last_period_label} registró una variación de {format_percentage(last_var)}."
            )

    llm_prompt_parts.append("")
    if is_contribution:
        llm_prompt_parts.append(
            f"TAREA: Redacta una respuesta (máximo 2 oraciones) que MENCIONE el período ({display_period_label}) y cuánta fue la contribución del cierre (solo el porcentaje, 1 decimal). No menciones el valor del índice."
        )
        llm_prompt_parts.append(
            "Termina con una frase que introduzca la tabla (ej: 'La evolución fue:', 'Los datos mes a mes:' o 'El comportamiento fue:'). Factual y neutral."
        )
    else:
        llm_prompt_parts.append(
            f"TAREA: Redacta una respuesta (máximo 2 oraciones) que MENCIONE el período ({display_period_label}) y la variación del cierre."
        )
        llm_prompt_parts.append(
            "Termina con una frase que introduzca la tabla (ej: 'La evolución fue:', 'Los datos mes a mes:' o 'El comportamiento fue:'). Factual y neutral."
        )
    return "\n".join(llm_prompt_parts)


def _build_latest_prompt(
    *,
    obs_to_show: List[Dict[str, Any]],
    final_indicator_name: str,
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    freq: str,
    display_period_label: str,
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
) -> str:
    def _variation_label(var_key: Optional[str], freq_value: str) -> str:
        freq_norm = str(freq_value or "").strip().lower()
        if var_key == "yoy":
            if freq_norm == "q":
                return "variación interanual trimestral (a/a)"
            if freq_norm == "m":
                return "variación interanual mensual (a/a)"
            return "variación interanual (a/a)"
        if var_key == "prev_period":
            if freq_norm == "q":
                return "variación trimestral respecto al período anterior (t/t)"
            if freq_norm == "m":
                return "variación mensual respecto al período anterior (m/m)"
            if freq_norm == "a":
                return "variación anual respecto al período anterior"
            return "variación respecto al período anterior"
        return "variación"

    llm_prompt_parts: List[str] = []
    if is_contribution:
        row = obs_to_show[0]
        seasonality_lower = (seasonality_context_val or "").lower()
        prefer_yoy = seasonality_lower == "nsa"
        contrib_label = "contribución interanual" if prefer_yoy else "contribución vs período anterior"
        contrib_value = row.get("value")

        if is_specific_activity and all_series_data:
            imacec_total_value = None
            for s in all_series_data:
                if s.get("activity", "").lower() == "total":
                    imacec_total_value = s.get("value")
                    break

            activity_name_raw = component_context_val.strip().replace("_", " ") if component_context_val else ""
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

            llm_prompt_parts.append("CONTEXTO:")
            llm_prompt_parts.append(f"- Indicador: {indicator_context_val.upper()}")
            llm_prompt_parts.append(f"- Período: {display_period_label}")
            llm_prompt_parts.append(
                f"- Variación total: {format_percentage(imacec_total_value) if imacec_total_value else 'N/A'}"
            )
            llm_prompt_parts.append(f"- Actividad con mayor contribución: {activity_name}")
            llm_prompt_parts.append(f"- Valor de contribución: {format_percentage(contrib_value)}")
            llm_prompt_parts.append("")
            llm_prompt_parts.append("TAREA:")
            llm_prompt_parts.append("Redacta 2 oraciones siguiendo esta estructura (sin usar comillas):")
            llm_prompt_parts.append("")
            llm_prompt_parts.append("Primera oración:")
            llm_prompt_parts.append(
                f"- Inicia con: De acuerdo con la BDE, el {indicator_context_val.upper()} de [período] creció [variación] en comparación con igual mes del año anterior (ver tabla)."
            )
            llm_prompt_parts.append("")
            llm_prompt_parts.append("Segunda oración:")
            llm_prompt_parts.append(
                f"- Explica: El resultado del {indicator_context_val.upper()} se explicó por el crecimiento de la actividad [nombre actividad] con una contribución de [valor]."
            )
            llm_prompt_parts.append("")
            llm_prompt_parts.append("IMPORTANTE:")
            llm_prompt_parts.append("- NO uses comillas en el texto")
            llm_prompt_parts.append("- Redacta de forma natural y fluida")
            llm_prompt_parts.append(f"- Para el período usa: {display_period_label}")
            llm_prompt_parts.append(
                f"- Para la variación usa: {format_percentage(imacec_total_value) if imacec_total_value else 'N/A'}"
            )
            llm_prompt_parts.append(f"- Para la actividad usa una forma natural de: {activity_name}")
            llm_prompt_parts.append(f"- Para la contribución usa: {format_percentage(contrib_value)}")
            llm_prompt_parts.append("- Si el valor es negativo, cambia 'creció' por 'decreció' o 'cayó'")
            llm_prompt_parts.append("- Ajusta el género de la actividad (ej: 'la actividad minera', 'el comercio', 'los servicios')")
        elif is_specific_activity:
            activity_name = component_context_val.strip().replace("_", " ") if component_context_val else ""
            llm_prompt_parts.append(
                f"El usuario preguntó cuál actividad contribuyó más al crecimiento del {indicator_context_val.upper()} en {display_period_label}."
            )
            llm_prompt_parts.append(
                f"La actividad con mayor contribución fue: {activity_name}, con {format_percentage(contrib_value)} (1 decimal)."
            )
            llm_prompt_parts.append("")
            llm_prompt_parts.append(
                f"TAREA: Responde indicando que {activity_name} fue la actividad que más contribuyó al crecimiento del {indicator_context_val.upper()} en {display_period_label}, con una contribución de {format_percentage(contrib_value)}. Máximo 2 oraciones. Neutral y factual."
            )
        else:
            llm_prompt_parts.append(
                f"El usuario preguntó por la contribución de {final_indicator_name} en {display_period_label}."
            )
            llm_prompt_parts.append(
                f"Cierre: {contrib_label} de {format_percentage(contrib_value)} (1 decimal)."
            )
            llm_prompt_parts.append("")
            llm_prompt_parts.append(
                "TAREA: Reporta solo la contribución (porcentaje, 1 decimal). No menciones el valor absoluto del índice. Máximo 2 oraciones. Neutral y factual."
            )
    else:
        row = obs_to_show[0]
        if "yoy" in row:
            var_key = "yoy"
            var_value = row.get("yoy")
        elif "prev_period" in row:
            var_key = "prev_period"
            var_value = row.get("prev_period")
        else:
            var_key = None
            var_value = None
        freq_raw = str(freq or row.get("frequency") or row.get("freq") or "").strip().lower()
        var_label = _variation_label(var_key, freq_raw)

        llm_prompt_parts.append("SITUACIÓN: El usuario preguntó por un dato económico específico.")
        llm_prompt_parts.append("Reporta solo la variación (máximo 2 oraciones) informando:")
        llm_prompt_parts.append(f"- Indicador: {final_indicator_name}")
        llm_prompt_parts.append(f"- Período: {display_period_label}")
        if var_value is not None:
            llm_prompt_parts.append(f"- {var_label}: {format_percentage(var_value)}")
            llm_prompt_parts.append("IMPORTANTE: NO menciones el valor absoluto de la serie; reporta solo la variación")
            llm_prompt_parts.append("IMPORTANTE: redacta en forma directa, por ejemplo: 'La variación ... fue de X%.'")
            llm_prompt_parts.append("IMPORTANTE: NO agregues frases meta como 'no se proporciona el valor absoluto'")
        else:
            llm_prompt_parts.append("IMPORTANTE: si no hay variación disponible, indícalo explícitamente sin inventar cifras")

        llm_prompt_parts.append("\nREQUISITOS DE ESTILO:")
        llm_prompt_parts.append("- Usa un tono neutral y factual")
        llm_prompt_parts.append("- NO des opiniones, análisis, interpretaciones ni juicios sobre las cifras")
        llm_prompt_parts.append("- NO uses adjetivos que sugieran valoración (bueno, malo, preocupante, alentador, etc.)")
        llm_prompt_parts.append("- Sé conciso: máximo 2 oraciones")
    return "\n".join(llm_prompt_parts)


def _stream_llm_or_fallback(
    *,
    llm_prompt: str,
    req_form: str,
    final_indicator_name: str,
    date_range_label: str,
    display_period_label: str,
    obs_to_show: List[Dict[str, Any]],
) -> Iterable[str]:
    try:
        llm = build_llm(streaming=True, temperature=0.7, mode="fallback")
        chunks: List[str] = []
        for chunk in llm.stream(llm_prompt, history=[], intent_info=None):
            text = str(chunk)
            if text:
                chunks.append(text)

        generated_text = "".join(chunks).strip()
        if not generated_text or generated_text.lower().startswith("(error generando)"):
            raise RuntimeError("llm_generation_failed")

        for chunk in chunks:
            yield chunk
        yield "\n\n"


    # === Secciones específicas: tabla, referencias, sugerencias, adjuntos ===
    except Exception as exc:
        logger.warning("Error generando con LLM: %s", exc)
        if req_form in {"range", "specific_point"}:
            yield f"{final_indicator_name} ({date_range_label}): {len(obs_to_show)} observaciones"
        else:
            row = obs_to_show[0] if obs_to_show else {}
            if "yoy" in row and row.get("yoy") is not None:
                var_label = "variación interanual"
                var_value = row.get("yoy")
            elif "prev_period" in row and row.get("prev_period") is not None:
                var_label = "variación respecto al período anterior"
                var_value = row.get("prev_period")
            else:
                var_label = "variación"
                var_value = None

            if var_value is None:
                yield f"{final_indicator_name} en {display_period_label}: no hay variación disponible"
            else:
                yield f"{final_indicator_name} en {display_period_label}: {var_label} de {format_percentage(var_value)}"
        yield "\n\n"


def _specific_table(
    *,
    req_form: str,
    obs_to_show: List[Dict[str, Any]],
    is_contribution: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    freq: str,
) -> Iterable[str]:
    if is_contribution and all_series_data:
        yield "Actividad | Contribución (a/a)\n"
        yield "----------|-------------------\n"

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

        activity_order = [
            "total", "bienes", "minero", "industria", "resto_bienes",
            "comercio", "servicios", "impuestos sobre los productos", "no_minero",
        ]

        series_by_activity = {s["activity"]: s for s in all_series_data}

        max_activity = None
        max_value = float("-inf")
        for s in all_series_data:
            activity = s.get("activity", "")
            value = s.get("value", 0)
            if activity and activity.lower() != "total" and value > max_value:
                max_value = value
                max_activity = activity

        for activity_key in activity_order:
            if activity_key in series_by_activity:
                series_info = series_by_activity[activity_key]
                display_name = activity_display_names.get(activity_key, activity_key)
                value = series_info.get("value", 0)

                if activity_key == max_activity:
                    yield f"**{display_name}** | **{format_percentage(value)}**\n"
                else:
                    yield f"{display_name} | {format_percentage(value)}\n"

        yield "\n"
        yield "_Tasa de variación porcentual_\n"

    elif is_contribution:
        yield "Periodo | Contribución\n"
        yield "--------|---------------\n"
        for row in obs_to_show:
            date_str = row.get("date", "")
            var_value = row.get("value")
            period_label = format_period_labels(date_str, freq)[0]
            yield f"{period_label} | {format_percentage(var_value)}\n"
    else:
        yield "Periodo | Valor | Variación\n"
        yield "--------|-------|----------\n"
        for row in obs_to_show:
            date_str = row.get("date", "")
            value = row.get("value")
            var_value = row.get("yoy") if "yoy" in row else row.get("prev_period")
            period_label = format_period_labels(date_str, freq)[0]
            yield f"{period_label} | {format_value(value)} | {format_percentage(var_value)}\n"
    yield "\n"


def _specific_references(
    *,
    indicator_context_val: Optional[str],
    is_contribution: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    source_urls: List[str],
) -> Iterable[str]:
    bde_urls = source_urls or ["https://si3.bcentral.cl/siete"]
    if is_contribution and all_series_data:
        if len(bde_urls) == 1:
            yield f"**Fuente:** 🔗 [Base de Datos Estadísticos (BDE)]({bde_urls[0]}) del Banco Central de Chile."
        else:
            yield "**Fuentes:**\n"
            for idx, url in enumerate(bde_urls, start=1):
                yield f"- 🔗 [Cuadro {idx}]({url})\n"
            yield "\nBanco Central de Chile."
    else:
        if indicator_context_val == "imacec":
            yield r"\* _Índice_" + "\n\n"
        else:
            yield r"\* _Miles de millones de pesos_" + "\n\n"

        if len(bde_urls) == 1:
            yield f"**Fuente:** 🔗 [Base de Datos Estadísticos (BDE)]({bde_urls[0]}) del Banco Central de Chile."
        else:
            yield "**Fuentes:**\n"
            for idx, url in enumerate(bde_urls, start=1):
                yield f"- 🔗 [Cuadro {idx}]({url})\n"
            yield "\nBanco Central de Chile."


def _general_suggestions() -> Iterable[str]:
    prompt = (
        "Genera 3 preguntas de seguimiento útiles y concretas para un usuario que "
        "consulta indicadores económicos chilenos. Devuelve solo 3 líneas, una por pregunta, "
        "sin numeración ni viñetas."
    )
    return _generate_llm_suggestions(prompt, fallback=_general_suggestions_default)


def _general_suggestions_default() -> Iterable[str]:
    suggestions = [
        "¿Quieres que busque los datos más recientes?",
        "¿Te muestro un gráfico con la última variación?",
        "¿Prefieres consultar IMACEC o PIB?",
    ]
    return _build_followup_block(suggestions)


def _specific_suggestions(
    *,
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    final_indicator_name: str,
    metric_type_val: Optional[str],
    intent_cls: Optional[str],
    req_form: str,
    display_period_label: str,
) -> Iterable[str]:
    indicator = None
    if isinstance(indicator_context_val, str) and indicator_context_val.strip():
        indicator = indicator_context_val.strip()
    elif final_indicator_name:
        indicator = final_indicator_name

    indicator_label = indicator or "indicador"
    seasonality_label = seasonality_context_val or "no especificado"
    component_label = component_context_val or "total"
    metric_label = metric_type_val or "index"

    prompt = (
        "Genera 3 preguntas de seguimiento basadas en el siguiente contexto. "
        "Las preguntas deben ser específicas, breves y coherentes con la respuesta mostrada. "
        "No uses numeración ni viñetas; devuelve solo 3 líneas, una por pregunta.\n\n"
        f"Indicador: {indicator_label}\n"
        f"Componente/actividad: {component_label}\n"
        f"Estacionalidad: {seasonality_label}\n"
        f"Tipo de métrica: {metric_label}\n"
        f"Tipo de consulta: {req_form}\n"
        f"Periodo mostrado: {display_period_label}\n"
        f"Intento: {intent_cls or 'value'}\n"
        "\n"
        "Sugerencias:"
    )
    return _generate_llm_suggestions(prompt, fallback=lambda: _specific_suggestions_default(
        indicator_context_val=indicator_context_val,
        component_context_val=component_context_val,
        seasonality_context_val=seasonality_context_val,
        final_indicator_name=final_indicator_name,
        intent_cls=intent_cls,
    ))


def _specific_suggestions_default(
    *,
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    final_indicator_name: str,
    intent_cls: Optional[str],
) -> Iterable[str]:
    indicator = None
    if isinstance(indicator_context_val, str) and indicator_context_val.strip():
        indicator = indicator_context_val.strip()
    elif final_indicator_name:
        indicator = final_indicator_name

    if not indicator:
        return _general_suggestions_default()

    suggestions: List[str] = []
    seasonality_lower = (seasonality_context_val or "").lower()
    if seasonality_lower:
        if "desestacionalizado" in seasonality_lower or seasonality_lower == "sa":
            suggestions.append(f"¿Cuál es el {indicator} sin desestacionalizar?")
        else:
            suggestions.append(f"¿Cuál es el {indicator} desestacionalizado?")
    else:
        suggestions.append(f"¿Cuál es el {indicator} desestacionalizado?")

    indicator_lower = indicator.lower()
    component_lower = (component_context_val or "").lower()
    if "imacec" in indicator_lower:
        if not component_lower or component_lower == "total":
            suggestions.append("¿Cómo estuvo el IMACEC minero?")
        elif "minero" in component_lower:
            suggestions.append("¿Cómo estuvo el IMACEC no minero?")
        else:
            suggestions.append("¿Cómo estuvo el IMACEC total?")

    if "pib" in indicator_lower and not component_lower:
        suggestions.append("¿Cuál es la variación del PIB por sectores?")

    suggestions.append(f"¿Qué mide el {indicator}?")

    if intent_cls in ("methodology", "definition"):
        suggestions.insert(0, f"¿Cuál es el último valor del {indicator}?")

    unique = _unique_suggestions(suggestions)
    return _build_followup_block(unique)


def _generate_llm_suggestions(prompt: str, fallback) -> Iterable[str]:
    try:
        llm = build_llm(streaming=False, temperature=0.3, mode="fallback")
        raw = llm.generate(prompt, history=[], intent_info=None)
        if not raw or "(error generando)" in raw.lower():
            return fallback()
        parsed = _parse_llm_suggestions(raw)
        if parsed:
            return _build_followup_block(parsed)
    except Exception as exc:
        logger.debug("No se pudo generar sugerencias con LLM: %s", exc)
    return fallback()


def _parse_llm_suggestions(text: str) -> List[str]:
    if not text:
        return []
    lines: List[str] = []
    for raw in text.splitlines():
        cleaned = raw.strip().lstrip("-•0123456789.) ")
        if cleaned:
            lines.append(cleaned)
    return _unique_suggestions(lines)


def _unique_suggestions(items: Iterable[str]) -> List[str]:
    seen = set()
    unique_items: List[str] = []
    for suggestion in items:
        normalized = "".join(ch for ch in suggestion.lower() if ch.isalnum())
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_items.append(suggestion)
    return unique_items[:3]


def _build_followup_block(suggestions: Iterable[str]) -> Iterable[str]:
    items = list(suggestions)
    if not items:
        return []
    lines: List[str] = ["\n\n##FOLLOWUP_START\n"]
    for i, question in enumerate(items[:3], start=1):
        lines.append(f"suggestion_{i}={question}\n")
    lines.append("##FOLLOWUP_END")
    return lines


def _default_charts() -> Iterable[str]:
    return iter(())


def _specific_attachments(
    *,
    series_id: str,
    req_form: str,
    obs_to_show: List[Dict[str, Any]],
    is_contribution: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
) -> Iterable[str]:
    if req_form not in {"range", "specific_point"} and obs_to_show and not (is_contribution and all_series_data):
        first_row = obs_to_show[0]
        var_value = first_row.get("yoy") if "yoy" in first_row else first_row.get("prev_period")
        var_label = "Variación anual" if "yoy" in first_row else "Variación período anterior"
        var_key = "yoy" if "yoy" in first_row else "prev_period"
        yield from generate_csv_marker(first_row, series_id, var_value, var_label, var_key)
