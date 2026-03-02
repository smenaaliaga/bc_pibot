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
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from orchestrator.llm.llm_adapter import build_llm
from rules.business_rule import resolve_response_rule

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


def _normalize_text_field(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "null", "nan"}:
        return ""
    return text


def _extract_year_int(value: Any) -> Optional[int]:
    text = _normalize_text_field(value)
    if not text:
        return None
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
    if not years:
        return None
    try:
        return max(int(year) for year in years)
    except Exception:
        return None


def _has_year_posterior_than_limit(text: str, latest_available_period: Optional[str]) -> bool:
    max_allowed_year = _extract_year_int(latest_available_period)
    if max_allowed_year is None:
        return False
    generated_years = re.findall(r"\b(19\d{2}|20\d{2})\b", str(text or ""))
    if not generated_years:
        return False
    try:
        return any(int(year) > max_allowed_year for year in generated_years)
    except Exception:
        return False


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
        if str(freq or "").upper() == "A":
            return [str(y), str(y)]
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
    response_type = resolve_response_rule(req_form="latest", series_id=series_id)
    if response_type != "response_general":
        logger.debug("[RESPONSE] Regla ajustada a %s para general_response", response_type)

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


def _build_specific_sections(
    *,
    req_form: str,
    series_id: str,
    series_title: Optional[str],
    obs_to_show: List[Dict[str, Any]],
    parsed_point: Optional[str],
    parsed_range: Optional[Tuple[str, str]],
    final_indicator_name: str,
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    metric_type_val: Optional[str],
    calc_mode_cls: Optional[str],
    intent_cls: Optional[str],
    display_period_label: str,
    freq: str,
    date_range_label: str,
    reference_period: Optional[str],
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    source_urls: List[str],
    user_question: Optional[str],
    conversation_context: Optional[str],
    latest_available_period: Optional[str],
    chart_context_series: Optional[List[str]],
    intro_llm_temperature: float,
) -> ResponseSections:
    return ResponseSections(
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
            calc_mode_cls=calc_mode_cls,
            series_title=series_title,
            freq=freq,
            display_period_label=display_period_label,
            date_range_label=date_range_label,
            reference_period=reference_period,
            is_contribution=is_contribution,
            is_specific_activity=is_specific_activity,
            all_series_data=all_series_data,
            user_question=user_question,
            conversation_context=conversation_context,
            latest_available_period=latest_available_period,
            intro_llm_temperature=intro_llm_temperature,
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
        charts=_default_charts(
            series_id=series_id,
            obs_to_show=obs_to_show,
            req_form=req_form,
            indicator_context_val=indicator_context_val,
            chart_context_series=chart_context_series,
        ),
        attachments=_specific_attachments(
            series_id=series_id,
            req_form=req_form,
            obs_to_show=obs_to_show,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
        ),
    )


def _stream_by_response_rule(
    *,
    req_form: str,
    series_id: str,
    series_title: Optional[str],
    obs_to_show: List[Dict[str, Any]],
    parsed_point: Optional[str],
    parsed_range: Optional[Tuple[str, str]],
    final_indicator_name: str,
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    metric_type_val: Optional[str],
    calc_mode_cls: Optional[str],
    intent_cls: Optional[str],
    display_period_label: str,
    freq: str,
    date_range_label: str,
    reference_period: Optional[str],
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    source_urls: List[str],
    user_question: Optional[str],
    conversation_context: Optional[str],
    latest_available_period: Optional[str],
    chart_context_series: Optional[List[str]],
    intro_llm_temperature: float,
) -> Iterable[str]:
    response_type = resolve_response_rule(req_form=req_form, series_id=series_id)

    # ## REGLA 1
    # 1. ##Título: Respuesta general sin serie
    # 2. ##Descripción: Si no hay serie válida, no se construye respuesta específica.
    # 3. ##Condiciones: `response_type == "response_general"`.
    # 4. ##Resultado: Ejecuta `general_response(...)`.
    if response_type == "response_general":
        return general_response(source_urls, series_id=series_id)

    # ## REGLA 3
    # 1. ##Título: Respuesta de punto específico
    # 2. ##Descripción: Usa plantilla dedicada para consultas `specific_point`.
    # 3. ##Condiciones: `response_type == "response_specific_point"`.
    # 4. ##Resultado: Construye secciones con `req_form="specific_point"`.
    if response_type == "response_specific_point":
        sections = _build_specific_sections(
            req_form="specific_point",
            series_id=series_id,
            series_title=series_title,
            obs_to_show=obs_to_show,
            parsed_point=parsed_point,
            parsed_range=parsed_range,
            final_indicator_name=final_indicator_name,
            indicator_context_val=indicator_context_val,
            component_context_val=component_context_val,
            seasonality_context_val=seasonality_context_val,
            metric_type_val=metric_type_val,
            calc_mode_cls=calc_mode_cls,
            intent_cls=intent_cls,
            display_period_label=display_period_label,
            freq=freq,
            date_range_label=date_range_label,
            reference_period=reference_period,
            is_contribution=is_contribution,
            is_specific_activity=is_specific_activity,
            all_series_data=all_series_data,
            source_urls=source_urls,
            user_question=user_question,
            conversation_context=conversation_context,
            latest_available_period=latest_available_period,
            chart_context_series=chart_context_series,
            intro_llm_temperature=intro_llm_temperature,
        )
        return compose_response(sections)

    # ## REGLA 2
    # 1. ##Título: Respuesta específica estándar
    # 2. ##Descripción: Ruta normal para `latest`, `range` y `point`.
    # 3. ##Condiciones: `response_type == "response_specific"` (o fallback con serie válida).
    # 4. ##Resultado: Construye secciones específicas con `req_form` original.
    sections = _build_specific_sections(
        req_form=req_form,
        series_id=series_id,
        series_title=series_title,
        obs_to_show=obs_to_show,
        parsed_point=parsed_point,
        parsed_range=parsed_range,
        final_indicator_name=final_indicator_name,
        indicator_context_val=indicator_context_val,
        component_context_val=component_context_val,
        seasonality_context_val=seasonality_context_val,
        metric_type_val=metric_type_val,
        calc_mode_cls=calc_mode_cls,
        intent_cls=intent_cls,
        display_period_label=display_period_label,
        freq=freq,
        date_range_label=date_range_label,
        reference_period=reference_period,
        is_contribution=is_contribution,
        is_specific_activity=is_specific_activity,
        all_series_data=all_series_data,
        source_urls=source_urls,
        user_question=user_question,
        conversation_context=conversation_context,
        latest_available_period=latest_available_period,
        chart_context_series=chart_context_series,
        intro_llm_temperature=intro_llm_temperature,
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
    calc_mode_cls: Optional[str],
    intent_cls: Optional[str],
    display_period_label: str,
    freq: str,
    date_range_label: str,
    reference_period: Optional[str],
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    source_urls: List[str],
    user_question: Optional[str] = None,
    conversation_context: Optional[str] = None,
    latest_available_period: Optional[str] = None,
    chart_context_series: Optional[List[str]] = None,
    intro_llm_temperature: float = 0.7,
) -> Iterable[str]:
    return _stream_by_response_rule(
        req_form=req_form,
        series_id=series_id,
        series_title=series_title,
        obs_to_show=obs_to_show,
        parsed_point=parsed_point,
        parsed_range=parsed_range,
        final_indicator_name=final_indicator_name,
        indicator_context_val=indicator_context_val,
        component_context_val=component_context_val,
        seasonality_context_val=seasonality_context_val,
        metric_type_val=metric_type_val,
        calc_mode_cls=calc_mode_cls,
        intent_cls=intent_cls,
        display_period_label=display_period_label,
        freq=freq,
        date_range_label=date_range_label,
        reference_period=reference_period,
        is_contribution=is_contribution,
        is_specific_activity=is_specific_activity,
        all_series_data=all_series_data,
        source_urls=source_urls,
        user_question=user_question,
        conversation_context=conversation_context,
        latest_available_period=latest_available_period,
        chart_context_series=chart_context_series,
        intro_llm_temperature=intro_llm_temperature,
    )


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
    calc_mode_cls: Optional[str],
    intent_cls: Optional[str],
    display_period_label: str,
    freq: str,
    date_range_label: str,
    reference_period: Optional[str],
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    source_urls: List[str],
    user_question: Optional[str] = None,
    conversation_context: Optional[str] = None,
    latest_available_period: Optional[str] = None,
    chart_context_series: Optional[List[str]] = None,
    intro_llm_temperature: float = 0.7,
) -> Iterable[str]:
    return _stream_by_response_rule(
        req_form="specific_point",
        series_id=series_id,
        series_title=series_title,
        obs_to_show=obs_to_show,
        parsed_point=parsed_point,
        parsed_range=parsed_range,
        final_indicator_name=final_indicator_name,
        indicator_context_val=indicator_context_val,
        component_context_val=component_context_val,
        seasonality_context_val=seasonality_context_val,
        metric_type_val=metric_type_val,
        calc_mode_cls=calc_mode_cls,
        intent_cls=intent_cls,
        display_period_label=display_period_label,
        freq=freq,
        date_range_label=date_range_label,
        reference_period=reference_period,
        is_contribution=is_contribution,
        is_specific_activity=is_specific_activity,
        all_series_data=all_series_data,
        source_urls=source_urls,
        user_question=user_question,
        conversation_context=conversation_context,
        latest_available_period=latest_available_period,
        chart_context_series=chart_context_series,
        intro_llm_temperature=intro_llm_temperature,
    )


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
            f"**Código de serie:** {series_code}\n\n",
            f"**Descripción de la serie:** {description}\n\n",
        )
    return (f"**Código de serie:** {series_code}\n\n",)


def _general_references(source_urls: List[str]) -> Iterable[str]:
    if source_urls:
        return

    bde_urls = ["https://si3.bcentral.cl/siete"]
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
    calc_mode_cls: Optional[str],
    series_title: Optional[str],
    freq: str,
    display_period_label: str,
    date_range_label: str,
    reference_period: Optional[str],
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    user_question: Optional[str],
    conversation_context: Optional[str],
    latest_available_period: Optional[str],
    intro_llm_temperature: float = 0.7,
) -> Iterable[str]:
    def _clean_text(value: Any) -> str:
        text = str(value or "").strip()
        if text.lower() in {"", "none", "null", "nan"}:
            return ""
        return text

    def _percentage_es(value: Any) -> str:
        formatted = format_percentage(value)
        return formatted.replace(".", ",") if formatted != "--" else formatted

    def _comparison_label(var_key: Optional[str], freq_value: str) -> str:
        freq_norm = str(freq_value or "").strip().lower()
        if var_key == "yoy":
            if freq_norm == "m":
                return "en comparación con el mismo período del año anterior"
            if freq_norm == "q":
                return "en comparación con el mismo trimestre del año anterior"
            if freq_norm == "a":
                return "en comparación con el año anterior"
            return "en comparación con el mismo período del año anterior"
        if var_key == "prev_period":
            return "en comparación con el período anterior"
        return ""

    fallback_intro = None
    if req_form == "latest" and obs_to_show:
        fallback_intro = _build_latest_intro_fallback(
            obs_to_show=obs_to_show,
            freq=freq,
            indicator_context_val=indicator_context_val,
            final_indicator_name=final_indicator_name,
            series_title=series_title,
            display_period_label=display_period_label,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
        )

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
        calc_mode_cls=calc_mode_cls,
        series_title=series_title,
        freq=freq,
        display_period_label=display_period_label,
        date_range_label=date_range_label,
        reference_period=reference_period,
        is_contribution=is_contribution,
        is_specific_activity=is_specific_activity,
        all_series_data=all_series_data,
        user_question=user_question,
        conversation_context=conversation_context,
        latest_available_period=latest_available_period,
    )
    yield from _stream_llm_or_fallback(
        llm_prompt=llm_prompt,
        llm_temperature=intro_llm_temperature,
        req_form=req_form,
        final_indicator_name=final_indicator_name,
        date_range_label=date_range_label,
        display_period_label=display_period_label,
        obs_to_show=obs_to_show,
        is_contribution=is_contribution,
        latest_available_period=latest_available_period,
        fallback_text=fallback_intro,
    )


def _build_latest_intro_fallback(
    *,
    obs_to_show: List[Dict[str, Any]],
    freq: str,
    indicator_context_val: Optional[str],
    final_indicator_name: str,
    series_title: Optional[str],
    display_period_label: str,
    is_contribution: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
) -> str:
    def _clean_text(value: Any) -> str:
        text = str(value or "").strip()
        if text.lower() in {"", "none", "null", "nan"}:
            return ""
        return text

    def _percentage_es(value: Any) -> str:
        formatted = format_percentage(value)
        return formatted.replace(".", ",") if formatted != "--" else formatted

    def _comparison_label(var_key: Optional[str], freq_value: str) -> str:
        freq_norm = str(freq_value or "").strip().lower()
        if var_key == "yoy":
            if freq_norm == "m":
                return "en comparación con el mismo período del año anterior"
            if freq_norm == "q":
                return "en comparación con el mismo trimestre del año anterior"
            if freq_norm == "a":
                return "en comparación con el año anterior"
            return "en comparación con el mismo período del año anterior"
        if var_key == "prev_period":
            return "en comparación con el período anterior"
        return ""

    row = obs_to_show[0] if obs_to_show else {}
    freq_norm = str(freq or row.get("frequency") or row.get("freq") or "").strip().lower()
    indicator_norm = _clean_text(indicator_context_val).lower()
    if indicator_norm == "pib":
        generic_indicator = "PIB"
    elif indicator_norm == "imacec":
        generic_indicator = "IMACEC"
    else:
        generic_indicator = _clean_text(final_indicator_name) or "indicador"

    if is_contribution:
        contrib_value = row.get("value")
        if contrib_value is None and all_series_data:
            total_row = next(
                (
                    s
                    for s in all_series_data
                    if isinstance(s, dict)
                    and str(s.get("activity") or "").strip().lower() == "total"
                ),
                None,
            )
            if total_row is not None:
                contrib_value = total_row.get("value")

        period_label = _clean_text(display_period_label) or "el último período disponible"
        return (
            f"La contribución de {generic_indicator} en {period_label} fue de {_percentage_es(contrib_value)}, "
            "según datos de la BDE."
        )

    if row.get("yoy") is not None:
        var_key = "yoy"
        var_value = row.get("yoy")
    elif row.get("prev_period") is not None:
        var_key = "prev_period"
        var_value = row.get("prev_period")
    else:
        var_key = None
        var_value = None

    series_desc = _clean_text(series_title) or _clean_text(final_indicator_name)
    if not series_desc:
        series_desc = generic_indicator
    series_desc_lower = series_desc.lower()
    comparison_text = _comparison_label(var_key, freq_norm)
    freq_label = {
        "a": "anual",
        "q": "trimestral",
        "m": "mensual",
    }.get(freq_norm, "periódica")

    if "variación" in series_desc_lower:
        intro_base = f"El dato de la serie {series_desc}"
    else:
        intro_base = f"La variación {freq_label} de la serie {series_desc}"

    if var_value is None:
        return f"{intro_base}, según datos de la BDE, no está disponible."
    if comparison_text:
        return f"{intro_base}, {comparison_text}, fue de {_percentage_es(var_value)}, según datos de la BDE."
    return f"{intro_base} fue de {_percentage_es(var_value)}, según datos de la BDE."


def _deterministic_variation_intro(
    *,
    req_form: str,
    obs_to_show: List[Dict[str, Any]],
    freq: str,
    indicator_context_val: Optional[str],
) -> Iterable[str]:
    if not obs_to_show:
        yield "La variacion no esta disponible aun.\n\n"
        return

    req_form_value = str(req_form or "").strip().lower()
    indicator_value = str(indicator_context_val or "").strip().lower()
    if req_form_value == "range" and indicator_value == "pib":
        first_row = obs_to_show[0] if obs_to_show else {}
        last_row = obs_to_show[-1] if obs_to_show else {}
        first_label = format_period_labels(first_row.get("date"), freq)[0]
        last_label = format_period_labels(last_row.get("date"), freq)[0]
        yield f"A continuación te muestro los valores del PIB entre {first_label} y {last_label}.\n\n"
        return

    row = obs_to_show[-1]
    if row.get("yoy") is not None:
        var_value = row.get("yoy")
    elif row.get("prev_period") is not None:
        var_value = row.get("prev_period")
    else:
        var_value = None

    freq_norm = str(freq or row.get("frequency") or row.get("freq") or "").strip().lower()
    if freq_norm == "a":
        frecuencia = "anual"
    elif freq_norm == "q":
        frecuencia = "trimestral"
    elif freq_norm == "m":
        frecuencia = "mensual"
    else:
        frecuencia = ""

    if var_value is None:
        if frecuencia:
            yield f"La variacion {frecuencia} con respecto al año anterior no esta disponible aun.\n\n"
        else:
            yield "La variacion con respecto al año anterior no esta disponible aun.\n\n"
        return

    if frecuencia:
        yield f"La variacion {frecuencia} con respecto al año anterior es {format_percentage(var_value)}.\n\n"
    else:
        yield f"La variacion con respecto al año anterior es {format_percentage(var_value)}.\n\n"


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
    calc_mode_cls: Optional[str],
    series_title: Optional[str],
    freq: str,
    display_period_label: str,
    date_range_label: str,
    reference_period: Optional[str],
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    user_question: Optional[str],
    conversation_context: Optional[str],
    latest_available_period: Optional[str],
) -> str:
    if req_form in {"range", "specific_point"}:
        return _build_range_prompt(
            obs_to_show=obs_to_show,
            final_indicator_name=final_indicator_name,
            calc_mode_cls=calc_mode_cls,
            series_title=series_title,
            seasonality_context_val=seasonality_context_val,
            freq=freq,
            display_period_label=display_period_label,
            is_contribution=is_contribution,
            user_question=user_question,
            conversation_context=conversation_context,
            latest_available_period=latest_available_period,
        )
    return _build_latest_prompt(
        obs_to_show=obs_to_show,
        final_indicator_name=final_indicator_name,
        calc_mode_cls=calc_mode_cls,
        series_title=series_title,
        indicator_context_val=indicator_context_val,
        component_context_val=component_context_val,
        seasonality_context_val=seasonality_context_val,
        freq=freq,
        display_period_label=display_period_label,
        reference_period=reference_period,
        is_contribution=is_contribution,
        is_specific_activity=is_specific_activity,
        all_series_data=all_series_data,
        user_question=user_question,
        conversation_context=conversation_context,
        latest_available_period=latest_available_period,
    )


def _build_range_prompt(
    *,
    obs_to_show: List[Dict[str, Any]],
    final_indicator_name: str,
    calc_mode_cls: Optional[str],
    series_title: Optional[str],
    seasonality_context_val: Optional[str],
    freq: str,
    display_period_label: str,
    is_contribution: bool,
    user_question: Optional[str],
    conversation_context: Optional[str],
    latest_available_period: Optional[str],
) -> str:
    freq_label = {
        "a": "anual",
        "q": "trimestral",
        "m": "mensual",
    }.get(str(freq or "").strip().lower(), "periódica")
    calc_mode_value = str(calc_mode_cls or "").strip().lower()
    if calc_mode_value == "yoy":
        comparison_text = "con respecto al año anterior"
    elif calc_mode_value == "prev_period":
        comparison_text = f"con respecto al período anterior ({display_period_label})"
    else:
        comparison_text = "con respecto al período de referencia"
    series_title_clean = str(series_title or "").strip()
    if series_title_clean.lower() in {"", "none", "null", "nan"}:
        series_title_clean = ""
    series_desc = str(series_title_clean or final_indicator_name or "serie consultada").strip()
    user_question_text = _normalize_text_field(user_question)
    conversation_context_text = _normalize_text_field(conversation_context)
    latest_period_text = _normalize_text_field(latest_available_period)

    llm_prompt_parts: List[str] = []
    llm_prompt_parts.append("REGLAS DE RESPUESTA (OBLIGATORIAS):")
    if user_question_text:
        llm_prompt_parts.append(f"- Pregunta actual del usuario: {user_question_text}")
    if conversation_context_text:
        llm_prompt_parts.append(
            f"- Contexto conversacional para continuidad (sin desviar el foco): {conversation_context_text}"
        )
    if latest_period_text:
        llm_prompt_parts.append(
            f"- Límite temporal estricto: no menciones ni infieras fechas posteriores a {latest_period_text}."
        )
    llm_prompt_parts.append(
        "- Introducción en UN solo párrafo (máximo 3 oraciones), precisa, fluida y natural."
    )
    llm_prompt_parts.append(
        "- Debe sonar generada de forma natural (no plantilla rígida), manteniendo exactitud factual."
    )
    llm_prompt_parts.append("")
    llm_prompt_parts.append("INSTRUCCIÓN DE INTRODUCCIÓN (OBLIGATORIA):")
    llm_prompt_parts.append(
        "La primera oración debe seguir esta estructura y el LLM puede complementar el cierre: "
        f"'La variación {freq_label} de la {series_desc} {comparison_text}, según los datos de la BDE, es ...'."
    )
    llm_prompt_parts.append("NO uses la palabra 'interanual'.")
    llm_prompt_parts.append("Mantén tono factual, sin markdown, sin viñetas, sin referencias a que eres un modelo.")
    llm_prompt_parts.append("")

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
            f"TAREA: Redacta un párrafo breve que MENCIONE el período ({display_period_label}) y cuánta fue la contribución del cierre (solo porcentaje, 1 decimal). No menciones el valor del índice."
        )
        llm_prompt_parts.append(
            "Termina con una frase que introduzca la tabla (ej: 'La evolución fue:', 'Los datos mes a mes:' o 'El comportamiento fue:'). Factual y neutral."
        )
    else:
        llm_prompt_parts.append(
            f"TAREA: Redacta un párrafo breve que MENCIONE el período ({display_period_label}) y la variación del cierre."
        )
        llm_prompt_parts.append(
            "Termina con una frase que introduzca la tabla (ej: 'La evolución fue:', 'Los datos mes a mes:' o 'El comportamiento fue:'). Factual y neutral."
        )
    return "\n".join(llm_prompt_parts)


def _build_latest_prompt(
    *,
    obs_to_show: List[Dict[str, Any]],
    final_indicator_name: str,
    calc_mode_cls: Optional[str],
    series_title: Optional[str],
    indicator_context_val: Optional[str],
    component_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    freq: str,
    display_period_label: str,
    reference_period: Optional[str],
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    user_question: Optional[str],
    conversation_context: Optional[str],
    latest_available_period: Optional[str],
) -> str:
    def _variation_label(var_key: Optional[str], freq_value: str) -> str:
        freq_norm = str(freq_value or "").strip().lower()
        if var_key == "yoy":
            if freq_norm == "q":
                return "variación interanual trimestral (a/a)"
            if freq_norm == "m":
                return "variación interanual mensual (a/a)"
            if freq_norm == "a":
                return "variación anual con respecto al año anterior"
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
    freq_label = {
        "a": "anual",
        "q": "trimestral",
        "m": "mensual",
    }.get(str(freq or "").strip().lower(), "periódica")
    calc_mode_value = str(calc_mode_cls or "").strip().lower()
    if calc_mode_value == "yoy":
        comparison_text = "con respecto al año anterior"
    elif calc_mode_value == "prev_period":
        comparison_text = f"con respecto al período anterior ({display_period_label})"
    else:
        comparison_text = "con respecto al período de referencia"
    series_desc = str(series_title or final_indicator_name or "serie consultada").strip()
    user_question_text = _normalize_text_field(user_question)
    conversation_context_text = _normalize_text_field(conversation_context)
    latest_period_text = _normalize_text_field(latest_available_period)

    llm_prompt_parts.append("INSTRUCCIÓN DE INTRODUCCIÓN (OBLIGATORIA):")
    llm_prompt_parts.append(
        "La primera oración debe seguir esta estructura y el LLM puede complementar el cierre: "
        f"'La variación {freq_label} de la {series_desc} {comparison_text}, según los datos de la BDE, es ...'."
    )
    llm_prompt_parts.append("NO uses la palabra 'interanual'.")
    llm_prompt_parts.append(
        "Mantén tono factual, sin markdown, sin viñetas, sin referencias a que eres un modelo."
    )
    if user_question_text:
        llm_prompt_parts.append(f"PREGUNTA ACTUAL (prioritaria): {user_question_text}")
    if conversation_context_text:
        llm_prompt_parts.append(
            f"CONTINUIDAD CONVERSACIONAL (solo apoyo, no cambiar foco): {conversation_context_text}"
        )
    if latest_period_text:
        llm_prompt_parts.append(
            f"RESTRICCIÓN TEMPORAL ESTRICTA: no menciones fechas posteriores a {latest_period_text}."
        )
    llm_prompt_parts.append(
        "ESTILO OBLIGATORIO: un solo párrafo, máximo 3 oraciones, preciso y natural (menos determinista)."
    )
    llm_prompt_parts.append("")

    if is_contribution:
        row = obs_to_show[0]
        seasonality_lower = (seasonality_context_val or "").lower()
        prefer_yoy = seasonality_lower == "nsa"
        contrib_label = "contribución interanual" if prefer_yoy else "contribución vs período anterior"
        contrib_value = row.get("value")
        if contrib_value is None and all_series_data:
            total_row = next(
                (
                    s
                    for s in all_series_data
                    if isinstance(s, dict)
                    and str(s.get("activity") or "").strip().lower() == "total"
                ),
                None,
            )
            if total_row is not None:
                contrib_value = total_row.get("value")

        if is_specific_activity and all_series_data:
            imacec_total_value = None
            for s in all_series_data:
                if s.get("activity", "").lower() == "total":
                    imacec_total_value = s.get("value")
                    break

            period_comparison = {
                "q": "igual trimestre del año anterior",
                "m": "igual mes del año anterior",
                "a": "igual año anterior",
            }.get(str(freq or "").strip().lower(), "igual período del año anterior")

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
            if imacec_total_value is not None:
                llm_prompt_parts.append(f"- Variación total: {format_percentage(imacec_total_value)}")
            llm_prompt_parts.append(f"- Actividad con mayor contribución: {activity_name}")
            llm_prompt_parts.append(f"- Valor de contribución: {format_percentage(contrib_value)}")
            llm_prompt_parts.append("")
            llm_prompt_parts.append("TAREA:")
            llm_prompt_parts.append("Redacta 2 oraciones siguiendo esta estructura (sin usar comillas):")
            llm_prompt_parts.append("")
            llm_prompt_parts.append("Primera oración:")
            if imacec_total_value is not None:
                llm_prompt_parts.append(
                    f"- Inicia con: De acuerdo con la BDE, el {indicator_context_val.upper()} de [período] creció [variación] en comparación con {period_comparison} (ver tabla)."
                )
            else:
                llm_prompt_parts.append(
                    f"- Inicia con: De acuerdo con la BDE, en [período] la actividad [nombre actividad] aportó [valor] al crecimiento del {indicator_context_val.upper()} (ver tabla)."
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
            if reference_period:
                llm_prompt_parts.append(
                    f"PERÍODO OBLIGATORIO: usa exactamente el período {display_period_label} (referencia técnica: {reference_period}) y NO menciones ninguna otra fecha."
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
            if freq_raw == "a":
                llm_prompt_parts.append("IMPORTANTE: para frecuencia anual, inicia con: 'La variación anual con respecto al año anterior es ...'")
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
    llm_temperature: float,
    req_form: str,
    final_indicator_name: str,
    date_range_label: str,
    display_period_label: str,
    obs_to_show: List[Dict[str, Any]],
    is_contribution: bool,
    latest_available_period: Optional[str],
    fallback_text: Optional[str] = None,
) -> Iterable[str]:
    def _sanitize_generated_text(text: str) -> str:
        clean_text = str(text or "").strip()
        if not clean_text:
            return ""
        lowered = clean_text.lower()
        if any(token in lowered for token in ["none", "null", "nan"]):
            return ""
        return clean_text

    try:
        llm = build_llm(streaming=True, temperature=llm_temperature, mode="fallback")
        chunks: List[str] = []
        for chunk in llm.stream(llm_prompt, history=[], intent_info=None):
            text = str(chunk)
            if text:
                chunks.append(text)

        generated_text = _sanitize_generated_text("".join(chunks))
        if not generated_text or generated_text.lower().startswith("(error generando)"):
            raise RuntimeError("llm_generation_failed")

        final_text = generated_text
        if req_form == "latest" and is_contribution:
            period_label = str(display_period_label or "").strip()
            if period_label and period_label != "--":
                if period_label.lower() not in generated_text.lower():
                    row = obs_to_show[0] if obs_to_show else {}
                    contrib_value = row.get("value") if isinstance(row, dict) else None
                    if contrib_value is not None:
                        enforced = (
                            f"La contribución de {final_indicator_name} en {period_label} fue {format_percentage(contrib_value)}."
                        )
                    else:
                        enforced = (
                            f"La contribución de {final_indicator_name} para {period_label} se presenta en la siguiente tabla."
                        )
                    final_text = f"{enforced} {generated_text}".strip()

        if _has_year_posterior_than_limit(final_text, latest_available_period):
            limit_label = _normalize_text_field(latest_available_period) or "la última actualización disponible"
            final_text = (
                f"No puedo informar fechas posteriores a {limit_label}. "
                f"Te comparto el último dato disponible en el período válido y su variación en la tabla."
            )

        yield final_text
        yield "\n\n"


    # === Secciones específicas: tabla, referencias, sugerencias, adjuntos ===
    except Exception as exc:
        logger.warning("Error generando con LLM: %s", exc)
        if fallback_text:
            yield fallback_text
            yield "\n\n"
            return
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

        valid_series = [s for s in all_series_data if isinstance(s, dict)]
        series_by_activity = {
            str(s.get("activity") or "").strip().lower(): s
            for s in valid_series
            if str(s.get("activity") or "").strip()
        }
        has_imacec_breakdown = "total" in series_by_activity

        max_activity = None
        max_value = float("-inf")
        if has_imacec_breakdown:
            for s in valid_series:
                activity = str(s.get("activity") or "").strip().lower()
                value = s.get("value", 0)
                if activity and activity != "total" and value > max_value:
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
        else:
            max_title = None
            for s in valid_series:
                title = str(s.get("title") or s.get("activity") or "").strip() or "Actividad"
                value = s.get("value", 0)
                if value > max_value:
                    max_value = value
                    max_title = title

            for s in valid_series:
                title = str(s.get("title") or s.get("activity") or "").strip() or "Actividad"
                value = s.get("value")
                if title == max_title:
                    yield f"**{title}** | **{format_percentage(value)}**\n"
                else:
                    yield f"{title} | {format_percentage(value)}\n"

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


def _default_charts(
    *,
    series_id: str,
    obs_to_show: List[Dict[str, Any]],
    req_form: str,
    indicator_context_val: Optional[str],
    chart_context_series: Optional[List[str]],
) -> Iterable[str]:
    if not obs_to_show:
        return iter(())

    safe_series = str(series_id or "").strip()
    if not safe_series:
        return iter(())

    indicator_label = str(indicator_context_val or "").strip().upper() or "SERIE"
    chart_type = "line"
    if req_form in {"range", "specific_point"}:
        chart_type = "line"

    recent_series = [
        str(item).strip()
        for item in (chart_context_series or [])
        if str(item).strip()
    ]
    if safe_series not in recent_series:
        recent_series.insert(0, safe_series)
    recent_series = recent_series[:5]

    rows: List[Dict[str, Any]] = []
    for row in obs_to_show:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "date": row.get("date"),
                "value": row.get("value"),
                "yoy_pct": row.get("yoy"),
                "prev_period": row.get("prev_period"),
            }
        )

    if not rows:
        return iter(())

    temp_path = None
    try:
        import pandas as _pd

        df = _pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
            prefix="chart_",
            mode="w",
            encoding="utf-8",
        ) as tmp:
            df.to_csv(tmp, index=False)
            temp_path = tmp.name
    except Exception:
        try:
            import csv as _csv

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".csv",
                prefix="chart_",
                mode="w",
                encoding="utf-8",
                newline="",
            ) as tmp:
                writer = _csv.DictWriter(tmp, fieldnames=["date", "value", "yoy_pct", "prev_period"])
                writer.writeheader()
                writer.writerows(rows)
                temp_path = tmp.name
        except Exception:
            temp_path = None

    if not temp_path or not Path(temp_path).exists():
        return iter(())

    block: List[str] = ["##CHART_START\n"]
    block.append(f"title=Evolución de {indicator_label}\n")
    block.append(f"domain={indicator_label}\n")
    block.append(f"type={chart_type}\n")
    block.append(f"data_path={temp_path}\n")
    block.append(f"series_id={safe_series}\n")
    block.append(f"recent_series={','.join(recent_series)}\n")
    block.append("##CHART_END\n")
    return iter(block)


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
