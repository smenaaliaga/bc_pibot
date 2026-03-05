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
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from orchestrator.llm.llm_adapter import build_llm

logger = logging.getLogger(__name__)
BDE_SERIES_URL = "https://si3.bcentral.cl/siete/"


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

def _indicator_display_name(
    *,
    indicator_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    fallback: Optional[str] = None,
) -> str:
    indicator_norm = str(indicator_context_val or "").strip().lower()
    fallback_text = str(fallback or "").strip()
    if indicator_norm == "pib":
        base = "PIB"
    elif indicator_norm == "imacec":
        base = "IMACEC"
    elif fallback_text:
        base = fallback_text
    else:
        base = "INDICADOR"

    seasonality_norm = str(seasonality_context_val or "").strip().lower()
    if (
        seasonality_norm == "sa"
        and str(base or "").strip().upper() in {"PIB", "IMACEC"}
        and "desestacionalizado" not in str(base or "").strip().lower()
    ):
        return f"{base} desestacionalizado"
    return base


def build_no_series_message(
    *,
    question: Optional[str] = None,
    requested_activity: Optional[str] = None,
    normalized_activity: Optional[str] = None,
    indicator_label: Optional[str] = None,
) -> str:
    return (
        "No encontré una serie que coincida con tu consulta. "
        f"Puedes explorar más series directamente en la BDE: 🔗 {BDE_SERIES_URL}."
    )


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

        freq_norm = str(freq or "").strip().upper()

        if freq_norm in {"Q", "T", "QUARTERLY", "TRIMESTRAL"}:
            q = ((m - 1) // 3) + 1
            ordinal = {1: "1er", 2: "2do", 3: "3er", 4: "4to"}.get(q, f"{q}º")
            long_label = f"{ordinal} trimestre {y}"
            short_label = f"{q}T {y}"
            return [long_label, short_label]
        if freq_norm in {"A", "ANNUAL", "ANUAL"}:
            return [f"el año {y}", str(y)]
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
        return f"{float(value):.1f}%".replace(".", ",")
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
    calc_mode_cls: Optional[str],
    intent_cls: Optional[str],
    display_period_label: str,
    freq: str,
    date_range_label: str,
    reference_period: Optional[str],
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    used_latest_fallback_for_point: bool = False,
    source_urls: List[str],
    intro_llm_temperature: float = 0.7,
    question: Optional[str] = None,
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
            calc_mode_cls=calc_mode_cls,
            series_title=series_title,
            freq=freq,
            display_period_label=display_period_label,
            date_range_label=date_range_label,
            reference_period=reference_period,
            is_contribution=is_contribution,
            is_specific_activity=is_specific_activity,
            all_series_data=all_series_data,
            used_latest_fallback_for_point=used_latest_fallback_for_point,
            intro_llm_temperature=intro_llm_temperature,
            question=question,
        ),
        metadata=_metadata_block(series_id, series_title),
        table=_specific_table(
            req_form=req_form,
            obs_to_show=obs_to_show,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
            freq=freq,
            component_context_val=component_context_val,
            is_specific_activity=is_specific_activity,
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
    calc_mode_cls: Optional[str],
    intent_cls: Optional[str],
    display_period_label: str,
    freq: str,
    date_range_label: str,
    reference_period: Optional[str],
    is_contribution: bool,
    is_specific_activity: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    used_latest_fallback_for_point: bool = False,
    source_urls: List[str],
    intro_llm_temperature: float = 0.7,
    question: Optional[str] = None,
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
            calc_mode_cls=calc_mode_cls,
            series_title=series_title,
            freq=freq,
            display_period_label=display_period_label,
            date_range_label=date_range_label,
            reference_period=reference_period,
            is_contribution=is_contribution,
            is_specific_activity=is_specific_activity,
            all_series_data=all_series_data,
            used_latest_fallback_for_point=used_latest_fallback_for_point,
            intro_llm_temperature=intro_llm_temperature,
            question=question,
        ),
        metadata=_metadata_block(series_id, series_title),
        table=_specific_table(
            req_form="specific_point",
            obs_to_show=obs_to_show,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
            freq=freq,
            component_context_val=component_context_val,
            is_specific_activity=is_specific_activity,
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
    return ()


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
    used_latest_fallback_for_point: bool = False,
    intro_llm_temperature: float = 0.7,
    question: Optional[str] = None,
) -> Iterable[str]:
    if req_form in {"latest", "point", "specific_point"} and is_contribution:
        yield _build_latest_contribution_intro(
            obs_to_show=obs_to_show,
            indicator_context_val=indicator_context_val,
            seasonality_context_val=seasonality_context_val,
            display_period_label=display_period_label,
            all_series_data=all_series_data,
            freq=freq,
            component_context_val=component_context_val,
            is_specific_activity=is_specific_activity,
            used_latest_fallback_for_point=used_latest_fallback_for_point,
            question=question,
        )
        yield "\n\n"
        return

    if req_form == "latest" and not is_contribution and isinstance(all_series_data, list) and all_series_data:
        yield _build_latest_intro_fallback(
            obs_to_show=obs_to_show,
            freq=freq,
            indicator_context_val=indicator_context_val,
            seasonality_context_val=seasonality_context_val,
            final_indicator_name=final_indicator_name,
            series_title=series_title,
            display_period_label=display_period_label,
            is_contribution=is_contribution,
            all_series_data=all_series_data,
            question=question,
        )
        yield "\n\n"
        return

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
            seasonality_context_val=seasonality_context_val,
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
    )
    yield from _stream_llm_or_fallback(
        llm_prompt=llm_prompt,
        llm_temperature=intro_llm_temperature,
        req_form=req_form,
        indicator_context_val=indicator_context_val,
        seasonality_context_val=seasonality_context_val,
        final_indicator_name=final_indicator_name,
        series_title=series_title,
        freq=freq,
        date_range_label=date_range_label,
        display_period_label=display_period_label,
        obs_to_show=obs_to_show,
        is_contribution=is_contribution,
        all_series_data=all_series_data,
        used_latest_fallback_for_point=used_latest_fallback_for_point,
        fallback_text=fallback_intro,
    )


def _build_latest_contribution_intro(
    *,
    obs_to_show: List[Dict[str, Any]],
    indicator_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    display_period_label: str,
    all_series_data: Optional[List[Dict[str, Any]]],
    freq: Optional[str] = None,
    component_context_val: Optional[str] = None,
    is_specific_activity: bool = False,
    used_latest_fallback_for_point: bool = False,
    question: Optional[str] = None,
) -> str:
    def _is_aggregate_indicator_row(series: Dict[str, Any]) -> bool:
        activity_value = str(series.get("activity") or "").strip().lower()
        region_value = str(series.get("region") or "").strip().lower()
        investment_value = str(series.get("investment") or "").strip().lower()
        title_value = str(series.get("title") or "").strip().lower()

        if activity_value in {"total", "imacec", "pib"}:
            return True

        has_dimension = (
            (activity_value and activity_value not in {"total", "imacec", "pib"})
            or (region_value and region_value not in {"total", "general", "none", "null"})
            or (investment_value and investment_value not in {"total", "general", "none", "null"})
        )
        if title_value in {"subtotal regionalizado", "extrarregional"}:
            return True
        if not has_dimension and title_value in {"pib", "imacec", "producto interno bruto"}:
            return True
        if investment_value == "demanda_interna" and title_value == "demanda interna":
            return True
        return False

    def _percentage_es(value: Any, *, absolute: bool = False) -> str:
        try:
            numeric = float(value)
            if absolute:
                numeric = abs(numeric)
            return f"{numeric:.1f}%".replace(".", ",")
        except Exception:
            return "--"

    def _normalize_period_label(period_text: str) -> str:
        text = str(period_text or "").strip()
        if not text:
            return "el último período disponible"
        parts = text.split()
        if len(parts) == 2 and parts[1].isdigit():
            month = parts[0].lower()
            year = parts[1]
            return f"{month} de {year}"
        return text

    def _comparison_phrase(freq_value: Optional[str]) -> str:
        freq_norm = str(freq_value or "").strip().lower()
        if freq_norm == "m":
            return "en comparación con igual mes del año anterior"
        if freq_norm == "q":
            return "en comparación con igual trimestre del año anterior"
        if freq_norm == "a":
            return "en comparación con igual año anterior"
        return "en comparación con igual período del año anterior"

    def _sector_phrase(raw_sector_name: str) -> str:
        normalized = str(raw_sector_name or "").strip()
        normalized_lower = normalized.lower()
        mapping = {
            "no minero": "el sector no minero",
            "producción de bienes": "la producción de bienes",
            "minería": "la minería",
            "industria": "la industria",
            "industria manufacturera": "la industria manufacturera",
            "resto de bienes": "el resto de bienes",
            "comercio": "el comercio",
            "servicios": "los servicios",
            "impuestos sobre los productos": "los impuestos sobre los productos",
        }
        mapped = mapping.get(normalized_lower)
        if mapped:
            return mapped
        if normalized_lower.startswith("región ") or normalized_lower.startswith("region "):
            return f"la {normalized}"
        return normalized

    def _normalize_activity_key(raw_value: Any) -> str:
        return str(raw_value or "").strip().lower().replace(" ", "_")

    def _prepend_de(noun_phrase: str) -> str:
        phrase = str(noun_phrase or "").strip()
        lowered = phrase.lower()
        if lowered.startswith("el "):
            return f"del {phrase[3:]}"
        if lowered.startswith("la "):
            return f"de la {phrase[3:]}"
        if lowered.startswith("los "):
            return f"de los {phrase[4:]}"
        if lowered.startswith("las "):
            return f"de las {phrase[4:]}"
        return f"de {phrase}"

    def _indicator_with_period(indicator: str, period_label: str) -> str:
        period_clean = str(period_label or "").strip()
        if period_clean.lower().startswith("el "):
            return f"el {indicator} del {period_clean[3:]}"
        return f"el {indicator} de {period_clean}"

    def _find_aggregate_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for series in rows:
            if not isinstance(series, dict):
                continue
            activity_value = str(series.get("activity") or "").strip().lower()
            title_value = str(series.get("title") or "").strip().lower()
            investment_value = str(series.get("investment") or "").strip().lower()
            if activity_value in {"total", "imacec", "pib"}:
                return series
            if title_value in {"pib", "imacec", "producto interno bruto"}:
                return series
            if investment_value == "demanda_interna" and title_value == "demanda interna":
                return series

        for series in rows:
            if not isinstance(series, dict):
                continue
            if _is_aggregate_indicator_row(series):
                return series
        return None

    indicator_raw = _indicator_display_name(
        indicator_context_val=indicator_context_val,
        seasonality_context_val=seasonality_context_val,
        fallback="INDICADOR",
    )

    row = obs_to_show[0] if obs_to_show else {}
    valid_rows = [series for series in all_series_data if isinstance(series, dict)] if isinstance(all_series_data, list) else []
    aggregate_row = _find_aggregate_row(valid_rows)
    base_row = aggregate_row if isinstance(aggregate_row, dict) else row
    contrib_value = base_row.get("value") if isinstance(base_row, dict) else None
    period_text = _normalize_period_label(display_period_label)
    observed_period_text = _normalize_period_label(
        format_period_labels(base_row.get("date") if isinstance(base_row, dict) else None, str(freq or ""))[0]
    )
    comparison_phrase = _comparison_phrase(freq)

    # Detectar si el agregado es "Demanda interna" para ajustar el sujeto
    aggregate_title_norm = str(base_row.get("title") or "").strip().lower() if isinstance(base_row, dict) else ""
    is_demanda_interna_aggregate = aggregate_title_norm == "demanda interna"

    if is_demanda_interna_aggregate:
        def _di_subject(period_label: str) -> str:
            p = str(period_label or "").strip()
            if p.lower().startswith("el "):
                return f"la Demanda interna del {p[3:]}"
            return f"la Demanda interna de {p}" if p else "la Demanda interna"
        requested_subject = _di_subject(period_text)
        observed_subject = _di_subject(observed_period_text)
    else:
        requested_subject = _indicator_with_period(indicator_raw, period_text)
        observed_subject = _indicator_with_period(indicator_raw, observed_period_text)

    first_sentence = (
        f"De acuerdo con la información publicada en la BDE, {requested_subject} "
        f"creció {_percentage_es(contrib_value, absolute=True)} {comparison_phrase}."
    )
    try:
        if float(contrib_value) < 0:
            first_sentence = (
                f"De acuerdo con la información publicada en la BDE, {requested_subject} "
                f"cayó {_percentage_es(contrib_value, absolute=True)} {comparison_phrase}."
            )
    except Exception:
        first_sentence = (
            f"De acuerdo con la información publicada en la BDE, {requested_subject} "
            f"registró una variación de {_percentage_es(contrib_value)} {comparison_phrase}."
        )

    fallback_period_mismatch = (
        used_latest_fallback_for_point
        and period_text
        and observed_period_text
        and period_text != observed_period_text
    )
    if fallback_period_mismatch:
        try:
            if float(contrib_value) < 0:
                first_sentence = (
                    f"No hay datos disponibles para {period_text}; sin embargo, según la BDE, {observed_subject} "
                    f"cayó {_percentage_es(contrib_value, absolute=True)} {comparison_phrase}."
                )
            else:
                first_sentence = (
                    f"No hay datos disponibles para {period_text}; sin embargo, según la BDE, {observed_subject} "
                    f"creció {_percentage_es(contrib_value, absolute=True)} {comparison_phrase}."
                )
        except Exception:
            first_sentence = (
                f"No hay datos disponibles para {period_text}; sin embargo, según la BDE, {observed_subject} "
                f"registró una variación de {_percentage_es(contrib_value)} {comparison_phrase}."
            )

    if not isinstance(all_series_data, list) or not all_series_data:
        return first_sentence

    if (
        len(valid_rows) == 1
        and isinstance(valid_rows[0], dict)
        and not _is_aggregate_indicator_row(valid_rows[0])
    ):
        selected_row = valid_rows[0]
        selected_title = str(
            selected_row.get("title")
            or selected_row.get("activity")
            or selected_row.get("region")
            or selected_row.get("investment")
            or ""
        ).strip().replace("_", " ")
        if selected_title:
            return (
                f"De acuerdo con la información publicada en la BDE, la contribución de {_sector_phrase(selected_title)} "
                f"al {indicator_raw} de {period_text} fue de {_percentage_es(selected_row.get('value'))} {comparison_phrase}."
            )

    if is_specific_activity and component_context_val:
        requested_activity = _normalize_activity_key(component_context_val)
        requested_row = next(
            (
                series
                for series in valid_rows
                if _normalize_activity_key(series.get("activity")) == requested_activity
            ),
            None,
        )
        if isinstance(requested_row, dict):
            requested_title = str(
                requested_row.get("title") or requested_row.get("activity") or component_context_val
            ).strip().replace("_", " ")
            requested_value = requested_row.get("value")
            return (
                f"De acuerdo con la información publicada en la BDE, la contribución de {_sector_phrase(requested_title)} "
                f"al {indicator_raw} de {period_text} fue de {_percentage_es(requested_value)} {comparison_phrase}."
            )

    candidates: List[Dict[str, Any]] = []
    for series in valid_rows:
        if _is_aggregate_indicator_row(series):
            continue

        value = series.get("value")
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        candidates.append(series)

    if not candidates:
        return first_sentence

    # ---------- LLM-based analysis ----------
    def _build_candidates_table() -> str:
        lines: List[str] = []
        for c in candidates:
            name = str(
                c.get("title") or c.get("activity") or c.get("region") or c.get("investment") or ""
            ).strip().replace("_", " ")
            val = _percentage_es(c.get("value"))
            lines.append(f"- {name}: {val}")
        return "\n".join(lines)

    def _deterministic_top_sentence() -> str:
        top_sector = max(candidates, key=lambda s: float(s.get("value") or 0.0))
        name = str(
            top_sector.get("title") or top_sector.get("activity")
            or top_sector.get("region") or top_sector.get("investment") or ""
        ).strip().replace("_", " ")
        if not name:
            return ""
        return (
            f"La mayor contribución provino {_prepend_de(_sector_phrase(name))}, "
            f"con {_percentage_es(top_sector.get('value'))}."
        )

    user_question = str(question or "").strip()
    if not user_question:
        fallback = _deterministic_top_sentence()
        return f"{first_sentence} {fallback}" if fallback else first_sentence

    candidates_table = _build_candidates_table()
    llm_prompt = (
        "Eres un analista económico del Banco Central de Chile.\n"
        f"El usuario preguntó: \"{user_question}\"\n\n"
        f"Contexto: {first_sentence}\n\n"
        f"Componentes disponibles (contribución al {indicator_raw}):\n{candidates_table}\n\n"
        "Instrucciones ESTRICTAS:\n"
        "- El texto de 'Contexto' YA se le mostró al usuario. NUNCA repitas, parafrasees ni resumas ese contexto. "
        "No menciones de nuevo el indicador, el período ni el porcentaje total. Comienza DIRECTAMENTE con el análisis de los componentes.\n"
        "- Responde ÚNICAMENTE lo que el usuario preguntó. NO agregues información que no fue solicitada.\n"
        "- DISTINCIÓN SEMÁNTICA CLAVE entre 'menos aportó' y 'restó':\n"
        "  * 'la que MENOS APORTÓ al crecimiento' = la de MENOR contribución POSITIVA (valor positivo más bajo). NO son las negativas.\n"
        "  * 'la que RESTÓ / DETRAJO del crecimiento' = la de contribución NEGATIVA.\n"
        "  Respeta esta diferencia rigurosamente.\n"
        "- Si pregunta qué actividad explicó el crecimiento, menciona SOLO la(s) de contribución positiva relevante(s). No menciones las negativas a menos que se pida.\n"
        "- Si pregunta cuál 'menos aportó', indica la de menor valor positivo (ej: 0,1%), NO las de valor negativo.\n"
        "- Si pregunta qué restó o detrajo del crecimiento, menciona SOLO las de contribución negativa.\n"
        "- Si pregunta por las N principales, lista exactamente N, ordenadas.\n"
        "- Usa formato de porcentajes con coma decimal (ej: 1,2%).\n"
        "- Responde en español, de forma concisa (1-2 oraciones máximo).\n"
        "- NO inventes datos; usa solo los valores del listado anterior.\n"
    )

    try:
        llm = build_llm(streaming=False, temperature=0.3, mode="fallback")
        raw_response = llm.generate(llm_prompt, history=[], intent_info=None)
        llm_text = str(raw_response or "").strip()
        if llm_text and "(error generando)" not in llm_text.lower():
            logger.info("[CONTRIBUTION_INTRO] LLM analysis OK (%d chars)", len(llm_text))
            return f"{first_sentence} {llm_text}"
    except Exception as exc:
        logger.warning("[CONTRIBUTION_INTRO] LLM analysis failed: %s", exc)

    fallback = _deterministic_top_sentence()
    return f"{first_sentence} {fallback}" if fallback else first_sentence


def _build_latest_intro_fallback(
    *,
    obs_to_show: List[Dict[str, Any]],
    freq: str,
    indicator_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    final_indicator_name: str,
    series_title: Optional[str],
    display_period_label: str,
    is_contribution: bool,
    all_series_data: Optional[List[Dict[str, Any]]],
    question: Optional[str] = None,
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

    def _build_top_variation_sentence(
        rows: Optional[List[Dict[str, Any]]],
        indicator_key: str,
        indicator_label_base: str,
    ) -> Optional[str]:
        if not isinstance(rows, list) or not rows:
            return None

        candidates: List[Tuple[float, str]] = []
        activity_candidate_count = 0
        region_candidate_count = 0
        investment_candidate_count = 0
        for series in rows:
            if not isinstance(series, dict):
                continue

            activity_value = str(series.get("activity") or "").strip().lower()
            region_value = str(series.get("region") or "").strip().lower()
            investment_value = str(series.get("investment") or "").strip().lower()

            has_activity_dimension = bool(activity_value) and activity_value not in {"total", "imacec", "pib"}
            has_region_dimension = bool(region_value) and region_value not in {"total", "general", "none"}
            has_investment_dimension = bool(investment_value) and investment_value not in {"total", "general", "none"}

            if has_activity_dimension:
                activity_candidate_count += 1
            if has_region_dimension:
                region_candidate_count += 1
            if has_investment_dimension:
                investment_candidate_count += 1

            var_candidate = series.get("comparison_value")
            if var_candidate is None:
                var_candidate = series.get("yoy") if series.get("yoy") is not None else series.get("prev_period")
            try:
                var_numeric = float(var_candidate)
            except (TypeError, ValueError):
                continue

            series_name = str(
                series.get("title")
                or series.get("activity")
                or series.get("region")
                or series.get("investment")
                or ""
            ).strip().replace("_", " ")
            if not series_name:
                continue

            series_name_norm = series_name.lower().strip()
            if series_name_norm in {
                "total",
                "pib",
                "imacec",
                "producto interno bruto",
                "pib a costo de factores",
            }:
                continue

            candidates.append((var_numeric, series_name))

        if not candidates:
            return None

        top_var, top_name = max(candidates, key=lambda item: item[0])
        base_has_sa = "desestacionalizado" in str(indicator_label_base or "").strip().lower()
        if (
            indicator_key == "pib"
            and region_candidate_count > 0
            and activity_candidate_count == 0
            and investment_candidate_count == 0
        ):
            indicator_label = "PIB regional desestacionalizado" if base_has_sa else "PIB regional"
        elif indicator_key == "imacec":
            indicator_label = "IMACEC desestacionalizado" if base_has_sa else "IMACEC"
        elif indicator_key == "pib":
            indicator_label = "PIB desestacionalizado" if base_has_sa else "PIB"
        else:
            indicator_label = "indicador"
        return (
            f"La mayor variación en el desglose del {indicator_label} la registró {top_name}, "
            f"con {_percentage_es(top_var)}."
        )

    row = obs_to_show[0] if obs_to_show else {}
    freq_norm = str(freq or row.get("frequency") or row.get("freq") or "").strip().lower()
    indicator_norm = _clean_text(indicator_context_val).lower()
    generic_indicator = _indicator_display_name(
        indicator_context_val=indicator_context_val,
        seasonality_context_val=seasonality_context_val,
        fallback=_clean_text(final_indicator_name) or "indicador",
    )

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

    row_period_label = format_period_labels(row.get("date"), freq_norm or freq)[0]
    display_period_clean = _clean_text(display_period_label)
    effective_period_label = (
        row_period_label
        if str(row_period_label or "").strip() not in {"", "--"}
        else display_period_clean
    )
    if not effective_period_label:
        effective_period_label = "el último período reportado"

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
        value_text = format_value(row.get("value"))
        return (
            f"{intro_base}, correspondiente a {effective_period_label} (último valor reportado por la BDE), "
            f"no está disponible; el valor informado para ese período es {value_text}."
        )
    if comparison_text:
        base_text = (
            f"{intro_base}, {comparison_text}, correspondiente a {effective_period_label} "
            f"(último valor reportado por la BDE), fue de {_percentage_es(var_value)}."
        )
    else:
        base_text = (
            f"{intro_base}, correspondiente a {effective_period_label} (último valor reportado por la BDE), "
            f"fue de {_percentage_es(var_value)}."
        )

    top_variation_sentence = _build_top_variation_sentence(
        all_series_data,
        indicator_norm,
        generic_indicator,
    )

    # ---------- LLM-based analysis for non-contribution breakdowns ----------
    user_question = str(question or "").strip()
    if user_question and isinstance(all_series_data, list) and all_series_data:
        variation_candidates: List[Tuple[str, str]] = []
        for series in all_series_data:
            if not isinstance(series, dict):
                continue
            series_name = str(
                series.get("title") or series.get("activity")
                or series.get("region") or series.get("investment") or ""
            ).strip().replace("_", " ")
            series_name_norm = series_name.lower().strip()
            if series_name_norm in {"total", "pib", "imacec", "producto interno bruto", "pib a costo de factores", ""}:
                continue
            var_val = series.get("comparison_value")
            if var_val is None:
                var_val = series.get("yoy") if series.get("yoy") is not None else series.get("prev_period")
            try:
                var_pct = f"{float(var_val):.1f}%".replace(".", ",")
            except (TypeError, ValueError):
                continue
            variation_candidates.append((series_name, var_pct))

        if variation_candidates:
            candidates_text = "\n".join(f"- {n}: {v}" for n, v in variation_candidates)
            llm_prompt = (
                "Eres un analista económico del Banco Central de Chile.\n"
                f"El usuario preguntó: \"{user_question}\"\n\n"
                f"Contexto: {base_text}\n\n"
                f"Componentes del desglose del {generic_indicator} (variación %):\n{candidates_text}\n\n"
                "Instrucciones ESTRICTAS:\n"
                "- El texto de 'Contexto' YA se le mostró al usuario. NUNCA repitas, parafrasees ni resumas ese contexto. "
                "No menciones de nuevo el indicador, el período ni el porcentaje total. Comienza DIRECTAMENTE con el análisis de los componentes.\n"
                "- Responde ÚNICAMENTE lo que el usuario preguntó. NO agregues información que no fue solicitada.\n"
                "- DISTINCIÓN SEMÁNTICA CLAVE entre 'menos aportó/creció' y 'restó/cayó':\n"
                "  * 'la que MENOS CRECIÓ / MENOS APORTÓ' = la de MENOR variación POSITIVA (valor positivo más bajo). NO son las negativas.\n"
                "  * 'la que CAYÓ / RESTÓ / SE CONTRAJO' = la de variación NEGATIVA.\n"
                "  Respeta esta diferencia rigurosamente.\n"
                "- Si pregunta qué actividad explicó el crecimiento, menciona SOLO la(s) de variación positiva relevante(s). No menciones las negativas a menos que se pida.\n"
                "- Si pregunta cuál 'menos creció' o 'menos aportó', indica la de menor valor positivo (ej: 0,1%), NO las de valor negativo.\n"
                "- Si pregunta qué cayó o restó al crecimiento, menciona SOLO las de variación negativa.\n"
                "- Si pregunta por las N principales, lista exactamente N, ordenadas.\n"
                "- Usa formato de porcentajes con coma decimal (ej: 1,2%).\n"
                "- Responde en español, de forma concisa (1-2 oraciones máximo).\n"
                "- NO inventes datos; usa solo los valores del listado anterior.\n"
            )
            try:
                llm = build_llm(streaming=False, temperature=0.3, mode="fallback")
                raw_response = llm.generate(llm_prompt, history=[], intent_info=None)
                llm_text = str(raw_response or "").strip()
                if llm_text and "(error generando)" not in llm_text.lower():
                    logger.info("[FALLBACK_INTRO] LLM analysis OK (%d chars)", len(llm_text))
                    return f"{base_text} {llm_text}"
            except Exception as exc:
                logger.warning("[FALLBACK_INTRO] LLM analysis failed: %s", exc)

    if top_variation_sentence:
        return f"{base_text} {top_variation_sentence}"

    return (
        f"{intro_base}, correspondiente a {effective_period_label} (último valor reportado por la BDE), "
        f"fue de {_percentage_es(var_value)}."
    )


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
        )
    return _build_latest_prompt(
        req_form=req_form,
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
    indicator_desc = series_desc or str(final_indicator_name or "").strip() or "serie consultada"

    llm_prompt_parts: List[str] = []
    llm_prompt_parts.append("INSTRUCCIÓN DE INTRODUCCIÓN (OBLIGATORIA):")
    llm_prompt_parts.append(
        "La primera oración debe seguir esta estructura y el LLM puede complementar el cierre: "
        f"'La variación {freq_label} para {series_desc} {comparison_text}, según los datos de la BDE, es ...'."
    )
    llm_prompt_parts.append("NO uses la palabra 'interanual'.")
    llm_prompt_parts.append(
        "Mantén tono factual, sin markdown, sin viñetas, sin referencias a que eres un modelo."
    )
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
                f"El usuario preguntó por {indicator_desc} en el período: {display_period_label}."
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
    req_form: str,
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
    indicator_desc = series_desc or str(final_indicator_name or "").strip() or "serie consultada"

    llm_prompt_parts.append("INSTRUCCIÓN DE INTRODUCCIÓN (OBLIGATORIA):")
    llm_prompt_parts.append(
        "La primera oración debe seguir esta estructura y el LLM puede complementar el cierre: "
        f"'La variación {freq_label} para {series_desc} {comparison_text}, según los datos de la BDE, es ...'."
    )
    llm_prompt_parts.append("NO uses la palabra 'interanual'.")
    llm_prompt_parts.append(
        "Mantén tono factual, sin markdown, sin viñetas, sin referencias a que eres un modelo."
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
        latest_period_label = format_period_labels(row.get("date"), freq_raw or freq)[0]
        if str(latest_period_label or "").strip() in {"", "--"}:
            latest_period_label = display_period_label

        llm_prompt_parts.append("SITUACIÓN: El usuario preguntó por un dato económico específico.")
        llm_prompt_parts.append("Reporta solo la variación (máximo 2 oraciones) informando:")
        llm_prompt_parts.append(f"- Indicador: {indicator_desc}")
        seasonality_norm = str(seasonality_context_val or "").strip().lower()
        indicator_norm = str(indicator_context_val or "").strip().lower()
        if seasonality_norm == "sa" and indicator_norm in {"pib", "imacec"}:
            indicator_sa_label = _indicator_display_name(
                indicator_context_val=indicator_context_val,
                seasonality_context_val=seasonality_context_val,
                fallback=final_indicator_name,
            )
            llm_prompt_parts.append(
                f"- IMPORTANTE: menciona explícitamente que corresponde a {indicator_sa_label}."
            )
        llm_prompt_parts.append(f"- Período: {display_period_label}")
        if var_value is not None:
            llm_prompt_parts.append(f"- {var_label}: {format_percentage(var_value)}")
            llm_prompt_parts.append("IMPORTANTE: NO menciones el valor absoluto de la serie; reporta solo la variación")
            llm_prompt_parts.append("IMPORTANTE: redacta en forma directa, por ejemplo: 'La variación ... fue de X%.'")
            llm_prompt_parts.append("IMPORTANTE: NO agregues frases meta como 'no se proporciona el valor absoluto'")
            llm_prompt_parts.append(
                f"IMPORTANTE: al ser una consulta latest, indica explícitamente que corresponde al último valor reportado por la BDE y menciona el período {latest_period_label}."
            )
            if freq_raw == "a":
                if str(req_form or "").strip().lower() == "point":
                    llm_prompt_parts.append(
                        f"IMPORTANTE: para frecuencia anual en point, inicia con: 'En {display_period_label}, la variación anual con respecto al año anterior fue ...'"
                    )
                else:
                    llm_prompt_parts.append("IMPORTANTE: para frecuencia anual, inicia con: 'La variación anual con respecto al año anterior es ...'")
        else:
            llm_prompt_parts.append(f"- Valor del período: {format_value(row.get('value'))}")
            llm_prompt_parts.append(
                "IMPORTANTE: si no hay variación disponible, indícalo explícitamente y reporta igualmente el valor del período sin inventar cifras."
            )

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
    indicator_context_val: Optional[str],
    seasonality_context_val: Optional[str],
    final_indicator_name: str,
    series_title: Optional[str],
    freq: str,
    date_range_label: str,
    display_period_label: str,
    obs_to_show: List[Dict[str, Any]],
    is_contribution: bool,
    all_series_data: Optional[List[Dict[str, Any]]] = None,
    used_latest_fallback_for_point: bool = False,
    fallback_text: Optional[str] = None,
) -> Iterable[str]:
    def _resolve_indicator_phrase() -> str:
        series_title_text = str(series_title or "").strip()
        final_indicator_text = str(final_indicator_name or "").strip()

        series_title_lower = series_title_text.lower()
        series_is_generic_indicator = series_title_lower in {"", "pib", "imacec", "indicador"}

        if series_title_text and not series_is_generic_indicator:
            return series_title_text
        if final_indicator_text:
            return final_indicator_text
        if series_title_text:
            return series_title_text
        return "indicador"

    def _normalize_numeric_spacing(text: str) -> str:
        normalized = str(text or "")
        normalized = re.sub(r"[\u00A0\u202F]", " ", normalized)
        normalized = re.sub(r"([\.,])\s+(\d)", r"\1\2", normalized)
        normalized = re.sub(r"(\d)\s+%", r"\1%", normalized)
        return normalized

    def _sanitize_generated_text(text: str) -> str:
        clean_text = str(text or "").strip()
        if not clean_text:
            return ""
        clean_text = _normalize_numeric_spacing(clean_text)
        lowered = clean_text.lower()
        if any(token in lowered for token in ["none", "null", "nan"]):
            return ""
        return clean_text

    def _dedupe_point_variation_text(text: str) -> str:
        raw_sentences = [segment.strip() for segment in str(text or "").replace("\n", " ").split(".") if segment.strip()]
        if len(raw_sentences) < 2:
            return str(text or "").strip()

        def _normalize_sentence(sentence: str) -> str:
            return " ".join(sentence.lower().split())

        unique_sentences: List[str] = []
        seen: set[str] = set()
        for sentence in raw_sentences:
            normalized = _normalize_sentence(sentence)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_sentences.append(sentence)

        if len(unique_sentences) >= 2 and req_form == "point" and not is_contribution:
            first_sentence = unique_sentences[0]
            second_sentence = unique_sentences[1]
            first_norm = _normalize_sentence(first_sentence)
            second_norm = _normalize_sentence(second_sentence)

            is_first_annual = "variación anual" in first_norm and "año anterior" in first_norm
            is_second_annual = "variación anual" in second_norm and "año anterior" in second_norm

            if is_first_annual and is_second_annual:
                requested_period = str(display_period_label or "").strip().lower()
                first_has_period = bool(requested_period) and requested_period != "--" and requested_period in first_norm
                second_has_period = bool(requested_period) and requested_period != "--" and requested_period in second_norm

                kept_sentence = first_sentence
                if first_has_period != second_has_period:
                    kept_sentence = first_sentence if first_has_period else second_sentence

                unique_sentences = [kept_sentence] + unique_sentences[2:]

        if not unique_sentences:
            return str(text or "").strip()
        return ". ".join(unique_sentences).strip() + "."

    def _dedupe_latest_variation_text(text: str) -> str:
        raw_sentences = [segment.strip() for segment in str(text or "").replace("\n", " ").split(".") if segment.strip()]
        if len(raw_sentences) < 2 or req_form != "latest" or is_contribution:
            return str(text or "").strip()

        def _normalize_sentence(sentence: str) -> str:
            return " ".join(sentence.lower().split())

        unique_sentences: List[str] = []
        seen: set[str] = set()
        for sentence in raw_sentences:
            normalized = _normalize_sentence(sentence)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_sentences.append(sentence)

        if len(unique_sentences) >= 2:
            first_sentence = unique_sentences[0]
            second_sentence = unique_sentences[1]
            first_norm = _normalize_sentence(first_sentence)
            second_norm = _normalize_sentence(second_sentence)

            both_variation = "variación" in first_norm and "variación" in second_norm
            if both_variation:
                requested_period = str(display_period_label or "").strip().lower()
                first_has_period = bool(requested_period) and requested_period != "--" and requested_period in first_norm
                second_has_period = bool(requested_period) and requested_period != "--" and requested_period in second_norm
                first_has_latest = "último valor reportado" in first_norm or "ultimo valor reportado" in first_norm
                second_has_latest = "último valor reportado" in second_norm or "ultimo valor reportado" in second_norm

                kept_sentence = first_sentence
                if first_has_period != second_has_period:
                    kept_sentence = first_sentence if first_has_period else second_sentence
                if first_has_latest != second_has_latest:
                    kept_sentence = first_sentence if first_has_latest else second_sentence

                unique_sentences = [kept_sentence] + unique_sentences[2:]

        if not unique_sentences:
            return str(text or "").strip()
        return ". ".join(unique_sentences).strip() + "."

    def _build_point_single_sentence() -> Optional[str]:
        def _missing_variation_sentence(comparison: str) -> str:
            if comparison == "respecto al mismo período del año anterior":
                return (
                    "No se reporta variación respecto al mismo período del año anterior, "
                    "porque no hay dato de referencia en la serie histórica."
                )
            if comparison == "respecto al período anterior":
                return (
                    "No se reporta variación respecto al período anterior, "
                    "porque no hay dato de referencia en la serie histórica."
                )
            return "No se reporta variación, porque no hay dato de referencia en la serie histórica."

        req_norm = str(req_form or "").strip().lower()
        if req_norm not in {"point", "specific_point"} or is_contribution:
            return None
        row = obs_to_show[0] if obs_to_show else {}
        if not isinstance(row, dict):
            return None

        if row.get("yoy") is not None:
            variation_value = row.get("yoy")
            comparison_text = "respecto al mismo período del año anterior"
        elif row.get("prev_period") is not None:
            variation_value = row.get("prev_period")
            comparison_text = "respecto al período anterior"
        else:
            variation_value = None
            if "yoy" in row:
                comparison_text = "respecto al mismo período del año anterior"
            elif "prev_period" in row:
                comparison_text = "respecto al período anterior"
            else:
                comparison_text = ""

        period_label = str(display_period_label or "").strip()
        if not period_label or period_label == "--":
            period_label = "el período consultado"

        observed_period_label = format_period_labels(row.get("date"), freq)[0]
        if not observed_period_label or observed_period_label == "--":
            observed_period_label = period_label

        indicator_phrase = _resolve_indicator_phrase()
        if not indicator_phrase:
            indicator_phrase = "indicador"

        indicator_norm = str(indicator_context_val or "").strip().lower()
        seasonality_norm = str(seasonality_context_val or "").strip().lower()
        if seasonality_norm == "sa" and indicator_norm in {"pib", "imacec"}:
            if "desestacionalizado" not in indicator_phrase.lower():
                indicator_phrase = f"{indicator_phrase} desestacionalizado"

        indicator_phrase_lower = indicator_phrase.lower()
        if "pib" in indicator_phrase_lower and ("mineria" in indicator_phrase_lower or "minería" in indicator_phrase_lower):
            indicator_phrase = "el PIB minero"
            indicator_phrase_lower = indicator_phrase.lower()
        if not indicator_phrase_lower.startswith(("el ", "la ", "los ", "las ")):
            indicator_phrase = f"el {indicator_phrase}"

        indicator_label_base = _indicator_display_name(
            indicator_context_val=indicator_context_val,
            seasonality_context_val=seasonality_context_val,
            fallback=final_indicator_name,
        )
        top_variation_sentence = _build_top_variation_sentence(
            all_series_data,
            indicator_norm,
            indicator_label_base,
        )

        if used_latest_fallback_for_point and observed_period_label != period_label:
            indicator_norm = str(indicator_context_val or "").strip().lower()
            if not indicator_norm:
                final_indicator_norm = str(final_indicator_name or "").strip().lower()
                if final_indicator_norm == "pib":
                    indicator_norm = "pib"
                elif final_indicator_norm == "imacec":
                    indicator_norm = "imacec"

            indicator_label_base = _indicator_display_name(
                indicator_context_val=indicator_context_val,
                seasonality_context_val=seasonality_context_val,
                fallback=final_indicator_name,
            )
            top_variation_sentence = _build_top_variation_sentence(
                all_series_data,
                indicator_norm,
                indicator_label_base,
            )
            if variation_value is None:
                base_text = (
                    f"No hay datos disponibles para {period_label}; sin embargo, según la BDE, el último valor disponible "
                    f"corresponde a {observed_period_label}, donde {indicator_phrase} registró un valor de "
                    f"{format_value(row.get('value'))}. {_missing_variation_sentence(comparison_text)}"
                )
                return f"{base_text} {top_variation_sentence}" if top_variation_sentence else base_text
            base_text = (
                f"No hay datos disponibles para {period_label}; sin embargo, según la BDE, el último valor disponible "
                f"corresponde a {observed_period_label}, donde {indicator_phrase} presentó una variación de "
                f"{format_percentage(variation_value)} {comparison_text}."
            )
            return f"{base_text} {top_variation_sentence}" if top_variation_sentence else base_text

        if variation_value is None:
            base_text = (
                f"En {period_label}, según los datos de la BDE, {indicator_phrase} registró un valor de "
                f"{format_value(row.get('value'))}. {_missing_variation_sentence(comparison_text)}"
            )
            return f"{base_text} {top_variation_sentence}" if top_variation_sentence else base_text

        base_text = (
            f"En {period_label}, según los datos de la BDE, {indicator_phrase} presentó una variación "
            f"de {format_percentage(variation_value)} {comparison_text}."
        )
        return f"{base_text} {top_variation_sentence}" if top_variation_sentence else base_text

    def _build_range_single_sentence() -> Optional[str]:
        def _missing_variation_sentence(comparison: str) -> str:
            if comparison == "respecto al mismo período del año anterior":
                return (
                    "No se reporta variación respecto al mismo período del año anterior, "
                    "porque no hay dato de referencia en la serie histórica."
                )
            if comparison == "respecto al período anterior":
                return (
                    "No se reporta variación respecto al período anterior, "
                    "porque no hay dato de referencia en la serie histórica."
                )
            return "No se reporta variación, porque no hay dato de referencia en la serie histórica."

        req_norm = str(req_form or "").strip().lower()
        if req_norm != "range" or is_contribution:
            return None
        if not obs_to_show:
            return None

        last_row = obs_to_show[-1]
        if not isinstance(last_row, dict):
            return None

        if last_row.get("yoy") is not None:
            variation_value = last_row.get("yoy")
            comparison_text = "respecto al mismo período del año anterior"
        elif last_row.get("prev_period") is not None:
            variation_value = last_row.get("prev_period")
            comparison_text = "respecto al período anterior"
        else:
            variation_value = None
            if "yoy" in last_row:
                comparison_text = "respecto al mismo período del año anterior"
            elif "prev_period" in last_row:
                comparison_text = "respecto al período anterior"
            else:
                comparison_text = ""

        range_label = str(date_range_label or display_period_label or "").strip()
        if not range_label or range_label == "--":
            first_date = str(obs_to_show[0].get("date") or "")
            last_date = str(last_row.get("date") or "")
            if first_date and last_date:
                first_label = format_period_labels(first_date, "m")[0]
                last_label = format_period_labels(last_date, "m")[0]
                range_label = f"desde {first_label} hasta {last_label}"
            else:
                range_label = "en el período consultado"

        last_period_label = format_period_labels(last_row.get("date"), freq)[0]
        first_row = obs_to_show[0] if obs_to_show else {}
        first_period_label = format_period_labels(first_row.get("date"), freq)[0] if isinstance(first_row, dict) else ""
        indicator_phrase = str(series_title or final_indicator_name or "indicador").strip() or "indicador"
        indicator_norm = str(indicator_context_val or "").strip().lower()
        seasonality_norm = str(seasonality_context_val or "").strip().lower()
        if seasonality_norm == "sa" and indicator_norm in {"pib", "imacec"}:
            if "desestacionalizado" not in indicator_phrase.lower():
                indicator_phrase = f"{indicator_phrase} desestacionalizado"

        is_single_period_range = (
            bool(first_period_label)
            and first_period_label != "--"
            and first_period_label == last_period_label
        )

        if is_single_period_range:
            if variation_value is not None:
                return (
                    f"En {last_period_label}, según los datos de la BDE, {indicator_phrase} registró un valor de "
                    f"{format_value(last_row.get('value'))} y una variación de {format_percentage(variation_value)} {comparison_text}."
                )
            return (
                f"En {last_period_label}, según los datos de la BDE, {indicator_phrase} registró un valor de "
                f"{format_value(last_row.get('value'))}. {_missing_variation_sentence(comparison_text)}"
            )

        return (
            f"En el período {range_label}, según los datos de la BDE, {indicator_phrase} registró en "
            f"{last_period_label} un valor de {format_value(last_row.get('value'))} y "
            + (
                f"una variación de {format_percentage(variation_value)} {comparison_text}."
                if variation_value is not None
                else _missing_variation_sentence(comparison_text)
            )
        )

    def _remove_latest_wording_for_non_latest(text: str) -> str:
        if str(req_form or "").strip().lower() == "latest":
            return str(text or "").strip()

        cleaned = str(text or "")
        patterns = [
            r",?\s*que\s+corresponde\s+al\s+último\s+valor\s+reportado\s+por\s+la\s+bde",
            r",?\s*que\s+corresponde\s+al\s+ultimo\s+valor\s+reportado\s+por\s+la\s+bde",
            r",?\s*correspondiente\s+al\s+último\s+valor\s+reportado\s+por\s+la\s+bde",
            r",?\s*correspondiente\s+al\s+ultimo\s+valor\s+reportado\s+por\s+la\s+bde",
            r",?\s*de\s+acuerdo\s+al\s+último\s+valor\s+reportado\s+por\s+la\s+bde",
            r",?\s*de\s+acuerdo\s+al\s+ultimo\s+valor\s+reportado\s+por\s+la\s+bde",
            r",?\s*según\s+el\s+último\s+valor\s+reportado\s+por\s+la\s+bde",
            r",?\s*segun\s+el\s+ultimo\s+valor\s+reportado\s+por\s+la\s+bde",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        cleaned = re.sub(r"\s+,", ",", cleaned)
        cleaned = re.sub(r"\s+\.", ".", cleaned)
        return cleaned.strip()

    def _build_top_variation_sentence(
        rows: Optional[List[Dict[str, Any]]],
        indicator_key: str,
        indicator_label_base: str,
    ) -> Optional[str]:
        if not isinstance(rows, list) or not rows:
            return None

        candidates: List[Tuple[float, str]] = []
        activity_candidate_count = 0
        region_candidate_count = 0
        investment_candidate_count = 0
        for series in rows:
            if not isinstance(series, dict):
                continue

            activity_value = str(series.get("activity") or "").strip().lower()
            region_value = str(series.get("region") or "").strip().lower()
            investment_value = str(series.get("investment") or "").strip().lower()

            has_activity_dimension = bool(activity_value) and activity_value not in {"total", "imacec", "pib"}
            has_region_dimension = bool(region_value) and region_value not in {"total", "general", "none"}
            has_investment_dimension = bool(investment_value) and investment_value not in {"total", "general", "none"}

            if has_activity_dimension:
                activity_candidate_count += 1
            if has_region_dimension:
                region_candidate_count += 1
            if has_investment_dimension:
                investment_candidate_count += 1

            var_candidate = series.get("comparison_value")
            if var_candidate is None:
                var_candidate = series.get("yoy") if series.get("yoy") is not None else series.get("prev_period")
            try:
                var_numeric = float(var_candidate)
            except (TypeError, ValueError):
                continue

            series_name = str(
                series.get("title")
                or series.get("activity")
                or series.get("region")
                or series.get("investment")
                or ""
            ).strip().replace("_", " ")
            if not series_name:
                continue

            series_name_norm = series_name.lower().strip()
            if series_name_norm in {
                "total",
                "pib",
                "imacec",
                "producto interno bruto",
                "pib a costo de factores",
            }:
                continue

            candidates.append((var_numeric, series_name))

        if not candidates:
            return None

        top_var, top_name = max(candidates, key=lambda item: item[0])
        base_has_sa = "desestacionalizado" in str(indicator_label_base or "").strip().lower()
        if (
            indicator_key == "pib"
            and region_candidate_count > 0
            and activity_candidate_count == 0
            and investment_candidate_count == 0
        ):
            indicator_label = "PIB regional desestacionalizado" if base_has_sa else "PIB regional"
        elif indicator_key == "imacec":
            indicator_label = "IMACEC desestacionalizado" if base_has_sa else "IMACEC"
        elif indicator_key == "pib":
            indicator_label = "PIB desestacionalizado" if base_has_sa else "PIB"
        else:
            indicator_label = "indicador"
        return (
            f"La mayor variación en el desglose del {indicator_label} la registró {top_name}, "
            f"con {format_percentage(top_var)}."
        )

    try:
        llm = build_llm(streaming=True, temperature=llm_temperature, mode="fallback")
        chunks: List[str] = []
        streamed_chars = 0
        max_streamed_chars = 1200
        for chunk in llm.stream(llm_prompt, history=[], intent_info=None):
            text = str(chunk)
            if text:
                chunks.append(text)
                streamed_chars += len(text)
                if streamed_chars >= max_streamed_chars:
                    logger.warning(
                        "Respuesta LLM truncada por límite de caracteres | max=%s",
                        max_streamed_chars,
                    )
                    break

        generated_text = _sanitize_generated_text("".join(chunks))
        if not generated_text or generated_text.lower().startswith("(error generando)"):
            raise RuntimeError("llm_generation_failed")

        final_text = generated_text

        point_single_sentence = _build_point_single_sentence()
        if point_single_sentence:
            final_text = point_single_sentence

        range_single_sentence = _build_range_single_sentence()
        if range_single_sentence:
            final_text = range_single_sentence

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

        final_text = _dedupe_point_variation_text(final_text)
        final_text = _dedupe_latest_variation_text(final_text)
        final_text = _remove_latest_wording_for_non_latest(final_text)
        final_text = _normalize_numeric_spacing(final_text).strip()

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
                yield (
                    f"{final_indicator_name} en {display_period_label}: no hay variación disponible; "
                    f"valor del período {format_value(row.get('value'))}"
                )
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
    component_context_val: Optional[str] = None,
    is_specific_activity: bool = False,
) -> Iterable[str]:
    def _is_aggregate_indicator_row(series: Dict[str, Any]) -> bool:
        activity_value = str(series.get("activity") or "").strip().lower()
        region_value = str(series.get("region") or "").strip().lower()
        investment_value = str(series.get("investment") or "").strip().lower()
        title_value = str(series.get("title") or "").strip().lower()

        if activity_value in {"total", "imacec", "pib"}:
            return True

        has_dimension = (
            (activity_value and activity_value not in {"total", "imacec", "pib"})
            or (region_value and region_value not in {"total", "general", "none", "null"})
            or (investment_value and investment_value not in {"total", "general", "none", "null"})
        )
        if title_value in {"subtotal regionalizado", "extrarregional"}:
            return True
        if not has_dimension and title_value in {"pib", "imacec", "producto interno bruto"}:
            return True
        return False

    def _normalize_activity_key(raw_value: Any) -> str:
        return str(raw_value or "").strip().lower().replace(" ", "_")

    freq_norm = str(freq or "").strip().lower()
    use_short_annual_label = freq_norm in {"a", "annual", "anual"}

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
        highlighted_activity = None
        if is_specific_activity and component_context_val:
            highlighted_activity = _normalize_activity_key(component_context_val)

        series_by_activity = {
            str(s.get("activity") or "").strip().lower(): s
            for s in valid_series
            if str(s.get("activity") or "").strip()
        }
        has_imacec_breakdown = "total" in series_by_activity

        max_activity = None
        max_value = float("-inf")

        def _safe_numeric(value: Any) -> Optional[float]:
            if isinstance(value, bool) or value is None:
                return None
            if isinstance(value, (int, float)):
                try:
                    if math.isnan(float(value)):
                        return None
                except Exception:
                    return None
                return float(value)
            return None

        if has_imacec_breakdown:
            for s in valid_series:
                activity = str(s.get("activity") or "").strip().lower()
                value = _safe_numeric(s.get("value"))
                if activity and activity != "total" and value is not None and value > max_value:
                    max_value = value
                    max_activity = activity

            for activity_key in activity_order:
                if activity_key in series_by_activity:
                    series_info = series_by_activity[activity_key]
                    display_name = activity_display_names.get(activity_key, activity_key)
                    value = series_info.get("value", 0)

                    should_highlight = (
                        activity_key == highlighted_activity
                        if highlighted_activity
                        else activity_key == max_activity
                    )

                    if should_highlight:
                        yield f"**{display_name}** | **{format_percentage(value)}**\n"
                    else:
                        yield f"{display_name} | {format_percentage(value)}\n"
        else:
            max_title = None
            max_activity_key = None
            max_series_id = None
            for s in valid_series:
                if _is_aggregate_indicator_row(s):
                    continue
                title = str(s.get("title") or s.get("activity") or "").strip() or "Actividad"
                value = _safe_numeric(s.get("value"))
                if value is not None and value > max_value:
                    max_value = value
                    max_title = title
                    max_activity_key = _normalize_activity_key(s.get("activity"))
                    max_series_id = str(s.get("series_id") or "").strip()

            for s in valid_series:
                title = str(s.get("title") or s.get("activity") or "").strip() or "Actividad"
                value = s.get("value")
                activity_key = _normalize_activity_key(s.get("activity"))
                series_id = str(s.get("series_id") or "").strip()
                has_activity_key = bool(activity_key)
                should_highlight = (
                    activity_key == highlighted_activity
                    if highlighted_activity
                    else (
                        activity_key == max_activity_key
                        if has_activity_key
                        else (
                            series_id == max_series_id
                            if max_series_id
                            else title == max_title
                        )
                    )
                )
                if should_highlight:
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
            labels = format_period_labels(date_str, freq)
            period_label = labels[1] if use_short_annual_label else labels[0]
            yield f"{period_label} | {format_percentage(var_value)}\n"
    elif all_series_data:
        valid_series = [s for s in all_series_data if isinstance(s, dict)]
        rows_with_variation = [
            s for s in valid_series
            if s.get("comparison_value") is not None or s.get("yoy") is not None or s.get("prev_period") is not None
        ]

        dimension_key = None
        for candidate_key in ("activity", "region", "investment"):
            if any(str(row.get(candidate_key) or "").strip() for row in rows_with_variation):
                dimension_key = candidate_key
                break

        if rows_with_variation and dimension_key:
            header_label = {
                "activity": "Actividad",
                "region": "Región",
                "investment": "Componente",
            }.get(dimension_key, "Serie")
            yield f"{header_label} | Valor * | Variación\n"
            yield "----------|-------|----------\n"

            max_row = None
            max_variation = float("-inf")
            for row in rows_with_variation:
                if _is_aggregate_indicator_row(row):
                    continue
                var_value = row.get("comparison_value")
                if var_value is None:
                    var_value = row.get("yoy") if row.get("yoy") is not None else row.get("prev_period")
                try:
                    var_numeric = float(var_value)
                except (TypeError, ValueError):
                    continue
                if var_numeric > max_variation:
                    max_variation = var_numeric
                    max_row = row

            for row in rows_with_variation:
                raw_name = str(row.get("title") or row.get(dimension_key) or "Serie").strip()
                display_name = raw_name.replace("_", " ")
                value_formatted = format_value(row.get("value"))
                var_value = row.get("comparison_value")
                if var_value is None:
                    var_value = row.get("yoy") if row.get("yoy") is not None else row.get("prev_period")
                var_formatted = format_percentage(var_value)

                if max_row is row:
                    yield f"**{display_name}** | **{value_formatted}** | **{var_formatted}**\n"
                else:
                    yield f"{display_name} | {value_formatted} | {var_formatted}\n"

            yield "\n"
            yield "_Tasa de variación porcentual_\n"
        else:
            yield "Periodo | Valor * | Variación\n"
            yield "--------|-------|----------\n"
            for row in obs_to_show:
                date_str = row.get("date", "")
                value = row.get("value")
                var_value = row.get("yoy") if "yoy" in row else row.get("prev_period")
                labels = format_period_labels(date_str, freq)
                period_label = labels[1] if use_short_annual_label else labels[0]
                value_formatted = format_value(value)
                yield f"{period_label} | {value_formatted} | {format_percentage(var_value)}\n"
    else:
        yield "Periodo | Valor * | Variación\n"
        yield "--------|-------|----------\n"
        for row in obs_to_show:
            date_str = row.get("date", "")
            value = row.get("value")
            var_value = row.get("yoy") if "yoy" in row else row.get("prev_period")
            labels = format_period_labels(date_str, freq)
            period_label = labels[1] if use_short_annual_label else labels[0]
            value_formatted = format_value(value)
            yield f"{period_label} | {value_formatted} | {format_percentage(var_value)}\n"
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
            yield r"\* _Miles de millones de pesos encadenados_" + "\n\n"

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
