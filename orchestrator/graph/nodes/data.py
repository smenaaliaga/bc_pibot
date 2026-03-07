"""Nodo DATA del grafo PIBot — orquesta la obtención de datos económicos.

Flujo principal (``data_node``):
    1. Extrae entidades y clasificación del estado del grafo.
    2. Aplica las reglas de negocio (``_business_rules``).
    3. Busca la familia y serie objetivo en el catálogo (``catalog_lookup``).
    4. Obtiene las observaciones según la rama correspondiente (``_fetch``).
    5. Construye el payload y lo envía al flujo de respuesta (streaming).

Módulos internos:
    - ``_helpers``: utilidades puras (parsing, coerción, URL).
    - ``_business_rules``: reglas de dominio sobre entidades.
    - ``catalog_lookup``: búsqueda de familia y serie en catálogo.
    - ``_fetch``: obtención de datos desde Redis (3 ramas).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.types import StreamWriter

from orchestrator.data import flow_data
from orchestrator.data.response import build_no_series_message

from ..state import AgentState, _clone_entities, _emit_stream_chunk

from orchestrator.data._helpers import (
    build_target_series_url,
    coerce_period,
    extract_year,
    first_non_empty,
)
from orchestrator.data._business_rules import ResolvedEntities, apply_business_rules
from orchestrator.catalog.catalog_lookup import lookup_series
from orchestrator.data._fetch import (
    fetch_contribution,
    fetch_general_breakdown,
    fetch_specific_series,
)

logger = logging.getLogger(__name__)

# Re-exportar funciones de módulos internos con los nombres originales
# para mantener compatibilidad con el test que hace monkeypatch de estos símbolos.
from orchestrator.data._helpers import (
    first_non_empty as _first_non_empty,
    coerce_period as _coerce_period,
    extract_year as _extract_year,
    build_target_series_url as _build_target_series_url,
    parse_iso_date as _parse_iso_date,
    quarter_from_date as _quarter_from_date,
    sort_observations_by_date_desc as _sort_observations_by_date_desc,
    same_requested_period as _same_requested_period,
    has_full_quarterly_year as _has_full_quarterly_year,
    latest_annual_observation_before_year as _latest_annual_observation_before_year,
)
from orchestrator.data._fetch import (
    load_series_observations as _load_series_observations,
    fetch_series_by_req_form as _fetch_series_by_req_form,
)


def _get_fetch_fn():
    """Retorna la referencia actual de _fetch_series_by_req_form.

    Esto permite que los tests hagan monkeypatch en data_module._fetch_series_by_req_form
    y la referencia parcheada se propague a las funciones de _fetch.py vía fetch_fn.
    """
    import sys
    mod = sys.modules[__name__]
    return getattr(mod, "_fetch_series_by_req_form")


def _get_load_fn():
    """Retorna la referencia actual de _load_series_observations."""
    import sys
    mod = sys.modules[__name__]
    return getattr(mod, "_load_series_observations")


# ---------------------------------------------------------------------------
# Extracción de entidades desde el estado del grafo
# ---------------------------------------------------------------------------

def _extract_entities_from_state(
    state: AgentState,
) -> tuple[
    str,                         # question
    str,                         # session_id
    List[Dict[str, Any]],       # entities
    ResolvedEntities,            # ent (mutable, pre-reglas)
]:
    """Extrae y normaliza las entidades desde el estado del grafo.

    Resuelve la prioridad entre ``predict_raw.entities_normalized``
    (follow-up resuelto) y ``classification.normalized`` (clasificación base).
    """
    question = state.get("question", "")
    session_id = state.get("session_id", "")
    entities_state = _clone_entities(state.get("entities"))

    classification = state.get("classification")
    predict_raw = getattr(classification, "predict_raw", None) if classification else None
    predict_raw = predict_raw if isinstance(predict_raw, dict) else {}

    calc_mode_cls = getattr(classification, "calc_mode", None) or {}
    activity_cls = getattr(classification, "activity", None) or {}
    region_cls = getattr(classification, "region", None) or {}
    investment_cls = getattr(classification, "investment", None) or {}
    req_form_cls = getattr(classification, "req_form", None) or {}

    classification_entities = getattr(classification, "entities", None) or {}
    normalized_from_classification = getattr(classification, "normalized", None) or {}

    entities: List[Dict[str, Any]]
    if isinstance(classification_entities, dict):
        entities = [dict(classification_entities)]
    else:
        entities = entities_state

    # Priorizar entities_normalized del predict_raw (follow-up)
    interpretation_root = predict_raw.get("interpretation")
    if not isinstance(interpretation_root, dict):
        interpretation_root = predict_raw
    interpretation_root = interpretation_root if isinstance(interpretation_root, dict) else {}

    normalized_from_predict = interpretation_root.get("entities_normalized")
    normalized_from_predict = (
        normalized_from_predict if isinstance(normalized_from_predict, dict) else {}
    )

    normalized = normalized_from_predict or (
        normalized_from_classification
        if isinstance(normalized_from_classification, dict)
        else {}
    )

    # Extraer valores de entidades
    period_ent = coerce_period(normalized.get("period"))
    period_values = period_ent if isinstance(period_ent, list) else []

    ent = ResolvedEntities(
        indicator_ent=first_non_empty(normalized.get("indicator")),
        seasonality_ent=first_non_empty(normalized.get("seasonality")),
        frequency_ent=first_non_empty(normalized.get("frequency")),
        activity_ent=first_non_empty(normalized.get("activity")),
        region_ent=first_non_empty(normalized.get("region")),
        investment_ent=first_non_empty(normalized.get("investment")),
        period_ent=period_ent,
        period_values=period_values,
        period_reference_year=extract_year(period_values[0]) if period_values else None,
        calc_mode_cls=calc_mode_cls,
        activity_cls=activity_cls,
        activity_cls_resolved=activity_cls,
        region_cls=region_cls,
        investment_cls=investment_cls,
        req_form_cls=req_form_cls,
    )

    return question, session_id, entities, ent


# ---------------------------------------------------------------------------
# Logging de diagnóstico
# ---------------------------------------------------------------------------

def _log_resolved_entities(ent: ResolvedEntities, entities: List[Dict[str, Any]], normalized: dict) -> None:
    """Registra el estado de las entidades resueltas para diagnóstico."""
    logger.info("[DATA_NODE] =========================================================")
    logger.info("[DATA_NODE] calc_mode=%s", ent.calc_mode_cls)
    logger.info("[DATA_NODE] activity=%s", ent.activity_cls_resolved)
    logger.info("[DATA_NODE] region=%s", ent.region_cls)
    logger.info("[DATA_NODE] investment=%s", ent.investment_cls)
    logger.info("[DATA_NODE] req_form=%s", ent.req_form_cls)
    logger.info("[DATA_NODE] entities=%s", entities)
    logger.info("[DATA_NODE] normalized=%s", normalized)
    logger.info("[DATA_NODE] indicator=%s", ent.indicator_ent)
    logger.info("[DATA_NODE] seasonality=%s", ent.seasonality_ent)
    logger.info("[DATA_NODE] frequency=%s", ent.frequency_ent)
    logger.info("[DATA_NODE] activity=%s", ent.activity_ent)
    logger.info("[DATA_NODE] region=%s", ent.region_ent)
    logger.info("[DATA_NODE] investment=%s", ent.investment_ent)
    logger.info("[DATA_NODE] price=%s", ent.price)
    logger.info("[DATA_NODE] hist=%s", ent.hist)
    logger.info("[DATA_NODE] period=%s", ent.period_ent)
    logger.info("[DATA_NODE] =========================================================")


def _preview_observations(rows: Any, *, max_items: int = 2) -> List[Dict[str, Any]]:
    """Entrega una vista compacta de observaciones para logging didactico."""
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return []

    preview: List[Dict[str, Any]] = []
    keep_keys = {
        "date",
        "value",
        "yoy",
        "prev_period",
        "title",
        "activity",
        "region",
        "investment",
    }
    for row in rows[:max_items]:
        if isinstance(row, dict):
            preview.append({k: row.get(k) for k in row.keys() if k in keep_keys})
        else:
            preview.append({"raw": str(row)})
    return preview


def _log_payload_snapshot(payload: Dict[str, Any]) -> None:
    """Log didactico del payload para inspeccion previa al streaming."""
    result_rows = payload.get("result")
    all_series_rows = payload.get("all_series_data")
    classification = payload.get("classification")

    result_count = len(result_rows) if isinstance(result_rows, list) else (1 if result_rows else 0)
    all_series_count = len(all_series_rows) if isinstance(all_series_rows, list) else 0

    summary = {
        "intent": payload.get("intent"),
        "series": payload.get("series"),
        "series_title": payload.get("series_title"),
        "parsed_point": payload.get("parsed_point"),
        "parsed_range": payload.get("parsed_range"),
        "reference_period": payload.get("reference_period"),
        "used_latest_fallback_for_point": payload.get("used_latest_fallback_for_point"),
        "result_count": result_count,
        "all_series_count": all_series_count,
        "source_url": payload.get("source_url"),
    }

    logger.info("[DATA_NODE][PAYLOAD] resumen=%s", summary)
    logger.info("[DATA_NODE][PAYLOAD] classification=%s", classification)
    if result_count:
        logger.info(
            "[DATA_NODE][PAYLOAD] result_preview=%s",
            _preview_observations(result_rows, max_items=2),
        )
    if all_series_count:
        logger.info(
            "[DATA_NODE][PAYLOAD] all_series_preview=%s",
            _preview_observations(all_series_rows, max_items=2),
        )


# ---------------------------------------------------------------------------
# Respuesta cuando no se encuentra serie
# ---------------------------------------------------------------------------

def _handle_no_series(
    *,
    question: str,
    entities: List[Dict[str, Any]],
    ent: ResolvedEntities,
    writer: Optional[StreamWriter],
) -> Dict[str, Any]:
    """Construye la respuesta cuando no se encontró familia o serie en catálogo."""
    primary_entity = entities[0] if isinstance(entities, list) and entities else {}
    requested_activity = None
    if isinstance(primary_entity, dict):
        requested_activity = first_non_empty(primary_entity.get("activity"))
        if requested_activity is not None:
            requested_activity = str(requested_activity).strip()

    if requested_activity in {None, "", "[]", "none", "null"}:
        requested_activity = None

    indicator_candidate = ent.indicator_ent
    if indicator_candidate in (None, "", [], {}, ()):
        indicator_candidate = (
            first_non_empty(primary_entity.get("indicator"))
            if isinstance(primary_entity, dict)
            else None
        )
    indicator_label = str(indicator_candidate or "").strip().upper()
    if indicator_label in {"", "[]", "NONE", "NULL"}:
        indicator_label = None

    normalized_activity = str(ent.activity_ent or "").strip()
    text = build_no_series_message(
        question=question,
        requested_activity=requested_activity,
        normalized_activity=normalized_activity,
        indicator_label=indicator_label,
    )

    logger.warning("[DATA_NODE] %s", text)
    _emit_stream_chunk(text, writer)
    return {
        "output": text,
        "entities": entities,
        "parsed_point": None,
        "parsed_range": None,
        "series": None,
        "data_classification": _build_classification_dict(ent),
    }


# ---------------------------------------------------------------------------
# Construcción del payload y streaming de la respuesta
# ---------------------------------------------------------------------------

def _build_classification_dict(ent: ResolvedEntities, *, req_form_override: Optional[str] = None) -> Dict[str, Any]:
    """Construye el diccionario de clasificación para el payload de respuesta."""
    return {
        "indicator": ent.indicator_ent,
        "seasonality": ent.seasonality_ent,
        "frequency": ent.frequency_ent,
        "period": ent.period_ent,
        "calc_mode_cls": ent.calc_mode_cls,
        "activity_cls": ent.activity_cls_resolved,
        "region_cls": ent.region_cls,
        "investment_cls": ent.investment_cls,
        "req_form_cls": req_form_override or ent.req_form_cls,
        "activity_value": ent.activity_ent,
        "region_value": ent.region_ent,
        "investment_value": ent.investment_ent,
    }


def _stream_response(
    *,
    payload: Dict[str, Any],
    session_id: str,
    entities: List[Dict[str, Any]],
    monthly_frequency_note: Optional[str],
    incomplete_frequency_note: Optional[str],
    writer: Optional[StreamWriter],
) -> Dict[str, Any]:
    """Envía el payload al flujo de datos y emite chunks de streaming."""
    collected: List[str] = []
    try:
        if monthly_frequency_note:
            collected.append(f"{monthly_frequency_note} ")
            _emit_stream_chunk(f"{monthly_frequency_note} ", writer)
        if incomplete_frequency_note:
            collected.append(f"{incomplete_frequency_note} ")
            _emit_stream_chunk(f"{incomplete_frequency_note} ", writer)
        stream = flow_data.stream_data_flow(payload, session_id=session_id)
        for chunk in stream:
            chunk_text = str(chunk)
            if not chunk_text:
                continue
            collected.append(chunk_text)
            _emit_stream_chunk(chunk_text, writer)
    except Exception:
        logger.exception("[DATA_NODE] Flujo fallido")
        if not collected:
            fallback = "Ocurrió un problema al obtener los datos solicitados."
            collected.append(fallback)
            _emit_stream_chunk(fallback, writer)

    return {
        "output": "".join(collected),
        "entities": entities,
        "parsed_point": payload.get("parsed_point"),
        "parsed_range": payload.get("parsed_range"),
        "series": payload.get("series"),
        "data_classification": payload.get("classification"),
    }


# ---------------------------------------------------------------------------
# Nodo principal
# ---------------------------------------------------------------------------

def make_data_node(memory_adapter: Any):
    """Fábrica del nodo DATA. Retorna la función ``data_node`` que se registra
    en el grafo de LangGraph.

    Args:
        memory_adapter: adaptador de memoria (reservado para uso futuro).
    """
    def data_node(state: AgentState, *, writer: Optional[StreamWriter] = None):
        """Nodo principal de obtención de datos económicos.

        Etapas:
          1. Extraer entidades y clasificación del estado.
          2. Aplicar reglas de negocio.
          3. Buscar serie en catálogo.
          4. Si no existe serie → respuesta informativa.
          5. Obtener datos según la rama correspondiente.
          6. Construir payload y hacer streaming de la respuesta.
        """
        # 1. Extraer entidades
        question, session_id, entities, ent = _extract_entities_from_state(state)

        # 2. Aplicar reglas de negocio
        apply_business_rules(ent)

        # Log de diagnóstico
        classification = state.get("classification")
        predict_raw = getattr(classification, "predict_raw", None) if classification else None
        predict_raw = predict_raw if isinstance(predict_raw, dict) else {}
        interpretation_root = predict_raw.get("interpretation")
        if not isinstance(interpretation_root, dict):
            interpretation_root = predict_raw
        interpretation_root = interpretation_root if isinstance(interpretation_root, dict) else {}
        normalized_from_predict = interpretation_root.get("entities_normalized")
        normalized_from_predict = (
            normalized_from_predict if isinstance(normalized_from_predict, dict) else {}
        )
        normalized_from_classification = getattr(classification, "normalized", None) or {}
        normalized = normalized_from_predict or (
            normalized_from_classification
            if isinstance(normalized_from_classification, dict) else {}
        )
        _log_resolved_entities(ent, entities, normalized)

        # 3. Buscar serie en catálogo
        sl = lookup_series(ent)

        # 4. Sin serie → respuesta informativa
        if not sl.source_url or not sl.target_series_id:
            return _handle_no_series(
                question=question,
                entities=entities,
                ent=ent,
                writer=writer,
            )

        # 5. Obtener datos según la rama
        firstdate = str(ent.period_values[0]) if ent.period_values else None
        lastdate = str(ent.period_values[-1]) if ent.period_values else None

        calc_mode_norm = str(ent.calc_mode_cls or "").strip().lower()
        is_non_contribution_general_breakdown = (
            calc_mode_norm != "contribution"
            and any(
                str(slot or "").strip().lower() == "general"
                for slot in (ent.activity_cls, ent.region_cls, ent.investment_cls)
            )
        )

        if ent.calc_mode_cls == "contribution":
            fr = fetch_contribution(
                fetch_fn=_get_fetch_fn(),
                family_series=sl.family_series,
                target_series_id=sl.target_series_id,
                target_series_title=sl.target_series_title,
                source_family_series=sl.source_url,
                family_name=sl.family_name,
                req_form_cls=ent.req_form_cls,
                frequency_ent=ent.frequency_ent,
                indicator_ent=ent.indicator_ent,
                period_ent=ent.period_ent,
                period_values=ent.period_values,
                calc_mode_cls=ent.calc_mode_cls,
                activity_cls_resolved=ent.activity_cls_resolved,
                activity_ent=ent.activity_ent,
                region_cls=ent.region_cls,
                investment_cls=ent.investment_cls,
            )
        elif is_non_contribution_general_breakdown:
            fr = fetch_general_breakdown(
                fetch_fn=_get_fetch_fn(),
                family_series=sl.family_series,
                target_series_id=sl.target_series_id,
                target_series_title=sl.target_series_title,
                source_family_series=sl.source_url,
                family_name=sl.family_name,
                req_form_cls=ent.req_form_cls,
                frequency_ent=ent.frequency_ent,
                indicator_ent=ent.indicator_ent,
                period_values=ent.period_values,
                calc_mode_cls=ent.calc_mode_cls,
                activity_cls=ent.activity_cls,
                region_cls=ent.region_cls,
                investment_cls=ent.investment_cls,
            )
        else:
            fr = fetch_specific_series(
                fetch_fn=_get_fetch_fn(),
                load_fn=_get_load_fn(),
                target_series_id=sl.target_series_id,
                req_form_cls=ent.req_form_cls,
                frequency_ent=ent.frequency_ent,
                indicator_ent=ent.indicator_ent,
                period_values=ent.period_values,
                period_ent=ent.period_ent,
                calc_mode_cls=ent.calc_mode_cls,
            )

        # Actualizar periodo si la rama lo modificó (ej. completitud anual)
        if fr.period_ent is not None:
            ent.period_ent = fr.period_ent
        if fr.period_values is not None:
            ent.period_values = fr.period_values

        # Usar titles/IDs actualizados por las ramas de fetch
        target_series_id = fr.target_series_id or sl.target_series_id
        target_series_title = fr.target_series_title or sl.target_series_title
        observations = fr.observations
        observations_all = fr.observations_all
        req_form_for_payload = fr.req_form_for_payload or ent.req_form_cls

        # Verificar out_of_range para point/range sin datos
        req_form_norm = str(ent.req_form_cls or "").strip().lower()
        out_of_range = fr.out_of_range_lt_first
        if isinstance(fr.observations_meta, dict):
            if str(fr.observations_meta.get("lastdate_position") or "").strip().lower() == "lt_first":
                out_of_range = True

        if out_of_range and req_form_norm in {"point", "specific_point", "range"}:
            if not observations and isinstance(fr.latest_obs, dict):
                observations = [fr.latest_obs]
                fr.used_latest_fallback_for_point = True
                req_form_for_payload = "point"

        # 5b. Sin datos → respuesta vacía
        if not observations and not observations_all:
            text = "No hay datos disponibles para la serie solicitada."
            logger.warning("[DATA_NODE] %s", text)
            _emit_stream_chunk(text, writer)
            return {
                "output": text,
                "entities": entities,
                "parsed_point": None,
                "parsed_range": (
                    (str(ent.period_values[0]), str(ent.period_values[-1]))
                    if ent.period_values else None
                ),
                "series": target_series_id,
                "data_classification": _build_classification_dict(
                    ent, req_form_override=req_form_for_payload
                ),
            }

        # 6. Construir payload y hacer streaming
        period_values = ent.period_values
        target_series_url = build_target_series_url(
            source_url=sl.source_url,
            series_id=target_series_id,
            period=ent.period_ent if isinstance(ent.period_ent, list) else None,
            req_form=ent.req_form_cls,
            observations=observations or observations_all,
            frequency=ent.frequency_ent,
            calc_mode=ent.calc_mode_cls,
        )

        effective_reference_period = fr.effective_reference_period

        payload = {
            "intent": "value",
            "classification": {
                **_build_classification_dict(ent, req_form_override=req_form_for_payload),
                "macro_cls": "1",
                "intent_cls": "value",
                "context_cls": "followup",
                "enable": None,
                "enable_all": None,
                "sub_activity_value": ent.activity_ent,
                "gasto_value": ent.investment_ent,
                "price": None,
                "history": ent.hist,
            },
            "series": target_series_id,
            "series_title": target_series_title or sl.family_name or None,
            "parsed_point": (
                str(period_values[-1])
                if (req_form_for_payload != "range" and ent.period_ent and period_values)
                else None
            ),
            "parsed_range": (
                (str(period_values[0]), str(period_values[-1]))
                if ent.period_ent else None if period_values else None
            ),
            "reference_period": effective_reference_period,
            "used_latest_fallback_for_point": fr.used_latest_fallback_for_point,
            "result": observations,
            "all_series_data": observations_all or None,
            "source_url": target_series_url,
            "question": question,
        }

        _log_payload_snapshot(payload)

        return _stream_response(
            payload=payload,
            session_id=session_id,
            entities=entities,
            monthly_frequency_note=ent.monthly_frequency_note,
            incomplete_frequency_note=fr.incomplete_frequency_note,
            writer=writer,
        )

    return data_node


__all__ = ["make_data_node"]
