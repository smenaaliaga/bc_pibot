"""Nodo DATA del grafo PIBot — orquesta la obtención de datos económicos.

Flujo principal (``data_node``):
    1. Extrae entidades y clasificación del estado del grafo.
    2. Aplica las reglas de negocio (``_business_rules``).
    3. Busca el payload en data_store con ``search_output_payloads``.
    4. Envía question + observations (payload completo) al response.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.types import StreamWriter

from orchestrator.data.response import handle_no_series, stream_data_response
from ..state import AgentState, _clone_entities, _emit_stream_chunk
from orchestrator.data._helpers import coerce_period, first_non_empty
from orchestrator.data._business_rules import ResolvedEntities, apply_business_rules
from orchestrator.data.catalog_data_search import search_output_payloads
from orchestrator.normalizer.routing_utils import INTENT_CONFIDENCE_THRESHOLD
from orchestrator.utils.component_normalizer import normalize_component

logger = logging.getLogger(__name__)

# Re-exportar para compatibilidad con tests que hacen monkeypatch.
from orchestrator.data._helpers import build_target_series_url as _build_target_series_url  # noqa: F401

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_DATA_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "memory" / "data_store"


# ---------------------------------------------------------------------------
# Extracción de entidades desde el estado del grafo
# ---------------------------------------------------------------------------

def _extract_entities_from_state(
    state: AgentState,
) -> tuple[
    str,                         # question
    List[Dict[str, Any]],       # entities
    ResolvedEntities,            # ent (mutable, pre-reglas)
]:
    """Extrae y normaliza las entidades desde el estado del grafo."""

    def _unwrap_first_value(value: Any) -> Any:
        current = value
        while isinstance(current, list):
            current = first_non_empty(current)
        if current in (None, "", [], {}, ()):  # pragma: no branch
            return None
        return current

    def _get_activity_from_dict(source: Any) -> Optional[str]:
        if not isinstance(source, dict):
            return None
        raw = _unwrap_first_value(source.get("activity"))
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    def _coerce_class_label(value: Any, *, apply_threshold: bool = True) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, dict):
            if apply_threshold:
                conf_raw = value.get("confidence")
                if conf_raw is not None:
                    try:
                        if float(conf_raw) < INTENT_CONFIDENCE_THRESHOLD:
                            return "none"
                    except (TypeError, ValueError):
                        pass
            lbl = value.get("label")
            return str(lbl).strip().lower() if lbl is not None else None
        text = str(value).strip().lower()
        return text or None

    question = state.get("question", "")
    entities_state = _clone_entities(state.get("entities"))

    classification = state.get("classification")
    predict_raw = getattr(classification, "predict_raw", None) if classification else None
    predict_raw = predict_raw if isinstance(predict_raw, dict) else {}

    calc_mode_cls = _coerce_class_label(getattr(classification, "calc_mode", None), apply_threshold=False)
    activity_cls = _coerce_class_label(getattr(classification, "activity", None), apply_threshold=True)
    region_cls = _coerce_class_label(getattr(classification, "region", None), apply_threshold=True)
    investment_cls = _coerce_class_label(getattr(classification, "investment", None), apply_threshold=True)
    req_form_cls = _coerce_class_label(getattr(classification, "req_form", None), apply_threshold=False)

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

    normalized_source = (
        "predict_raw.interpretation.entities_normalized"
        if normalized_from_predict
        else "classification.normalized"
    )
    logger.info("[DATA_NODE] normalized_entities source=%s data=%s", normalized_source, normalized)

    # Extraer valores de entidades
    period_ent = coerce_period(normalized.get("period"))

    ent = ResolvedEntities(
        indicator_ent=first_non_empty(normalized.get("indicator")),
        seasonality_ent=first_non_empty(normalized.get("seasonality")),
        frequency_ent=first_non_empty(normalized.get("frequency")),
        activity_ent=first_non_empty(normalized.get("activity")),
        region_ent=first_non_empty(normalized.get("region")),
        investment_ent=first_non_empty(normalized.get("investment")),
        price_ent=first_non_empty(normalized.get("price")),
        period_ent=period_ent,
        calc_mode_cls=calc_mode_cls,
        activity_cls=activity_cls,
        activity_cls_resolved=activity_cls,
        region_cls=region_cls,
        investment_cls=investment_cls,
        req_form_cls=req_form_cls,
    )

    # Fallback conservador: si activity_cls es specific y no hubo actividad
    # normalizada, intentar derivarla desde entidades crudas.
    if str(activity_cls or "").strip().lower() == "specific" and not ent.activity_ent:
        raw_activity = (
            _get_activity_from_dict(interpretation_root.get("entities"))
            or _get_activity_from_dict(classification_entities)
        )
        if raw_activity is None:
            for item in entities_state:
                raw_activity = _get_activity_from_dict(item)
                if raw_activity:
                    break

        if raw_activity:
            ent.activity_ent = normalize_component(raw_activity)
            logger.info(
                "[DATA_NODE] activity fallback from raw entities: raw=%s normalized=%s",
                raw_activity,
                ent.activity_ent,
            )

    logger.info("[DATA_NODE] resolved entities (pre-rules)=%s", asdict(ent))

    return question, entities, ent


# ---------------------------------------------------------------------------
# Helpers para búsqueda en data_store
# ---------------------------------------------------------------------------

def _build_search_kwargs(ent: ResolvedEntities) -> Dict[str, Any]:
    """Construye los kwargs para ``search_output_payloads`` a partir de las entidades resueltas."""
    kwargs: Dict[str, Any] = {}

    if ent.indicator_ent:
        kwargs["indicator"] = ent.indicator_ent

    calc = str(ent.calc_mode_cls or "").strip().lower()
    if calc:
        kwargs["calc_mode"] = calc

    if ent.seasonality_ent:
        kwargs["seasonality"] = ent.seasonality_ent

    if ent.frequency_ent:
        kwargs["frequency"] = ent.frequency_ent

    if ent.price:
        kwargs["price"] = ent.price

    # has_activity / has_region / has_investment: 1 si cls es "general" o "specific", 0 si "none"
    act = str(ent.activity_cls or "").strip().lower()
    if act in ("general", "specific"):
        kwargs["has_activity"] = 1
    elif act == "none":
        kwargs["has_activity"] = 0

    reg = str(ent.region_cls or "").strip().lower()
    if reg in ("general", "specific"):
        kwargs["has_region"] = 1
    elif reg == "none":
        kwargs["has_region"] = 0

    if ent.region_ent:
        kwargs["region"] = ent.region_ent

    inv = str(ent.investment_cls or "").strip().lower()
    if inv in ("general", "specific"):
        kwargs["has_investment"] = 1
    elif inv == "none":
        kwargs["has_investment"] = 0

    if ent.hist is not None:
        kwargs["hist"] = ent.hist

    return kwargs


def _filter_series_by_entities(
    observations: Dict[str, Any],
    ent: ResolvedEntities,
) -> Dict[str, Any]:
    """Filtra series por restricciones resueltas (price/seasonality/calc_mode) cuando existan.

    Esto evita que el LLM seleccione una serie incompatible cuando un payload
    contiene múltiples variantes (por ejemplo, enc y co).
    """
    series = observations.get("series") or []
    if not isinstance(series, list) or not series:
        return observations

    constraints: Dict[str, str] = {}
    price = str(ent.price or "").strip().lower()
    seasonality = str(ent.seasonality_ent or "").strip().lower()
    calc_mode = str(ent.calc_mode_cls or "").strip().lower()
    if price:
        constraints["price"] = price
    if seasonality:
        constraints["seasonality"] = seasonality
    if calc_mode:
        constraints["calc_mode"] = calc_mode

    if not constraints:
        return observations

    def _matches(series_item: Dict[str, Any]) -> bool:
        cls = series_item.get("classification_series") or {}
        if not isinstance(cls, dict):
            return True
        for key, expected in constraints.items():
            current = cls.get(key)
            # Regla estricta: si se pidió estacionalidad, la serie debe declararla
            # explícitamente para evitar mezclar SA/NsA por omisión de metadata.
            if key == "seasonality" and current is None:
                return False
            if current is None:
                continue
            if isinstance(current, list):
                normalized = {str(v).strip().lower() for v in current}
                if expected not in normalized:
                    return False
                continue
            if str(current).strip().lower() != expected:
                return False
        return True

    filtered = [s for s in series if _matches(s)]
    if not filtered:
        logger.info(
            "[DATA_NODE] series filter skipped (0 matches) constraints=%s total=%d",
            constraints,
            len(series),
        )
        return observations

    if len(filtered) == len(series):
        return observations

    cloned = dict(observations)
    cloned["series"] = filtered
    logger.info(
        "[DATA_NODE] series filtered by constraints=%s kept=%d total=%d",
        constraints,
        len(filtered),
        len(series),
    )
    return cloned


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
        # 1. Extraer entidades
        question, entities, ent = _extract_entities_from_state(state)

        # 2. Aplicar reglas de negocio
        apply_business_rules(ent)
        logger.info("[DATA_NODE] resolved entities (post-rules)=%s", asdict(ent))
        logger.info("[DATA_NODE] indicator=%s freq=%s activity=%s req_form=%s",
                    ent.indicator_ent, ent.frequency_ent, ent.activity_ent, ent.req_form_cls)

        # 3. Buscar payload en data_store con search_output_payloads
        search_kwargs = _build_search_kwargs(ent)
        logger.info("[DATA_NODE] search_output_payloads kwargs=%s", search_kwargs)

        try:
            matches = search_output_payloads(
                str(_DATA_STORE_DIR), **search_kwargs
            )
        except FileNotFoundError:
            logger.warning("[DATA_NODE] data_store directory not found: %s", _DATA_STORE_DIR)
            matches = []

        logger.info("[DATA_NODE] search_output_payloads found %d matches", len(matches))

        if not matches:
            return handle_no_series(
                question=question,
                entities=entities,
                ent=ent,
                writer=writer,
                emit_fn=_emit_stream_chunk,
                first_non_empty_fn=first_non_empty,
            )

        # El payload completo del data_store ES las observations
        observations = matches[0]["payload"]
        observations = _filter_series_by_entities(observations, ent)
        source_url = observations.get("source_url", "")

        logger.info("[DATA_NODE] cuadro=%s freq=%s series_count=%d",
                    observations.get("cuadro_name"),
                    observations.get("frequency"),
                    len(observations.get("series", [])))

        ent_dict = asdict(ent)

        # 4. Construir payload: question + observations (payload data_store)
        payload = {
            "question": question,
            "observations": observations,
            "entities": ent_dict,
        }

        collected: List[str] = []
        try:
            for chunk in stream_data_response(payload):
                if chunk:
                    collected.append(chunk)
                    _emit_stream_chunk(chunk, writer)
        except Exception:
            logger.exception("[DATA_NODE] Error en streaming de respuesta LLM")
            if not collected:
                fallback = "Ocurrió un problema al generar la respuesta."
                collected.append(fallback)
                _emit_stream_chunk(fallback, writer)

        pv = ent.period_ent or []
        rf = str(ent.req_form_cls or "").strip().lower()

        # Extraer serie objetivo del payload para el retorno
        store_series = observations.get("series") or []
        target_id = store_series[0].get("series_id") if store_series else None

        return {
            "output": "".join(collected),
            "entities": entities,
            "parsed_point": str(pv[-1]) if (rf != "range" and pv) else None,
            "parsed_range": (str(pv[0]), str(pv[-1])) if pv else None,
            "series": target_id,
            "data_classification": ent_dict,
        }

    return data_node


__all__ = ["make_data_node"]
