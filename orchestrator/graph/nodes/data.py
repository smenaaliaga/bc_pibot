"""Nodo DATA del grafo PIBot.

Extrae entidades, busca el payload en data_store y genera la respuesta.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.types import StreamWriter

from orchestrator.data.response import handle_no_series, stream_data_response
from ..state import AgentState, _clone_entities, _emit_stream_chunk
from orchestrator.data._helpers import first_non_empty
from orchestrator.catalog.catalog_data_search import search_output_payloads
from orchestrator.normalizer.normalizer import ResolvedEntities, resolve_entities_for_data_query
from orchestrator.normalizer.routing_utils import INTENT_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# Re-exportar para compatibilidad con tests que hacen monkeypatch.
from orchestrator.data._helpers import build_target_series_url as _build_target_series_url  # noqa: F401

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_DATA_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "memory" / "data_store"


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


def _extract_normalized_entities(classification: Any) -> Dict[str, Any]:
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
    normalized_from_classification = (
        normalized_from_classification if isinstance(normalized_from_classification, dict) else {}
    )

    normalized = normalized_from_predict or normalized_from_classification
    normalized_source = (
        "predict_raw.interpretation.entities_normalized"
        if normalized_from_predict
        else "classification.normalized"
    )
    logger.info("[DATA_NODE] normalized_entities source=%s data=%s", normalized_source, normalized)
    return normalized


# ---------------------------------------------------------------------------
# Extracción de entidades desde el estado del grafo
# ---------------------------------------------------------------------------

def _extract_entities_from_state(
    state: AgentState,
) -> tuple[
    str,                         # question
    List[Dict[str, Any]],       # entities
    ResolvedEntities,            # ent final para búsqueda
]:
    """Extrae y normaliza las entidades desde el estado del grafo."""

    question = state.get("question", "")
    entities_state = _clone_entities(state.get("entities"))

    classification = state.get("classification")

    calc_mode_cls = _coerce_class_label(getattr(classification, "calc_mode", None), apply_threshold=False)
    activity_cls = _coerce_class_label(getattr(classification, "activity", None), apply_threshold=True)
    region_cls = _coerce_class_label(getattr(classification, "region", None), apply_threshold=True)
    investment_cls = _coerce_class_label(getattr(classification, "investment", None), apply_threshold=True)
    req_form_cls = _coerce_class_label(getattr(classification, "req_form", None), apply_threshold=False)

    classification_entities = getattr(classification, "entities", None) or {}

    entities: List[Dict[str, Any]]
    if isinstance(classification_entities, dict):
        entities = [dict(classification_entities)]
    else:
        entities = entities_state

    normalized = _extract_normalized_entities(classification)

    ent = resolve_entities_for_data_query(
        normalized_entities=normalized,
        calc_mode_cls=calc_mode_cls,
        activity_cls=activity_cls,
        region_cls=region_cls,
        investment_cls=investment_cls,
        req_form_cls=req_form_cls,
    )

    logger.info("[DATA_NODE] resolved entities=%s", asdict(ent))

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

    # Flags has_*: 1 para general/specific, 0 para none.
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
    """Filtra series por price/seasonality/calc_mode."""
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
            # Si se pidió seasonality, exigir metadata explícita.
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


def _collect_target_series_ids(
    observations: Dict[str, Any],
    ent: ResolvedEntities,
) -> List[str]:
    """Retorna los ``series_id`` que cumplen activity/region/investment (AND).

    Si no hay filtros, retorna todos los IDs. Para region usa fallback al
    nivel de cuadro cuando la serie no trae ese campo.
    """
    series = observations.get("series") or []
    if not isinstance(series, list) or not series:
        return []

    activity = str(ent.activity_ent or "").strip().lower() or None
    region = str(ent.region_ent or "").strip().lower() or None
    investment = str(ent.investment_ent or "").strip().lower() or None

    # Sin filtros: devolver todos los IDs.
    if not any((activity, region, investment)):
        return [
            s["series_id"] for s in series
            if isinstance(s, dict) and s.get("series_id")
        ]

    # Fallback de region a nivel de cuadro (single-region).
    cuadro_cls = observations.get("classification") or {}
    cuadro_region = str(cuadro_cls.get("region") or "").strip().lower() or None

    ids: List[str] = []
    for s in series:
        if not isinstance(s, dict) or not s.get("series_id"):
            continue
        cls = s.get("classification_series") or {}
        if not isinstance(cls, dict):
            cls = {}

        if activity:
            val = str(cls.get("activity") or "").strip().lower()
            if val != activity:
                continue

        if region:
            val = str(cls.get("region") or "").strip().lower()
            if val != region:
                # Si la serie no trae region, usar region de cuadro.
                if cuadro_region != region:
                    continue

        if investment:
            val = str(cls.get("investment") or "").strip().lower()
            if val != investment:
                continue

        ids.append(s["series_id"])

    return ids


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
        # Extraer entidades.
        question, entities, ent = _extract_entities_from_state(state)

        # Entidades resueltas por normalizer.
        logger.info("[DATA_NODE] indicator=%s freq=%s activity=%s req_form=%s",
                    ent.indicator_ent, ent.frequency_ent, ent.activity_ent, ent.req_form_cls)

        # Buscar payload en data_store con filtros resueltos.
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

        # Sin match: respuesta estandar.
        if not matches:
            return handle_no_series(
                question=question,
                entities=entities,
                ent=ent,
                writer=writer,
                emit_fn=_emit_stream_chunk,
                first_non_empty_fn=first_non_empty,
            )

        # Tomar el primer match como observations.
        observations = matches[0]["payload"]

        # Filtrar series incompatibles.
        observations = _filter_series_by_entities(observations, ent)

        # IDs de series que cumplen activity/region/investment.
        series = _collect_target_series_ids(observations, ent)
        logger.info("[DATA_NODE] series=%s", series)

        logger.info("[DATA_NODE] cuadro=%s freq=%s series_count=%d",
                    observations.get("cuadro_name"),
                    observations.get("frequency"),
                    len(observations.get("series", [])))

        ent_dict = asdict(ent)

        # Armar payload para response.
        payload = {
            "question": question,
            "observations": observations,
            "series": series,
            "entities": ent_dict,
        }

        # Transmitir respuesta en streaming para response.
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

        return {
            "output": "".join(collected),
            "entities": entities,
            "series": series,
            "data_classification": ent_dict,
        }

    return data_node


__all__ = ["make_data_node"]
