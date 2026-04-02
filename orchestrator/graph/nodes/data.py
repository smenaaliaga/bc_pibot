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
from orchestrator.normalizer.normalizer import ResolvedEntities

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
    ResolvedEntities,            # ent final para búsqueda
]:
    """Extrae entidades desde el estado del grafo.

    El ``ResolvedEntities`` ya viene pre-computado por el clasificador
    (``classifier_agent.py``), por lo que aquí solo se extrae.
    """

    question = state.get("question", "")
    entities_state = _clone_entities(state.get("entities"))

    classification = state.get("classification")

    classification_entities = getattr(classification, "entities", None) or {}

    entities: List[Dict[str, Any]]
    if isinstance(classification_entities, dict):
        entities = [dict(classification_entities)]
    else:
        entities = entities_state

    ent = getattr(classification, "resolved_entities", None)
    if not isinstance(ent, ResolvedEntities):
        logger.warning("[DATA_NODE] classification.resolved_entities ausente; usando defaults vacíos")
        ent = ResolvedEntities()

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
        kwargs["region"] = ent.region_ent[0]

    inv = str(ent.investment_cls or "").strip().lower()
    if inv in ("general", "specific"):
        kwargs["has_investment"] = 1
    elif inv == "none":
        kwargs["has_investment"] = 0

    if ent.hist is not None:
        kwargs["hist"] = ent.hist

    return kwargs


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

    activity_set = {s.strip().lower() for s in ent.activity_ent if s} or None
    region_set = {s.strip().lower() for s in ent.region_ent if s} or None
    investment_set = {s.strip().lower() for s in ent.investment_ent if s} or None

    # Sin filtros: devolver todos los IDs.
    if not any((activity_set, region_set, investment_set)):
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

        if activity_set:
            val = str(cls.get("activity") or "").strip().lower()
            if val not in activity_set:
                continue

        if region_set:
            val = str(cls.get("region") or "").strip().lower()
            if val not in region_set:
                # Si la serie no trae region, usar region de cuadro.
                if cuadro_region not in region_set:
                    continue

        if investment_set:
            val = str(cls.get("investment") or "").strip().lower()
            if val not in investment_set:
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

        logger.info("[DATA_NODE] cuadro=%s | freq=%s",
                    observations.get("cuadro_name"),
                    observations.get("frequency"))
        
        # IDs de series que cumplen activity/region/investment.
        series = _collect_target_series_ids(observations, ent)
        logger.info("[DATA_NODE] series=%s", series)

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
