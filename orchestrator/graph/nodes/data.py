"""Nodo DATA del grafo PIBot — orquesta la obtención de datos económicos.

Flujo principal (``data_node``):
    1. Extrae entidades y clasificación del estado del grafo.
    2. Aplica las reglas de negocio (``_business_rules``).
    3. Busca la familia y serie objetivo en el catálogo (``catalog_lookup``).
    4. Obtiene las observaciones según la rama correspondiente (``_fetch``).
    5. Construye el payload y lo envía al flujo de respuesta (streaming).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from langgraph.types import StreamWriter

from orchestrator.data.response import handle_no_series, stream_data_response
from ..state import AgentState, _clone_entities, _emit_stream_chunk
from orchestrator.data._helpers import coerce_period, extract_year, first_non_empty, to_period_end_str, has_full_quarterly_year
from orchestrator.data._business_rules import ResolvedEntities, apply_business_rules
from orchestrator.catalog.catalog_lookup import lookup_series

logger = logging.getLogger(__name__)

# Re-exportar para compatibilidad con tests que hacen monkeypatch.
from orchestrator.data._helpers import build_target_series_url as _build_target_series_url  # noqa: F401
from orchestrator.data._fetch import (
    load_series_observations as _load_series_observations,  # noqa: F401
    fetch_series_by_req_form as _fetch_series_by_req_form,  # noqa: F401
)


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
    List[Dict[str, Any]],       # entities
    ResolvedEntities,            # ent (mutable, pre-reglas)
]:
    """Extrae y normaliza las entidades desde el estado del grafo."""
    question = state.get("question", "")
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
        period_ent=period_ent,
        calc_mode_cls=calc_mode_cls,
        activity_cls=activity_cls,
        activity_cls_resolved=activity_cls,
        region_cls=region_cls,
        investment_cls=investment_cls,
        req_form_cls=req_form_cls,
    )

    logger.info("[DATA_NODE] resolved entities (pre-rules)=%s", asdict(ent))

    return question, entities, ent





def _obs_to_list(obs_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normaliza observaciones crudas al formato compacto del payload."""
    return [
        {
            "date": str(o.get("date", "")),
            "value": o.get("value"),
            "yoy_pct": o.get("yoy_pct"),
            "pct": o.get("pct"),
        }
        for o in obs_raw
        if isinstance(o, dict)
    ]


def load_observations(
    series_list: List[Dict[str, Any]],
    load_fn,
    period_values: Optional[List[Any]] = None,
    frequency: Optional[str] = None,
    indicator: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Carga observaciones de cada serie y retorna dict indexado por series_id.

    Para PIB con frecuencia anual, obtiene datos tanto anuales como
    trimestrales.  Los años anuales sin los 4 trimestres completos se
    eliminan de la lista anual y se señalan en meta.  Las observaciones
    se agrupan por frecuencia: ``{"A": [...], "Q": [...]}``.

    Estructura retornada::

        {
            "<series_id>": {
                "meta": { ... },
                "observations": [...]             # caso normal (lista)
            },
        }

    Para PIB anual::

        {
            "<series_id>": {
                "meta": { ... },
                "observations": {                 # dict con sub-llaves
                    "A": [...],
                    "Q": [...],
                },
            },
        }
    """
    firstdate = str(period_values[0]) if period_values else None
    lastdate = str(period_values[-1]) if period_values else None

    # Ajustar lastdate al fin de periodo para coincidir con fechas de observación
    if lastdate and frequency:
        lastdate = to_period_end_str(lastdate, frequency)

    result: Dict[str, Dict[str, Any]] = {}

    for s in (series_list or []):
        sid = s.get("id") if isinstance(s, dict) else None
        if not sid:
            continue

        target_freq = frequency.upper() if frequency else None

        obs_raw, meta = load_fn(
            series_id=sid,
            firstdate=firstdate,
            lastdate=lastdate,
            target_frequency=target_freq,
            agg_mode="sum",
            calc_mode=None,
        )

        # Fallback: si el periodo solicitado queda fuera del rango disponible,
        # cargar sin filtro y tomar el extremo más cercano.
        if not obs_raw and period_values and meta and isinstance(meta, dict):
            last_avail = meta.get("last_available_date") or ""
            first_avail = meta.get("first_available_date") or ""
            if lastdate and last_avail and lastdate > last_avail:
                obs_raw, meta = load_fn(
                    series_id=sid,
                    firstdate=last_avail,
                    lastdate=last_avail,
                    target_frequency=target_freq,
                    agg_mode="sum",
                    calc_mode=None,
                )
                logger.info("[load_observations] Periodo posterior al disponible; usando last_available_date=%s", last_avail)
            elif firstdate and first_avail and firstdate < first_avail:
                obs_raw, meta = load_fn(
                    series_id=sid,
                    firstdate=first_avail,
                    lastdate=first_avail,
                    target_frequency=target_freq,
                    agg_mode="sum",
                    calc_mode=None,
                )
                logger.info("[load_observations] Periodo anterior al disponible; usando first_available_date=%s", first_avail)

        # Anotar frecuencia remuestreada si difiere de la original
        if meta and isinstance(meta, dict) and target_freq:
            orig = (meta.get("original_frequency") or "").upper()
            if orig and target_freq != orig:
                meta["target_frequency"] = target_freq

        result[sid] = {
            "meta": meta,
            "observations": _obs_to_list(obs_raw),
        }

    # ------------------------------------------------------------------
    # Post-proceso: para PIB anual, obtener también datos trimestrales
    # y depurar años anuales incompletos.
    # ------------------------------------------------------------------
    is_pib_annual = (
        str(indicator or "").strip().lower() == "pib"
        and frequency
        and frequency.strip().upper() == "A"
    )
    if is_pib_annual:
        for sid in list(result.keys()):
            entry = result[sid]
            meta = entry.get("meta") or {}
            obs = entry.get("observations") or []
            orig = (meta.get("original_frequency") or "").upper()
            # Fallback: si el cache reporta "U", inferir desde el sufijo de la serie
            if orig not in ("Q", "T", "M"):
                suffix = sid.strip().split(".")[-1].upper()
                if suffix == "T":
                    orig = "Q"
            if orig not in ("Q", "T", "M"):
                continue

            # Años presentes en la lista anual
            years = sorted({extract_year(o.get("date")) for o in obs if extract_year(o.get("date"))})
            if not years:
                continue

            q_first = f"{min(years)}-01-01"
            q_last = f"{max(years)}-12-31"

            q_obs, q_meta = load_fn(
                series_id=sid,
                firstdate=q_first,
                lastdate=q_last,
                target_frequency="Q",
                agg_mode="sum",
                calc_mode=None,
            )

            # Remover años incompletos de la lista anual
            removed_years: List[int] = []
            clean_annual: List[Dict[str, Any]] = []
            for o in obs:
                y = extract_year(o.get("date"))
                if y and has_full_quarterly_year(q_obs, y):
                    clean_annual.append(o)
                elif y:
                    removed_years.append(y)

            if removed_years:
                meta["incomplete_annual_note"] = (
                    f"Los años {', '.join(str(y) for y in removed_years)} no tienen "
                    f"los 4 trimestres completos; ver datos trimestrales."
                ) if len(removed_years) > 1 else (
                    f"El año {removed_years[0]} no tiene los 4 trimestres "
                    f"completos; ver datos trimestrales."
                )
                logger.info(
                    "[load_observations] Años anuales incompletos removidos: %s para %s",
                    removed_years, sid,
                )

            # Agrupar observaciones por frecuencia en un solo dict
            result[sid]["observations"] = {
                "A": clean_annual,
                "Q": _obs_to_list(q_obs),
            }

    return result


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

        # 3. Buscar serie en catálogo
        sl = lookup_series(ent)
        logger.info("[DATA_NODE] family=%s target=%s", sl.family_name, sl.target_series_id)

        # 4. Sin serie → respuesta informativa
        if not sl.source_url or not sl.target_series_id:
            return handle_no_series(
                question=question,
                entities=entities,
                ent=ent,
                writer=writer,
                emit_fn=_emit_stream_chunk,
                first_non_empty_fn=first_non_empty,
            )

        # 5. Cargar observaciones
        is_contribution = str(ent.calc_mode_cls or "").strip().lower() == "contribution"
        _cls_vals = [
            str(ent.activity_cls or "").strip().lower(),
            str(ent.region_cls or "").strip().lower(),
            str(ent.investment_cls or "").strip().lower(),
        ]
        all_none = all(v in ("none", "", "{}") for v in _cls_vals)
        any_specific = any(v == "specific" for v in _cls_vals)
        if is_contribution:
            series_to_load = sl.family_series
            logger.info("[DATA_NODE][STEP-5] Familia completa por contribución (%d series)", len(series_to_load or []))
        elif all_none or any_specific:
            series_to_load = [{"id": sl.target_series_id}]
            logger.info("[DATA_NODE][STEP-5] Solo serie target (all_none=%s any_specific=%s)", all_none, any_specific)
        else:
            series_to_load = sl.family_series
            logger.info("[DATA_NODE][STEP-5] Familia completa (%d series)", len(series_to_load or []))

        observations = load_observations(
            series_to_load, _get_load_fn(), ent.period_ent, ent.frequency_ent,
            indicator=ent.indicator_ent,
        )

        # Para contribuciones, pct y yoy_pct no tienen sentido: el valor ya
        # representa puntos porcentuales de contribución al crecimiento.
        if is_contribution:
            for _entry in observations.values():
                obs = _entry.get("observations")
                if isinstance(obs, list):
                    for o in obs:
                        o.pop("pct", None)
                        o.pop("yoy_pct", None)
                elif isinstance(obs, dict):
                    for sub in obs.values():
                        if isinstance(sub, list):
                            for o in sub:
                                o.pop("pct", None)
                                o.pop("yoy_pct", None)

        # 6. Construir payload y hacer streaming de la respuesta LLM
        payload = {
            "question": question,
            "classification": asdict(ent),
            "price": ent.price,
            "observations": observations,
            "family_name": sl.family_name,
            "series": sl.target_series_id,
            "series_title": sl.target_series_title,
            "source_url": sl.source_url,
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

        ent_dict = asdict(ent)
        pv = ent.period_ent or []
        rf = str(ent.req_form_cls or "").strip().lower()

        return {
            "output": "".join(collected),
            "entities": entities,
            "parsed_point": str(pv[-1]) if (rf != "range" and pv) else None,
            "parsed_range": (str(pv[0]), str(pv[-1])) if pv else None,
            "series": sl.target_series_id,
            "data_classification": ent_dict,
        }

    return data_node


__all__ = ["make_data_node"]
