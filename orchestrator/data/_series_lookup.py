"""Búsqueda de familia y selección de serie en el catálogo de series.

Encapsula la lógica de:
  1. Determinar los parámetros de búsqueda para la familia de series.
  2. Localizar la familia en el catálogo agrupado.
  3. Seleccionar la serie objetivo dentro de esa familia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ._business_rules import ResolvedEntities
from ._helpers import build_target_series_url

logger = logging.getLogger(__name__)


@dataclass
class SeriesLookupResult:
    """Resultado de la búsqueda de serie en el catálogo."""

    family_dict: Optional[Dict[str, Any]] = None
    family_series: List[Dict[str, Any]] = None  # type: ignore[assignment]
    family_name: Optional[str] = None
    source_url: Optional[str] = None
    target_series: Optional[Dict[str, Any]] = None
    target_series_id: Optional[str] = None
    target_series_title: Optional[str] = None
    target_series_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.family_series is None:
            self.family_series = []


def lookup_series(ent: ResolvedEntities) -> SeriesLookupResult:
    """Localiza la familia de series y selecciona la serie objetivo.

    Pasos:
      1. Calcula los filtros de familia (frecuencia, precio, estacionalidad, etc.).
      2. Busca la familia en el catálogo con ``find_family_by_classification``.
      3. Construye el diccionario de igualdad (``series_eq``) para seleccionar la
         serie concreta dentro de la familia.
      4. Retorna un ``SeriesLookupResult`` con toda la información necesaria.
    """
    from orchestrator.catalog.series_search import (
        family_to_series_rows,
        find_family_by_classification,
        select_target_series_by_classification,
    )

    result = SeriesLookupResult()

    # --- Filtros de familia ---------------------------------------------------
    family_frequency = None if ent.indicator_ent == "imacec" else ent.frequency_ent
    if ent.calc_mode_cls != "contribution":
        family_frequency = None
    family_price = None if ent.indicator_ent == "imacec" else ent.price

    is_pib_aggregate = (
        ent.indicator_ent == "pib"
        and ent.activity_cls_resolved in (None, "none", "general")
        and ent.region_cls in (None, "none")
        and ent.investment_cls in (None, "none")
    )

    family_calc_mode = ent.calc_mode_cls or "original"

    # Estacionalidad: para PIB agregado sin contribución se omite salvo que
    # el usuario la haya solicitado explícitamente.
    requested_seasonality = str(ent.seasonality_ent or "").strip().lower()
    has_requested_seasonality = requested_seasonality not in {"", "none", "null"}
    if is_pib_aggregate and ent.calc_mode_cls != "contribution" and not has_requested_seasonality:
        family_seasonality = None
    else:
        family_seasonality = ent.seasonality_ent

    # --- Buscar familia -------------------------------------------------------
    result.family_dict = find_family_by_classification(
        "orchestrator/catalog/catalog.json",
        indicator=ent.indicator_ent,
        activity_value=(
            ent.activity_ent if ent.activity_ent is not None
            else ent.activity_cls_resolved
        ),
        region_value=(
            ent.region_ent if ent.region_ent is not None else ent.region_cls
        ),
        investment_value=(
            ent.investment_ent if ent.investment_ent is not None
            else ent.investment_cls
        ),
        calc_mode=family_calc_mode,
        price=family_price,
        seasonality=family_seasonality,
        frequency=family_frequency,
        hist=ent.hist,
    )

    # PIB agregado no-contribution: en catalogo la estacionalidad puede estar
    # definida a nivel de serie (no de familia). Si no hay match por family
    # seasonality, reintentar sin ese filtro para no perder la familia correcta.
    if (
        not isinstance(result.family_dict, dict)
        and is_pib_aggregate
        and str(ent.calc_mode_cls or "").strip().lower() != "contribution"
        and has_requested_seasonality
    ):
        result.family_dict = find_family_by_classification(
            "orchestrator/catalog/catalog.json",
            indicator=ent.indicator_ent,
            activity_value=(
                ent.activity_ent if ent.activity_ent is not None
                else ent.activity_cls_resolved
            ),
            region_value=(
                ent.region_ent if ent.region_ent is not None else ent.region_cls
            ),
            investment_value=(
                ent.investment_ent if ent.investment_ent is not None
                else ent.investment_cls
            ),
            calc_mode=family_calc_mode,
            price=family_price,
            seasonality=None,
            frequency=family_frequency,
            hist=ent.hist,
        )
        logger.info(
            "[DATA_NODE] fallback family lookup (sin seasonality) | matched=%s",
            isinstance(result.family_dict, dict),
        )

    if isinstance(result.family_dict, dict):
        result.family_series = family_to_series_rows(result.family_dict)
        result.source_url = result.family_dict.get("source_url")
        result.family_name = result.family_dict.get("family_name")
    else:
        result.family_series = []

    logger.info("[DATA_NODE] family_name=%s", result.family_name)
    logger.info("[DATA_NODE] family_source_url=%s", result.source_url)
    logger.info("[DATA_NODE] =========================================================")

    # --- Seleccionar serie objetivo -------------------------------------------
    # Solo incluir dimensiones de seleccion de serie (nivel fila de serie).
    # calc_mode/frequency se usan para elegir familia; al reusarlas aqui pueden
    # bloquear matches cuando vienen como lista o null en el catalogo.
    series_eq: Dict[str, Any] = {
        "indicator": ent.indicator_ent,
        "seasonality": ent.seasonality_ent,
        "activity": ent.activity_ent,
        "region": ent.region_ent,
        "investment": ent.investment_ent,
    }

    # Para contribuciones generales de PIB/IMACEC, ajustar la clave de actividad
    # al token que exista en la familia de series.
    if (
        ent.calc_mode_cls == "contribution"
        and ent.activity_cls_resolved in (None, "none", "general")
        and ent.region_cls in (None, "none")
        and ent.investment_cls in (None, "none")
    ):
        indicator_norm = str(ent.indicator_ent or "").strip().lower()
        if indicator_norm in {"pib", "imacec"}:
            activity_tokens_in_family = {
                str(
                    ((row.get("classification") or {}).get("activity") or "")
                ).strip().lower()
                for row in result.family_series
                if isinstance(row, dict)
            }
            if indicator_norm in activity_tokens_in_family:
                series_eq["activity"] = indicator_norm
            else:
                series_eq.pop("activity", None)
                series_eq["indicator"] = indicator_norm

    if ent.activity_cls_resolved == "specific" and ent.activity_ent is None:
        series_eq["activity"] = "__missing_specific_activity__"

    # Consultas agregadas (sin actividad especifica): no forzar filtro activity,
    # ya que para IMACEC/PIB la serie principal suele venir solo con indicator.
    if (
        str(ent.calc_mode_cls or "").strip().lower() != "contribution"
        and ent.activity_cls_resolved in (None, "none", "general")
        and ent.region_cls in (None, "none")
        and ent.investment_cls in (None, "none")
    ):
        series_eq.pop("activity", None)

    result.target_series = select_target_series_by_classification(
        result.family_series,
        eq=series_eq,
        fallback_to_first=True,
    )

    if isinstance(result.target_series, dict):
        result.target_series_id = result.target_series.get("id")
        long_raw = result.target_series.get("long_title")
        display_raw = result.target_series.get("display_title")
        result.target_series_title = str(long_raw or display_raw or "").strip()

    result.target_series_url = build_target_series_url(
        source_url=result.source_url,
        series_id=result.target_series_id,
        period=ent.period_ent if isinstance(ent.period_ent, list) else None,
        frequency=ent.frequency_ent,
        calc_mode=ent.calc_mode_cls,
    )

    logger.info("[DATA_NODE] target_series_id=%s", result.target_series_id)
    logger.info("[DATA_NODE] target_series_title=%s", result.target_series_title)
    logger.info("[DATA_NODE] target_series_url=%s", result.target_series_url)
    logger.info("[DATA_NODE] =========================================================")

    return result
