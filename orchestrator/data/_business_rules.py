"""Reglas de negocio aplicadas a las entidades normalizadas antes de buscar series.

Estas reglas ajustan indicador, frecuencia, actividad, inversión y precio
según las restricciones del dominio económico del Banco Central de Chile.

Cada regla está documentada con su justificación.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from orchestrator.data.response import format_period_labels

logger = logging.getLogger(__name__)


@dataclass
class ResolvedEntities:
    """Contenedor mutable con las entidades ya ajustadas por reglas de negocio.

    Los campos ``*_cls`` corresponden a la clasificación del agente
    (nivel de clasificación: "general", "specific", "none"),
    mientras que los campos ``*_ent`` contienen los valores normalizados
    del NER (ej. "pib", "imacec", "manufactura").
    """

    indicator_ent: Optional[str] = None
    seasonality_ent: Optional[str] = None
    frequency_ent: Optional[str] = None
    activity_ent: Optional[str] = None
    region_ent: Optional[str] = None
    investment_ent: Optional[str] = None
    price_ent: Optional[str] = None
    period_ent: List[Any] = field(default_factory=list)

    # Clasificaciones del agente
    calc_mode_cls: Any = None
    activity_cls: Any = None
    activity_cls_resolved: Any = None
    region_cls: Any = None
    investment_cls: Any = None
    req_form_cls: Any = None

    # Campos derivados por reglas
    price: Optional[str] = None
    hist: Optional[int] = None


def apply_business_rules(ent: ResolvedEntities) -> ResolvedEntities:
    """Aplica las reglas de negocio secuencialmente y retorna el mismo objeto mutado.

    Reglas implementadas:

    1. **Contribución + inversión específica sin región**: forzar actividad "general".
    2. **IMACEC → frecuencia mensual**: IMACEC solo se publica mensual.
    3. **IMACEC sin actividad**: se asigna "imacec" como actividad por defecto.
    4. **Precio**: se asigna "enc" salvo para PIB agregado sin desglose.
    5. **PIB histórico**: se activa flag *hist* para consultas previas a 1996.
    6. **Contribución + inversión específica = demanda_interna**: reclasificar a general.
    7. **PIB mensual no existe**: se redirige a trimestral con nota informativa.
    """
    _rule_contribution_investment_force_general(ent)
    _rule_imacec_force_monthly(ent)
    _rule_imacec_default_activity(ent)
    _rule_assign_price(ent)
    _rule_pib_hist_flag(ent)
    _rule_contribution_demanda_interna(ent)
    _rule_redirect_pib_monthly_to_quarterly(ent)
    return ent


# ---------------------------------------------------------------------------
# Reglas individuales
# ---------------------------------------------------------------------------

def _rule_contribution_investment_force_general(ent: ResolvedEntities) -> None:
    """Si es contribución con inversión específica pero sin inversión ni región
    normalizadas, se fuerza actividad 'general' para obtener el desglose completo."""
    if (
        ent.calc_mode_cls == "contribution"
        and ent.investment_cls == "specific"
        and ent.investment_cls in (None, "none")
        and ent.region_cls in (None, "none")
    ):
        ent.activity_cls_resolved = "general"


def _rule_imacec_force_monthly(ent: ResolvedEntities) -> None:
    """IMACEC se publica exclusivamente con frecuencia mensual."""
    if (
        str(ent.indicator_ent or "").strip().lower() == "imacec"
        and str(ent.frequency_ent or "").strip().lower() != "m"
    ):
        ent.frequency_ent = "m"
        logger.info("[DATA_NODE] IMACEC: forzando frecuencia mensual (m)")


def _rule_imacec_default_activity(ent: ResolvedEntities) -> None:
    """Si el indicador es IMACEC y no se especificó actividad, se usa 'imacec'."""
    if ent.indicator_ent == "imacec" and ent.activity_ent is None:
        ent.activity_ent = "imacec"


def _is_empty_cls(value) -> bool:
    """True si el valor de clasificación representa ausencia de desglose."""
    return value in (None, "none", "general", "", {}, [], ())


def _rule_assign_price(ent: ResolvedEntities) -> None:
    """Determina si se requiere el parámetro de precio para la búsqueda de series.

    Si el normalizador detectó un precio explícito ("enc" o "co") se usa directamente.
    En caso contrario se usa "enc" como valor por defecto.
    """
    if ent.price_ent:
        ent.price = ent.price_ent
    else:
        ent.price = "enc"


def _rule_pib_hist_flag(ent: ResolvedEntities) -> None:
    """Define hist para el filtro: 1 si period[0] <= 1996, en otro caso 0."""
    from orchestrator.data._helpers import extract_year
    ref_year = extract_year(ent.period_ent[0]) if ent.period_ent else None
    ent.hist = 1 if (ref_year is not None and ref_year < 1996) else 0


def _rule_contribution_demanda_interna(ent: ResolvedEntities) -> None:
    """Si contribución + inversión específica = 'demanda_interna',
    se reclasifica a 'general' para obtener el desglose completo."""
    if (
        ent.calc_mode_cls == "contribution"
        and ent.investment_cls == "specific"
        and ent.investment_ent == "demanda_interna"
    ):
        ent.investment_cls = "general"


def _rule_redirect_pib_monthly_to_quarterly(ent: ResolvedEntities) -> None:
    """El PIB no existe en frecuencia mensual: se redirige a trimestral
    y se genera una nota informativa para el usuario."""
    if ent.indicator_ent != "pib":
        return
    if str(ent.frequency_ent or "").strip().lower() != "m":
        return

    requested_month_label = None
    if ent.period_ent:
        requested_month_label = format_period_labels(str(ent.period_ent[-1]), "m")[0]
        if requested_month_label == "--":
            requested_month_label = None

    if str(ent.calc_mode_cls or "").strip().lower() == "contribution":
        logger.warning("[DATA_NODE] PIB contribución mensual no existe; redirigiendo a trimestral")
    else:
        logger.warning("[DATA_NODE] PIB mensual no existe; redirigiendo a trimestral")

    ent.frequency_ent = "q"
    ent.req_form_cls = "latest"
    ent.period_ent = []
