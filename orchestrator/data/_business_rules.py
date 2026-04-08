"""Reglas de negocio aplicadas a las entidades normalizadas antes de buscar series.

Estas reglas ajustan indicador, frecuencia, actividad, inversión y precio
según las restricciones del dominio económico del Banco Central de Chile.

Cada regla está documentada con su justificación.
"""

from __future__ import annotations

import logging
import re
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
    historical_floor_instruction: Optional[str] = None


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
    """Aplica pisos históricos por indicador y ajusta filtro/periodo.

    - IMACEC: piso 1996. Nunca usa ``hist=1``; se normaliza a ``hist=0``.
    - PIB: piso 1960. Usa ``hist=1`` solo cuando el año de referencia queda < 1996.
    """
    from orchestrator.data._helpers import extract_year

    indicator = str(ent.indicator_ent or "").strip().lower()
    period = list(ent.period_ent or [])
    ref_year = extract_year(period[0]) if period else None

    if indicator == "imacec":
        ent.hist = 0
        if ref_year is not None and ref_year < 1996:
            ent.period_ent = _rewrite_period_to_floor(period, floor_year=1996)
            ent.historical_floor_instruction = (
                "REGLA DE DISPONIBILIDAD HISTÓRICA (IMACEC): solo hay datos empalmados "
                "de IMACEC desde 1996. Debes indicarlo explícitamente y reportar 1996 "
                "(o rango desde 1996, según corresponda)."
            )
        return

    # Default histórico para PIB y otros indicadores compatibles.
    ent.hist = 1 if (ref_year is not None and ref_year < 1996) else 0

    if indicator == "pib" and ref_year is not None and ref_year < 1960:
        ent.period_ent = _rewrite_period_to_floor(period, floor_year=1960)
        ent.historical_floor_instruction = (
            "REGLA DE DISPONIBILIDAD HISTÓRICA (PIB): hay datos empalmados desde 1960. "
            "Debes indicarlo explícitamente y reportar 1960 "
            "(o rango desde 1960, según corresponda)."
        )
        adjusted_year = extract_year(ent.period_ent[0]) if ent.period_ent else None
        ent.hist = 1 if (adjusted_year is not None and adjusted_year < 1996) else 0


def _rewrite_period_to_floor(period_ent: List[Any], floor_year: int) -> List[Any]:
    """Reescribe período al año piso manteniendo forma point/range cuando aplica."""
    if not period_ent:
        return []

    if len(period_ent) == 1:
        return [_replace_year_token(period_ent[0], floor_year)]

    start_raw = period_ent[0]
    end_raw = period_ent[-1]
    start = _replace_year_token(start_raw, floor_year)

    # Evita rangos invertidos cuando el fin también cae antes del piso.
    from orchestrator.data._helpers import extract_year

    end_year = extract_year(end_raw)
    if end_year is not None and end_year < floor_year:
        end = _replace_year_token(end_raw, floor_year)
    else:
        end = end_raw
    return [start, end]


def _replace_year_token(value: Any, year: int) -> str:
    """Reemplaza el año del token por ``year`` preservando el resto del formato."""
    text = str(value or "").strip()
    if not text:
        return str(year)
    if re.fullmatch(r"(19|20)\d{2}", text):
        return str(year)
    return re.sub(r"(19|20)\d{2}", str(year), text, count=1)


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
