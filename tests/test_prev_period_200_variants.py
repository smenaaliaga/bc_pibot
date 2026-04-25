"""Harness de 200 variantes para validar la solución del requerimiento
'VARIACIÓN DESESTACIONALIZADA Y CON RESPECTO AL PERIODO ANTERIOR'.

Cada variante se asocia a un escenario con expectativas claras sobre:
  - calc_mode_cls final (post reglas de negocio)
  - seasonality_ent final
  - intent override (solo escenarios B)
  - frequency esperada por indicador (m para imacec, q para pib, a anuales)

Los casos están construidos para evitar palabras-gatillo de yoy
(interanual, año anterior, mismo período del año, etc.) excepto en el
grupo de control GRP_YOY_CONTROL que sí debe mantenerse en 'yoy'.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

import pytest

from orchestrator.data._business_rules import (
    ResolvedEntities,
    apply_business_rules,
)
from orchestrator.graph.nodes.ingest import (
    _is_value_desestacionalizado,
    _normalize_intent_label,
)


# ---------------------------------------------------------------------------
# Generadores de variantes
# ---------------------------------------------------------------------------

def _gen(prefixes: List[str], cores: List[str], suffixes: List[str]) -> List[str]:
    out: List[str] = []
    for p, c, s in itertools.product(prefixes, cores, suffixes):
        text = f"{p} {c} {s}".strip()
        text = " ".join(text.split())
        if text:
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# Grupo A: 'variación mensual del IMACEC' → prev_period (Req. 3a)
# ---------------------------------------------------------------------------

GRP_VAR_MENSUAL_IMACEC = _gen(
    prefixes=[
        "cual es", "cuál es", "dame", "muestrame", "muéstrame",
        "podrias mostrarme", "necesito", "quiero saber",
    ],
    cores=[
        "la variación mensual del imacec",
        "la variacion mensual del imacec",
        "la variación mensual del IMACEC",
    ],
    suffixes=["", "por favor", "actual", "más reciente", "del último mes"],
)[:25]


# ---------------------------------------------------------------------------
# Grupo B: 'variación trimestral del PIB' → prev_period (Req. 3b)
# ---------------------------------------------------------------------------

GRP_VAR_TRIMESTRAL_PIB = _gen(
    prefixes=[
        "cual es", "cuál es", "dame", "muestrame", "muéstrame",
        "podrias mostrarme", "necesito", "quiero saber",
    ],
    cores=[
        "la variación trimestral del pib",
        "la variacion trimestral del pib",
        "la variación trimestral del PIB",
    ],
    suffixes=["", "por favor", "actual", "más reciente", "del último trimestre"],
)[:25]


# ---------------------------------------------------------------------------
# Grupo C: 'imacec/pib desestacionalizado' (con 'valor' u otro pedido de dato)
# → calc_mode='prev_period' (Req. 1, 2, 6)
# ---------------------------------------------------------------------------

GRP_DESEST_IMACEC = _gen(
    prefixes=["cual es", "cuál es", "dame", "muestrame", "necesito"],
    cores=[
        "el valor del imacec desestacionalizado",
        "el imacec desestacionalizado",
        "el nivel del imacec desestacionalizado",
        "el dato del imacec desestacionalizado",
        "la cifra del imacec desestacionalizado",
    ],
    suffixes=["", "por favor", "actual", "más reciente"],
)[:25]

GRP_DESEST_PIB = _gen(
    prefixes=["cual es", "cuál es", "dame", "muestrame", "necesito"],
    cores=[
        "el valor del pib desestacionalizado",
        "el pib desestacionalizado",
        "el nivel del pib desestacionalizado",
        "el dato del pib desestacionalizado",
        "la cifra del pib desestacionalizado",
    ],
    suffixes=["", "por favor", "actual", "más reciente"],
)[:25]


# ---------------------------------------------------------------------------
# Grupo D: 'respecto al periodo anterior' explícito → prev_period (Req. 3)
# ---------------------------------------------------------------------------

GRP_PERIODO_ANTERIOR_IMACEC = _gen(
    prefixes=["cual es", "cuál es", "dame"],
    cores=[
        "la variación del imacec respecto al periodo anterior",
        "la variación del imacec con respecto al periodo anterior",
        "el cambio del imacec respecto al periodo anterior",
        "el último valor del imacec con respecto al periodo anterior",
    ],
    suffixes=["", "por favor"],
)[:15]

GRP_PERIODO_ANTERIOR_PIB = _gen(
    prefixes=["cual es", "cuál es", "dame"],
    cores=[
        "la variación del pib respecto al periodo anterior",
        "la variación del pib con respecto al periodo anterior",
        "el cambio del pib respecto al periodo anterior",
        "el último valor del pib con respecto al periodo anterior",
    ],
    suffixes=["", "por favor"],
)[:15]


# ---------------------------------------------------------------------------
# Grupo E: 'valor desestacionalizado' (intent override de Capa B)
# Las preguntas que el clasificador suele etiquetar como 'methodology' por la
# baja confianza (ver log run_detail_20260425, t=21:35:40).
# ---------------------------------------------------------------------------

GRP_VALUE_DESEST_OVERRIDE = [
    "cual es el valor desestacionalizado del imacec",
    "cuál es el valor desestacionalizado del imacec",
    "cual es el valor desestacionalizado del pib",
    "cuál es el valor desestacionalizado del pib",
    "cuanto es el valor desestacionalizado del imacec",
    "dame el valor desestacionalizado del imacec",
    "muéstrame el valor desestacionalizado del pib",
    "necesito el valor desestacionalizado del imacec",
    "quiero el valor desestacionalizado del pib",
    "cual fue el valor desestacionalizado del imacec",
    "cuanto es el imacec desestacionalizado en valor",
    "el valor desestacionalizado del imacec actual",
    "valor desestacionalizado imacec",
    "valor desestacionalizado pib",
    "cual es el dato desestacionalizado del imacec",
    "cual es el nivel desestacionalizado del pib",
    "cual es la cifra desestacionalizada del imacec",
    "dame la cifra desestacionalizada del pib",
    "cuanto es el nivel desestacionalizado del imacec",
    "cual es el valor sin estacionalidad del imacec",
]


# ---------------------------------------------------------------------------
# Grupo F (control): variación interanual / yoy explícita → MANTENER yoy
# ---------------------------------------------------------------------------

GRP_YOY_CONTROL = [
    "cual es la variación interanual del imacec",
    "cual es la variación interanual del pib",
    "cuál fue la variación del imacec respecto al año anterior",
    "cuál fue la variación del pib respecto al mismo período del año anterior",
    "variación anual del imacec",
    "variación anual del pib",
    "imacec respecto al mismo mes del año anterior",
    "pib respecto al mismo trimestre del año anterior",
    "variación 12 meses del imacec",
    "imacec a doce meses",
    "cuál es la variación interanual del imacec en marzo",
    "cuál es la variación interanual del pib en el segundo trimestre",
    "imacec respecto al año anterior",
    "pib respecto al año anterior",
    "variación anual del pib en 2024",
    "imacec respecto al mismo mes del año pasado",
    "pib respecto al mismo trimestre del año pasado",
    "variación 12 meses del pib",
    "imacec doce meses",
    "pib a doce meses",
]


# ---------------------------------------------------------------------------
# Grupo G (control): 'cuanto es el pib' (nivel, no desestacionalizado) → no tocar
# ---------------------------------------------------------------------------

GRP_NIVEL_NSA_CONTROL = [
    "cual es el valor del pib",
    "cual es el valor del imacec",
    "cuanto es el pib en 2024",
    "cuanto fue el imacec en 2023",
    "dame el imacec",
    "dame el pib",
    "imacec actual",
    "pib actual",
    "imacec 2025",
    "pib 2025",
    "cual fue el imacec en enero",
    "cual fue el pib en el primer trimestre",
    "cuanto fue el imacec el mes pasado",
    "muéstrame el imacec",
    "muéstrame el pib",
    "necesito el imacec actual",
    "necesito el pib actual",
    "valor del imacec",
    "valor del pib",
    "imacec del último mes",
    "pib del último trimestre",
    "cuanto asciende el pib",
    "cuanto asciende el imacec",
    "dame la cifra del pib",
    "dame el dato del imacec",
    "cual es el imacec",
    "cual es el pib",
    "imacec del mes",
    "pib del trimestre",
    "imacec en el mes pasado",
]


# ---------------------------------------------------------------------------
# Composición final: ~200 casos
# ---------------------------------------------------------------------------

ALL_PREV_PERIOD_CASES: List[Tuple[str, str, str]] = (
    [(q, "imacec", "var_mensual") for q in GRP_VAR_MENSUAL_IMACEC]
    + [(q, "pib", "var_trimestral") for q in GRP_VAR_TRIMESTRAL_PIB]
    + [(q, "imacec", "desest_imacec") for q in GRP_DESEST_IMACEC]
    + [(q, "pib", "desest_pib") for q in GRP_DESEST_PIB]
    + [(q, "imacec", "periodo_anterior_imacec") for q in GRP_PERIODO_ANTERIOR_IMACEC]
    + [(q, "pib", "periodo_anterior_pib") for q in GRP_PERIODO_ANTERIOR_PIB]
)

# Para los overrides de intent (Capa B) inferimos indicador del texto:
def _infer_indicator(text: str) -> str:
    return "imacec" if "imacec" in text.lower() else "pib"

ALL_VALUE_DESEST_OVERRIDE: List[str] = list(GRP_VALUE_DESEST_OVERRIDE)
ALL_YOY_CONTROL: List[str] = list(GRP_YOY_CONTROL)
ALL_NIVEL_NSA_CONTROL: List[str] = list(GRP_NIVEL_NSA_CONTROL)


# ---------------------------------------------------------------------------
# Helpers de simulación
# ---------------------------------------------------------------------------

def _simulate_classifier_for_prev_period_case(question: str, indicator: str, group: str) -> ResolvedEntities:
    """Simula la salida típica del clasificador para los casos del log.

    El clasificador real produjo:
      - 'variación mensual del imacec'      → calc_mode='yoy', seasonality='nsa'
      - 'variación trimestral del pib'      → calc_mode='yoy', seasonality='nsa'
      - 'imacec/pib desestacionalizado'     → calc_mode='original', seasonality='sa'
      - '... respecto al periodo anterior'  → calc_mode='prev_period', seasonality='sa'
    """
    if group in ("var_mensual", "var_trimestral"):
        calc_mode = "yoy"
        seasonality = "nsa"
    elif group in ("desest_imacec", "desest_pib"):
        calc_mode = "original"
        seasonality = "sa"
    else:  # periodo_anterior_*
        calc_mode = "prev_period"
        seasonality = "sa"

    freq = "m" if indicator == "imacec" else "q"
    return ResolvedEntities(
        indicator_ent=indicator,
        seasonality_ent=seasonality,
        frequency_ent=freq,
        calc_mode_cls=calc_mode,
        question=question,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_total_variant_count_at_least_200():
    """Sanity: el harness genera al menos 200 variantes en total."""
    total = (
        len(ALL_PREV_PERIOD_CASES)
        + len(ALL_VALUE_DESEST_OVERRIDE)
        + len(ALL_YOY_CONTROL)
        + len(ALL_NIVEL_NSA_CONTROL)
    )
    assert total >= 200, f"Solo {total} variantes generadas"


@pytest.mark.parametrize("question,indicator,group", ALL_PREV_PERIOD_CASES)
def test_prev_period_cases_resolve_to_prev_period(question: str, indicator: str, group: str):
    """Toda variante de los grupos A–D debe terminar con calc_mode='prev_period'
    y seasonality='sa' tras aplicar las reglas de negocio."""
    ent = _simulate_classifier_for_prev_period_case(question, indicator, group)
    apply_business_rules(ent)

    assert ent.calc_mode_cls == "prev_period", (
        f"[{group}] q={question!r} → calc_mode={ent.calc_mode_cls} "
        f"(esperado prev_period)"
    )
    assert ent.seasonality_ent == "sa", (
        f"[{group}] q={question!r} → seasonality={ent.seasonality_ent} "
        f"(esperado sa)"
    )


@pytest.mark.parametrize("question", ALL_VALUE_DESEST_OVERRIDE)
def test_value_desestacionalizado_intent_override_detected(question: str):
    """Las preguntas tipo 'valor desestacionalizado del X' deben ser detectadas
    por el override de intent (Capa B) → reruteo a data."""
    assert _is_value_desestacionalizado(question), (
        f"q={question!r} no fue detectada por _is_value_desestacionalizado"
    )


@pytest.mark.parametrize("question", ALL_VALUE_DESEST_OVERRIDE)
def test_value_desestacionalizado_resolves_prev_period(question: str):
    """Tras override de intent + reglas de negocio, estos casos resuelven
    a calc_mode='prev_period' y seasonality='sa'."""
    indicator = _infer_indicator(question)
    freq = "m" if indicator == "imacec" else "q"
    # El clasificador da 'method'+'original'+'sa' (caso real del log):
    ent = ResolvedEntities(
        indicator_ent=indicator,
        seasonality_ent="sa",
        frequency_ent=freq,
        calc_mode_cls="original",
        question=question,
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls == "prev_period", (
        f"q={question!r} → calc_mode={ent.calc_mode_cls}"
    )
    assert ent.seasonality_ent == "sa"


@pytest.mark.parametrize("question", ALL_YOY_CONTROL)
def test_yoy_control_preserves_yoy(question: str):
    """Las preguntas yoy explícitas deben mantenerse en yoy + nsa."""
    indicator = _infer_indicator(question)
    freq = "m" if indicator == "imacec" else "q"
    ent = ResolvedEntities(
        indicator_ent=indicator,
        seasonality_ent="nsa",
        frequency_ent=freq,
        calc_mode_cls="yoy",
        question=question,
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls == "yoy", (
        f"q={question!r} → calc_mode={ent.calc_mode_cls} (esperado yoy)"
    )
    assert ent.seasonality_ent == "nsa"


@pytest.mark.parametrize("question", ALL_NIVEL_NSA_CONTROL)
def test_nivel_nsa_control_no_changes(question: str):
    """Preguntas tipo 'cuál es el valor del PIB' (sin desestacionalizado)
    deben mantenerse nsa + original (no se les fuerza prev_period)."""
    indicator = _infer_indicator(question)
    freq = "m" if indicator == "imacec" else "q"
    ent = ResolvedEntities(
        indicator_ent=indicator,
        seasonality_ent="nsa",
        frequency_ent=freq,
        calc_mode_cls="original",
        question=question,
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls != "prev_period", (
        f"q={question!r} → calc_mode pasó a prev_period sin justificación"
    )
    assert ent.seasonality_ent == "nsa"


# ---------------------------------------------------------------------------
# Tests de control sobre helpers de Capa B
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("cual es el valor desestacionalizado del imacec", True),
    ("cuál es el valor desestacionalizado del pib", True),
    ("dame el dato desestacionalizado del imacec", True),
    ("cifra desestacionalizada del imacec", True),  # 'cifra' es verbo de pedido
    ("qué es el imacec desestacionalizado", False),  # methodology genuino
    ("cómo se calcula el imacec desestacionalizado", False),
    ("metodología del imacec desestacionalizado", False),
    ("cual es la variación del imacec respecto al periodo anterior", False),
    ("variación interanual del pib", False),
])
def test_is_value_desestacionalizado_helper(question: str, expected: bool):
    assert _is_value_desestacionalizado(question) is expected, (
        f"q={question!r}"
    )
