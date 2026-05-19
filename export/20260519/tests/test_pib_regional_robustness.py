"""Regression suite for PIB regional robustness (run_detail_20260430.log).

Cubre los 4 escenarios reportados por el usuario en el log del 30/04/2026 y
agrega casos negativos para verificar que NO se afecte al resto de las
consultas (PIB nacional, IMACEC, sectorial, inversión, regional con calc_mode
explícito o req_form=range).

Foco:
  Cambio 1 — Filtro por región específica en _filter_series_by_entities.
  Cambio 2 — Orden determinista (single-region antes que multi-region).
  Cambio 3 — Default YoY para PIB regional (Rule14_PibRegionalDefaultYoY).
"""
from __future__ import annotations

import pytest

from orchestrator.data._business_rules import ResolvedEntities, apply_business_rules
from orchestrator.data.catalog_data_search import search_output_payloads
from orchestrator.graph.nodes.data import _filter_series_by_entities


# ---------------------------------------------------------------------------
# Cambio 3 — Rule14_PibRegionalDefaultYoY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "question,region_ent,frequency",
    [
        # Las 4 consultas problemáticas del run_detail_20260430.log:
        ("cúal es el pib de la región del bío bío del último trimestre", "biobio", "q"),
        ("cúal es el pib de la región del bío bío del último año", "biobio", "a"),
        ("cual es el pib de la región de los ríos del segundo trimestre del 2022", "los_rios", "q"),
        ("cual es el pib de la región del los lagos 2022", "los_lagos", "a"),
    ],
)
def test_pib_regional_point_forces_yoy(question, region_ent, frequency):
    ent = ResolvedEntities(
        question=question,
        indicator_ent="pib",
        frequency_ent=frequency,
        seasonality_ent="nsa",
        region_ent=region_ent,
        region_cls="specific",
        calc_mode_cls="original",
        req_form_cls="point",
        price_ent=None,
    )

    apply_business_rules(ent)

    assert ent.calc_mode_cls == "yoy", (
        f"Esperado calc_mode=yoy para PIB regional puntual sin señal de nivel "
        f"({question!r})"
    )
    # No debe pisar otras entidades.
    assert ent.indicator_ent == "pib"
    assert ent.region_ent == region_ent


def test_pib_regional_range_keeps_original():
    """req_form=range NO dispara el override (consultas multi-período)."""
    ent = ResolvedEntities(
        question="evolución del pib de la región del biobio entre 2020 y 2024",
        indicator_ent="pib",
        frequency_ent="a",
        region_ent="biobio",
        region_cls="specific",
        calc_mode_cls="original",
        req_form_cls="range",
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls == "original"


@pytest.mark.parametrize(
    "question",
    [
        "cuánto fue el monto del pib de la región del biobio en 2024",
        "valor en pesos del pib de la región de los lagos 2022",
        "cuántos pesos produjo el pib de la región de tarapaca el último trimestre",
        "nivel del pib de la región del biobio último trimestre",
        "miles de millones del pib de la región del maule en 2023",
    ],
)
def test_pib_regional_with_level_hint_keeps_original(question):
    """Si el usuario pide explícitamente el NIVEL/MONTO, no se fuerza yoy."""
    ent = ResolvedEntities(
        question=question,
        indicator_ent="pib",
        frequency_ent="a",
        region_ent="biobio",
        region_cls="specific",
        calc_mode_cls="original",
        req_form_cls="point",
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls == "original"


def test_pib_regional_with_explicit_yoy_keeps_yoy():
    ent = ResolvedEntities(
        question="cuánto creció el pib de la región del biobio en 2024",
        indicator_ent="pib",
        frequency_ent="a",
        region_ent="biobio",
        region_cls="specific",
        calc_mode_cls="yoy",
        req_form_cls="point",
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls == "yoy"


def test_pib_regional_with_prev_period_keeps_prev_period():
    ent = ResolvedEntities(
        question="variación margen pib biobio último trimestre",
        indicator_ent="pib",
        frequency_ent="q",
        region_ent="biobio",
        region_cls="specific",
        calc_mode_cls="prev_period",
        req_form_cls="point",
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls == "prev_period"


# ---------------------------------------------------------------------------
# Casos NEGATIVOS — la regla NO debe activarse
# ---------------------------------------------------------------------------

def test_pib_nacional_point_keeps_original():
    """PIB nacional (sin región) no se altera."""
    ent = ResolvedEntities(
        question="cuál es el pib del último trimestre",
        indicator_ent="pib",
        frequency_ent="q",
        region_ent=None,
        region_cls="none",
        calc_mode_cls="original",
        req_form_cls="point",
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls == "original"


def test_imacec_point_keeps_original():
    ent = ResolvedEntities(
        question="cuál es el último imacec",
        indicator_ent="imacec",
        frequency_ent="m",
        region_ent=None,
        region_cls="none",
        calc_mode_cls="original",
        req_form_cls="point",
    )
    apply_business_rules(ent)
    assert ent.indicator_ent == "imacec"
    # IMACEC no entra en la regla regional.
    assert ent.calc_mode_cls == "original"


def test_pib_sectorial_point_keeps_original():
    """PIB sectorial sin región: Rule14 NO aplica (es regional).
    Rule18_PibActivityDefaultYoY SÍ aplica → calc_mode=yoy."""
    ent = ResolvedEntities(
        question="pib de la minería el último trimestre",
        indicator_ent="pib",
        frequency_ent="q",
        activity_ent="mineria",
        activity_cls="specific",
        region_ent=None,
        region_cls="none",
        calc_mode_cls="original",
        req_form_cls="point",
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls == "yoy"


def test_pib_regional_general_keeps_original():
    """region_cls='general' (no específica): no aplica."""
    ent = ResolvedEntities(
        question="pib por región el último trimestre",
        indicator_ent="pib",
        frequency_ent="q",
        region_ent=None,
        region_cls="general",
        calc_mode_cls="original",
        req_form_cls="point",
    )
    apply_business_rules(ent)
    assert ent.calc_mode_cls == "original"


# ---------------------------------------------------------------------------
# Cambio 1 — _filter_series_by_entities filtra serie nacional cuando se pide
# una región específica.
# ---------------------------------------------------------------------------

def _multi_region_observations():
    """Replica la estructura de pib_trimestral_por_region_*_T.json:
    primera serie = nacional (sin region), seguida por series regionales."""
    return {
        "cuadro_name": "PIB trimestral por región",
        "frequency": "T",
        "series": [
            {
                "series_id": "F032.PIB.FLU.R.CLP.EP18.Z.Z.0.T",
                "short_title": "PIB",
                "classification_series": {"indicator": "pib"},
            },
            {
                "series_id": "F035.PIB.FLU.R.CLP.2018.Z.Z.Z.08.0.T",
                "short_title": "PIB Biobío",
                "classification_series": {"indicator": "pib", "region": "biobio"},
            },
            {
                "series_id": "F035.PIB.FLU.R.CLP.2018.Z.Z.Z.14.0.T",
                "short_title": "PIB Los Ríos",
                "classification_series": {"indicator": "pib", "region": "los_rios"},
            },
            {
                "series_id": "F035.PIB.FLU.R.CLP.2018.Z.Z.Z.10.0.T",
                "short_title": "PIB Los Lagos",
                "classification_series": {"indicator": "pib", "region": "los_lagos"},
            },
        ],
    }


@pytest.mark.parametrize(
    "region_ent,expected_ids",
    [
        ("biobio", {"F035.PIB.FLU.R.CLP.2018.Z.Z.Z.08.0.T"}),
        ("los_rios", {"F035.PIB.FLU.R.CLP.2018.Z.Z.Z.14.0.T"}),
        ("los_lagos", {"F035.PIB.FLU.R.CLP.2018.Z.Z.Z.10.0.T"}),
    ],
)
def test_filter_series_drops_national_when_specific_region(region_ent, expected_ids):
    obs = _multi_region_observations()
    ent = ResolvedEntities(
        indicator_ent="pib",
        region_ent=region_ent,
        region_cls="specific",
        calc_mode_cls=None,  # filtro de region debe operar incluso sin calc_mode
        seasonality_ent=None,
    )
    out = _filter_series_by_entities(obs, ent)
    got_ids = {s["series_id"] for s in out["series"]}
    assert got_ids == expected_ids, (
        f"Para region={region_ent!r} se esperaba {expected_ids}, se obtuvo {got_ids}"
    )


def test_filter_series_keeps_all_when_region_cls_none():
    """Sin región específica, el filtro de Cambio 1 NO se activa."""
    obs = _multi_region_observations()
    ent = ResolvedEntities(
        indicator_ent="pib",
        region_ent=None,
        region_cls="none",
    )
    out = _filter_series_by_entities(obs, ent)
    assert len(out["series"]) == len(obs["series"])


def test_filter_series_keeps_all_when_region_cls_general():
    obs = _multi_region_observations()
    ent = ResolvedEntities(
        indicator_ent="pib",
        region_ent=None,
        region_cls="general",
    )
    out = _filter_series_by_entities(obs, ent)
    assert len(out["series"]) == len(obs["series"])


# ---------------------------------------------------------------------------
# Cambio 2 — Orden determinista de matches (single-region antes que multi-region).
# ---------------------------------------------------------------------------

def test_search_output_payloads_prefers_single_region(tmp_path):
    import json

    # Single-region cuadro: classification.region == 'biobio'.
    single = {
        "cuadro_name": "PIB anual por actividad - Biobío",
        "frequency": "A",
        "classification": {
            "indicator": "pib",
            "frequency": "a",
            "region": "biobio",
            "has_region": 1,
        },
        "series": [{"series_id": "F035.SINGLE.BIO", "classification_series": {"region": "biobio"}}],
    }
    # Multi-region cuadro: region NO en classification, sí en series.
    multi = {
        "cuadro_name": "PIB anual por región (todas)",
        "frequency": "A",
        "classification": {
            "indicator": "pib",
            "frequency": "a",
            "has_region": 1,
        },
        "series": [
            {"series_id": "F032.NAC", "classification_series": {"indicator": "pib"}},
            {"series_id": "F035.MULTI.BIO", "classification_series": {"region": "biobio"}},
        ],
    }

    # Nombres alfabéticos para forzar que multi venga ANTES que single con
    # glob ordenado: aaa_multi.json < zzz_single.json.
    (tmp_path / "aaa_multi.json").write_text(json.dumps(multi), encoding="utf-8")
    (tmp_path / "zzz_single.json").write_text(json.dumps(single), encoding="utf-8")

    matches = search_output_payloads(
        str(tmp_path),
        indicator="pib",
        frequency="a",
        region="biobio",
    )

    assert len(matches) == 2
    # Single-region debe quedar primero pese al orden alfabético.
    assert matches[0]["payload"]["cuadro_name"] == "PIB anual por actividad - Biobío"
    assert matches[1]["payload"]["cuadro_name"] == "PIB anual por región (todas)"


def test_search_output_payloads_no_region_keeps_alphabetical(tmp_path):
    """Sin región pedida, el orden no se altera (no afecta otras consultas)."""
    import json

    a = {
        "cuadro_name": "AAA",
        "frequency": "A",
        "classification": {"indicator": "pib", "frequency": "a"},
        "series": [],
    }
    z = {
        "cuadro_name": "ZZZ",
        "frequency": "A",
        "classification": {"indicator": "pib", "frequency": "a"},
        "series": [],
    }
    (tmp_path / "aaa.json").write_text(json.dumps(a), encoding="utf-8")
    (tmp_path / "zzz.json").write_text(json.dumps(z), encoding="utf-8")

    matches = search_output_payloads(str(tmp_path), indicator="pib", frequency="a")
    names = [m["cuadro_name"] for m in matches]
    assert names == ["AAA", "ZZZ"]
