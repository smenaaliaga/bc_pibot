"""Tests de regresión para los tres bugs identificados en la minuta (mayo 2026).

BUG-A — "primer / más antiguo" devuelve el dato más reciente
    Reproduce: "cual fué el primer imacec" (imagen de la sesión) y variantes.
    Fix: Rule13_Historicos.apply_floor setea date_direction="earliest" cuando
         la pregunta contiene patrones de primer-dato histórico.

BUG-B1 — PIB regional con año < 1996 no encuentra la serie (hist=1 incorrecto)
    Reproduce: "pib de la araucanía desde 1990" → hist=1 → búsqueda falla.
    Fix: Rule13_Historicos._apply_pib_or_other guarda region_cls="specific" → hist=0.

BUG-B2 — "antártica" devuelve datos de Magallanes sin aclaración
    Reproduce: "pib de la antártica" → region_ent="magallanes" sin nota.
    Fix: Rule12_PibRegional.flag_antartica_standalone setea region_alias_note.
    También: _build_region_alias_note_instruction genera instrucción LLM.
"""
from __future__ import annotations

import pytest

from orchestrator.data._business_rules import ResolvedEntities, apply_business_rules
import orchestrator.data.response as response_module


# ===========================================================================
# BUG-A — date_direction="earliest" debe setearse para consultas de primer dato
# ===========================================================================

@pytest.mark.parametrize("question", [
    # Caso exacto de la imagen:
    "cual fué el primer imacec",
    "cual fue el primer imacec",
    # Variantes de "primer dato":
    "cual fue el primer pib",
    "cual fue el primer dato del pib",
    "cual fue el primer valor del imacec",
    "cual fue el primer registro del imacec",
    # "más antiguo":
    "cual fue el dato mas antiguo del imacec",
    "dame el dato mas antiguo del pib",
    "el imacec mas antiguo disponible",
    # "más viejo":
    "cual fue el imacec mas viejo",
    # "primero disponible":
    "primer dato disponible del imacec",
    "primer valor disponible del pib",
])
def test_bug_a_earliest_query_sets_date_direction(question):
    """BUG-A: consultas de primer/antiguo deben fijar date_direction='earliest'."""
    ent = ResolvedEntities(
        question=question,
        indicator_ent="imacec",
        frequency_ent="m",
        seasonality_ent="nsa",
        calc_mode_cls="original",
        req_form_cls="point",
        period_ent=[],
    )

    apply_business_rules(ent)

    assert ent.date_direction == "earliest", (
        f"Esperado date_direction='earliest' para: {question!r}\n"
        f"Obtenido: {ent.date_direction!r}"
    )


@pytest.mark.parametrize("question", [
    # Caso exacto de la imagen — respuesta INCORRECTA antes del fix:
    "cual fué el primer imacec",
    "cual fue el primer imacec",
    "cual fue el primer pib",
    "dame el dato mas antiguo del imacec",
])
def test_bug_a_earliest_query_does_not_set_hist_flag(question):
    """BUG-A: cuando date_direction='earliest' se activa, apply_floor sale antes;
    no debe setear hist=1 (que triggearía búsqueda de cuadro histórico separado)."""
    ent = ResolvedEntities(
        question=question,
        indicator_ent="imacec",
        frequency_ent="m",
        seasonality_ent="nsa",
        calc_mode_cls="original",
        req_form_cls="point",
        period_ent=[],
    )

    apply_business_rules(ent)

    assert ent.hist in (None, 0), (
        f"hist no debe ser 1 cuando date_direction='earliest': {question!r}"
    )


# ---------------------------------------------------------------------------
# Guard: "primer trimestre" / "primera región" NO deben activar earliest
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "cual fue el pib del primer trimestre de 2025",
    "pib del primer trimestre 2024",
    "primer trimestre del año 2023",
    "pib de la primera region del pais",
    "cuanto crecio el pib en el primer trimestre",
])
def test_bug_a_guard_primer_trimestre_no_activa_earliest(question):
    """El guard de EARLIEST_GUARD_RE debe impedir que 'primer trimestre'
    active date_direction='earliest'."""
    ent = ResolvedEntities(
        question=question,
        indicator_ent="pib",
        frequency_ent="q",
        calc_mode_cls="original",
        req_form_cls="point",
        period_ent=["2025-01-01", "2025-03-31"],
    )

    apply_business_rules(ent)

    assert ent.date_direction is None, (
        f"Guard falló: date_direction='earliest' NO debe activarse para: {question!r}"
    )


# ===========================================================================
# BUG-B1 — PIB regional con año antiguo no debe setear hist=1
# ===========================================================================

@pytest.mark.parametrize("question,region_ent,year_token", [
    ("pib de la araucania desde 1990", "araucania", "1990-01-01"),
    ("cual fue el pib de la region metropolitana en 1985", "metropolitana", "1985-01-01"),
    ("pib del maule de 1970 a 1980", "maule", "1970-01-01"),
    ("pib de los lagos 1960", "los_lagos", "1960-01-01"),
    ("pib regional de valparaiso desde 1950", "valparaiso", "1950-01-01"),
])
def test_bug_b1_pib_regional_antiguedad_no_activa_hist_flag(question, region_ent, year_token):
    """BUG-B1: PIB regional con year < 1996 NO debe setear hist=1.
    El cuadro PIB Regional no tiene un archivo histórico separado con hist=1."""
    ent = ResolvedEntities(
        question=question,
        indicator_ent="pib",
        frequency_ent="a",
        seasonality_ent="nsa",
        region_ent=region_ent,
        region_cls="specific",
        calc_mode_cls="original",
        req_form_cls="range",
        period_ent=[year_token],
    )

    apply_business_rules(ent)

    assert ent.hist == 0, (
        f"BUG-B1: region_cls='specific' con year antiguo no debe dar hist=1. "
        f"query={question!r} region={region_ent!r} → hist={ent.hist}"
    )


def test_bug_b1_pib_nacional_antiguo_sigue_seteando_hist():
    """Regresión: PIB NACIONAL con year < 1996 SIGUE usando hist=1 (comportamiento correcto)."""
    ent = ResolvedEntities(
        question="cual fue el pib de chile en 1980",
        indicator_ent="pib",
        frequency_ent="a",
        seasonality_ent="nsa",
        region_ent=None,
        region_cls=None,
        calc_mode_cls="original",
        req_form_cls="point",
        period_ent=["1980-01-01"],
    )

    apply_business_rules(ent)

    assert ent.hist == 1, (
        "PIB nacional con year < 1996 debe conservar hist=1 (no debe ser afectado por el fix de BUG-B1)"
    )


def test_bug_b1_pib_regional_reciente_mantiene_hist_cero():
    """PIB regional con year >= 1996 debe quedar hist=0 (sin cambio respecto a antes)."""
    ent = ResolvedEntities(
        question="pib de la araucania en 2022",
        indicator_ent="pib",
        frequency_ent="a",
        seasonality_ent="nsa",
        region_ent="araucania",
        region_cls="specific",
        calc_mode_cls="original",
        req_form_cls="point",
        period_ent=["2022-01-01"],
    )

    apply_business_rules(ent)

    assert ent.hist == 0


# ===========================================================================
# BUG-B2 — "antártica" sin "magallanes" debe setear region_alias_note
# ===========================================================================

@pytest.mark.parametrize("question", [
    "pib de la antartica",
    "cual fue el pib de la antartica en 2022",
    "crecimiento de la antartica chilena",
    "datos economicos de la antartica",
    "pib de antartica 2023",
])
def test_bug_b2_antartica_standalone_setea_alias_note(question):
    """BUG-B2: cuando region_ent='magallanes' pero la pregunta dice 'antartica'
    sin 'magallanes', debe setearse region_alias_note='antartica_standalone'."""
    ent = ResolvedEntities(
        question=question,
        indicator_ent="pib",
        frequency_ent="a",
        seasonality_ent="nsa",
        region_ent="magallanes",
        region_cls="specific",
        calc_mode_cls="original",
        req_form_cls="point",
        period_ent=["2022-01-01"],
    )

    apply_business_rules(ent)

    assert ent.region_alias_note == "antartica_standalone", (
        f"BUG-B2: esperado region_alias_note='antartica_standalone' para: {question!r}\n"
        f"Obtenido: {ent.region_alias_note!r}"
    )


@pytest.mark.parametrize("question", [
    # "magallanes" presente → no ambiguo, sin nota
    "pib de magallanes y la antartica chilena",
    "pib de magallanes",
    "crecimiento de magallanes 2022",
    "pib de la region de magallanes",
    # Otra región → no aplica
    "pib de la araucania",
    "pib del biobio",
])
def test_bug_b2_no_nota_cuando_magallanes_presente_o_otra_region(question):
    """No debe setearse region_alias_note cuando 'magallanes' aparece en la pregunta
    o cuando la región no es magallanes."""
    region = "magallanes" if "magallanes" in question or "antartica" in question else "araucania"
    ent = ResolvedEntities(
        question=question,
        indicator_ent="pib",
        frequency_ent="a",
        seasonality_ent="nsa",
        region_ent=region,
        region_cls="specific",
        calc_mode_cls="original",
        req_form_cls="point",
        period_ent=["2022-01-01"],
    )

    apply_business_rules(ent)

    assert ent.region_alias_note is None, (
        f"region_alias_note NO debe setearse para: {question!r}\n"
        f"Obtenido: {ent.region_alias_note!r}"
    )


# ===========================================================================
# response.py — _build_earliest_data_instruction
# ===========================================================================

def test_build_earliest_data_instruction_returns_text_when_date_direction_earliest():
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "earliest", "indicator_ent": "imacec"},
        observations={"first_available": {"M": "1996-01"}},
    )
    assert text is not None
    assert "primer" in text.lower() or "antiguo" in text.lower() or "histór" in text.lower() or "first_available" in text.lower() or "1996-01" in text


def test_build_earliest_data_instruction_anchors_first_available_period():
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "earliest", "indicator_ent": "imacec"},
        observations={"first_available": {"M": "1996-01"}},
    )
    assert text is not None
    assert "1996-01" in text


def test_build_earliest_data_instruction_returns_none_when_no_date_direction():
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"indicator_ent": "imacec"},
        observations={"first_available": {"M": "1996-01"}},
    )
    assert text is None


def test_build_earliest_data_instruction_returns_none_when_date_direction_is_not_earliest():
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "latest", "indicator_ent": "imacec"},
        observations={"first_available": {"M": "1996-01"}},
    )
    assert text is None


def test_build_earliest_data_instruction_fallback_when_no_first_available():
    """Cuando no hay first_available en observations, debe indicar cómo obtenerlo."""
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "earliest", "indicator_ent": "pib"},
        observations={},
    )
    assert text is not None
    assert "get_metadata" in text or "first_available" in text


# ===========================================================================
# response.py — _build_region_alias_note_instruction
# ===========================================================================

def test_build_region_alias_note_instruction_returns_text_for_antartica_standalone():
    text = response_module._build_region_alias_note_instruction(
        entities_ctx={"region_alias_note": "antartica_standalone"},
    )
    assert text is not None
    assert "Antártica" in text or "antartica" in text.lower() or "Ant\u00e1rtica" in text
    assert "Magallanes" in text or "magallanes" in text.lower()


def test_build_region_alias_note_instruction_returns_none_when_no_note():
    text = response_module._build_region_alias_note_instruction(
        entities_ctx={"region_alias_note": None},
    )
    assert text is None


def test_build_region_alias_note_instruction_returns_none_when_key_absent():
    text = response_module._build_region_alias_note_instruction(
        entities_ctx={},
    )
    assert text is None


def test_build_region_alias_note_instruction_returns_none_for_unknown_note():
    text = response_module._build_region_alias_note_instruction(
        entities_ctx={"region_alias_note": "other_note"},
    )
    assert text is None


# ===========================================================================
# Regresión de integración: pipeline completo para los casos del bug
# ===========================================================================

def test_full_pipeline_primer_imacec_imagen_caso_exacto():
    """Reproduce exactamente la consulta de la imagen: 'cual fué el primer imacec'.
    Pipeline completo debe setear date_direction='earliest' y NO date_direction=None."""
    ent = ResolvedEntities(
        question="cual fué el primer imacec",
        indicator_ent="imacec",
        frequency_ent="m",
        seasonality_ent="nsa",
        calc_mode_cls="original",
        req_form_cls="point",
        period_ent=[],
    )

    apply_business_rules(ent)

    assert ent.date_direction == "earliest"
    # No debe haber floor instruction (la regla retorna antes)
    assert ent.historical_floor_instruction is None


def test_full_pipeline_pib_araucania_1990():
    """Reproduce: 'pib de la araucanía desde 1990' → antes daba hist=1 y fallaba."""
    ent = ResolvedEntities(
        question="cual fue el pib de la araucania desde 1990",
        indicator_ent="pib",
        frequency_ent="a",
        seasonality_ent="nsa",
        region_ent="araucania",
        region_cls="specific",
        calc_mode_cls="original",
        req_form_cls="range",
        period_ent=["1990-01-01"],
    )

    apply_business_rules(ent)

    assert ent.hist == 0, "PIB regional con year < 1996 no debe activar hist=1"
    assert ent.region_ent == "araucania"


def test_full_pipeline_pib_antartica_setea_alias_note():
    """Reproduce: 'pib de la antártica' → antes devolvía datos de Magallanes sin aclaración."""
    ent = ResolvedEntities(
        question="cual fue el pib de la antartica en 2022",
        indicator_ent="pib",
        frequency_ent="a",
        seasonality_ent="nsa",
        region_ent="magallanes",
        region_cls="specific",
        calc_mode_cls="original",
        req_form_cls="point",
        period_ent=["2022-01-01"],
    )

    apply_business_rules(ent)

    assert ent.region_alias_note == "antartica_standalone"
    # La región sigue siendo magallanes (mapeo correcto del normalizer)
    assert ent.region_ent == "magallanes"


# ===========================================================================
# BUG-A v2 — Data-layer fixes (compute first_available from records;
# skip conflicting "latest" instructions and level_prefetch)
# ===========================================================================

def _make_imacec_observations(records_periods, freq="A", first_available_top=None):
    """Build a minimal observations dict like the one produced by data_node."""
    obs = {
        "series": [
            {
                "series_id": "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
                "short_title": "IMACEC",
                "data": {
                    freq: {
                        "records": [
                            {"period": p, "value": float(i + 1), "yoy_pct": None if i == 0 else 1.0}
                            for i, p in enumerate(records_periods)
                        ]
                    }
                },
            }
        ],
        "latest_available": {freq: records_periods[-1]},
    }
    if first_available_top is not None:
        obs["first_available"] = first_available_top
    return obs


def test_compute_first_available_from_observations_extracts_period_per_frequency():
    obs = _make_imacec_observations(["1996", "1997", "1998"], freq="A")
    result = response_module._compute_first_available_from_observations(obs)
    assert "A" in result
    assert result["A"]["period"] == "1996"
    assert result["A"]["value"] == 1.0


def test_compute_first_available_from_observations_empty_when_no_series():
    assert response_module._compute_first_available_from_observations({}) == {}
    assert response_module._compute_first_available_from_observations({"series": []}) == {}


def test_build_earliest_data_instruction_uses_records_when_top_level_missing():
    """BUG-A: cuando observations no expone first_available, debe computarlo de los records."""
    obs = _make_imacec_observations(["1996", "2000", "2025"], freq="A")
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "earliest", "indicator_ent": "imacec"},
        observations=obs,
    )
    assert text is not None
    assert "1996" in text, f"La instrucción debe anclar al primer período 1996. Output: {text}"
    # Debe explicitar que esta regla anula las de 'último período'
    assert "ANULA" in text.upper() or "IGNOR" in text.upper()


def test_build_earliest_data_instruction_mentions_value_of_first_record():
    obs = _make_imacec_observations(["1996", "2000"], freq="A")
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "earliest", "indicator_ent": "imacec"},
        observations=obs,
    )
    assert "value=1.0" in text or "value=1" in text


def test_build_no_explicit_period_latest_instruction_returns_none_when_earliest():
    """BUG-A: si la query es earliest, NO inyectar la regla de 'usa último período'."""
    obs = _make_imacec_observations(["1996", "2025"], freq="A")
    text = response_module._build_no_explicit_period_latest_instruction(
        question="cual fue el primer imacec",
        entities_ctx={"date_direction": "earliest", "req_form_cls": "latest"},
        observations=obs,
    )
    assert text is None, "No debe inyectar instrucción 'último período' cuando date_direction=earliest"


def test_build_level_prefetch_messages_returns_none_when_earliest():
    """BUG-A: skip level_prefetch — no precargar el record más reciente al LLM."""
    obs = _make_imacec_observations(["1996", "2025"], freq="A")
    result = response_module._build_level_prefetch_messages(
        question="cual fue el primer pib en pesos",
        entities_ctx={
            "date_direction": "earliest",
            "indicator_ent": "pib",
            "price_ent": "co",
            "req_form_cls": "latest",
            "calc_mode_cls": "original",
        },
        observations=obs,
    )
    assert result is None


# ---------------------------------------------------------------------------
# Smoke end-to-end: each failing question from run_detail.log must produce
# date_direction='earliest' AND an earliest instruction anchored at 1996.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "cual fué el primer imacec",
    "cuál es el valor del imacec más antiguo",
    "cuál es el valor del imacec historico disponible más antiguo",
    "cual fué el primer pib publicado",
])
def test_smoke_failing_queries_produce_earliest_instruction_anchored_at_records(question):
    """Smoke: las 4 preguntas que fallaban deben ahora producir instrucción anclada a 1996."""
    indicator = "imacec" if "imacec" in question.lower() else "pib"
    ent = ResolvedEntities(
        question=question,
        indicator_ent=indicator,
        region_ent=None,
        region_cls="national",
        calc_mode_cls="original",
        req_form_cls="latest",
        period_ent=[],
    )
    apply_business_rules(ent)
    assert ent.date_direction == "earliest", f"Rule layer falló para: {question!r}"

    obs = _make_imacec_observations(["1996", "2000", "2025"], freq="A")
    from dataclasses import asdict
    entities_ctx = asdict(ent)

    # 1) La instrucción 'latest' debe estar suprimida
    latest_text = response_module._build_no_explicit_period_latest_instruction(
        question=question, entities_ctx=entities_ctx, observations=obs,
    )
    assert latest_text is None, f"latest instruction NO fue suprimida para: {question!r}"

    # 2) La instrucción earliest debe estar presente y anclada en 1996
    earliest_text = response_module._build_earliest_data_instruction(entities_ctx, obs)
    assert earliest_text is not None
    assert "1996" in earliest_text, f"earliest instruction no ancla en 1996 para: {question!r}"

    # 3) level_prefetch NO debe disparar
    prefetch = response_module._build_level_prefetch_messages(question, entities_ctx, obs)
    assert prefetch is None, f"level_prefetch NO fue suprimido para: {question!r}"


# ===========================================================================
# BUG-A v3 — primer YoY disponible se debe complementar en la respuesta
# (PIB anual 1960 sin yoy → mencionar 1961 con yoy=5,5%; IMACEC mensual 1996-01
# sin yoy → mencionar 1997-01 con primera yoy disponible)
# ===========================================================================

def _make_obs_with_yoy(records, freq="A", series_id="X.SERIE", short_title="Indicador"):
    """records: list of dicts {period, value, yoy_pct}."""
    return {
        "series": [
            {
                "series_id": series_id,
                "short_title": short_title,
                "data": {freq: {"records": records}},
            }
        ],
        "latest_available": {freq: records[-1]["period"]},
    }


def test_compute_first_available_includes_first_yoy_record():
    """El helper debe exponer first_yoy_period y first_yoy_value (PIB 1960→1961)."""
    obs = _make_obs_with_yoy(
        [
            {"period": "1960", "value": 19142.33, "yoy_pct": None},
            {"period": "1961", "value": 20196.5, "yoy_pct": 5.5},
            {"period": "1962", "value": 21000.0, "yoy_pct": 4.0},
        ],
        freq="A",
        series_id="F032.PIB.FLU.R.CLP.EP18.Z.Z.0.A",
        short_title="PIB anual",
    )
    result = response_module._compute_first_available_from_observations(obs)
    assert result["A"]["period"] == "1960"
    assert result["A"]["yoy_pct"] is None
    assert result["A"]["first_yoy_period"] == "1961"
    assert result["A"]["first_yoy_value"] == 5.5


def test_compute_first_available_first_yoy_is_none_when_all_records_lack_yoy():
    obs = _make_obs_with_yoy(
        [
            {"period": "1996-01", "value": 42.5, "yoy_pct": None},
            {"period": "1996-02", "value": 43.0, "yoy_pct": None},
        ],
        freq="M",
    )
    result = response_module._compute_first_available_from_observations(obs)
    assert result["M"]["first_yoy_period"] is None
    assert result["M"]["first_yoy_value"] is None


def test_compute_first_available_first_yoy_skips_leading_nulls():
    """IMACEC: 1996-01..1996-12 sin yoy; 1997-01 primera yoy disponible."""
    records = [
        {"period": f"1996-{m:02d}", "value": 42.0 + m, "yoy_pct": None}
        for m in range(1, 13)
    ]
    records.append({"period": "1997-01", "value": 44.0, "yoy_pct": 3.5})
    records.append({"period": "1997-02", "value": 44.2, "yoy_pct": 3.8})
    obs = _make_obs_with_yoy(records, freq="M")
    result = response_module._compute_first_available_from_observations(obs)
    assert result["M"]["period"] == "1996-01"
    assert result["M"]["first_yoy_period"] == "1997-01"
    assert result["M"]["first_yoy_value"] == 3.5


def test_build_earliest_data_instruction_mentions_first_yoy_period_and_value():
    """La instrucción debe incluir first_yoy_period y first_yoy_pct cuando el
    primer record no tiene yoy pero existe yoy en records posteriores."""
    obs = _make_obs_with_yoy(
        [
            {"period": "1960", "value": 19142.33, "yoy_pct": None},
            {"period": "1961", "value": 20196.5, "yoy_pct": 5.5},
        ],
        freq="A",
        series_id="F032.PIB.FLU.R.CLP.EP18.Z.Z.0.A",
        short_title="PIB anual",
    )
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "earliest", "indicator_ent": "pib"},
        observations=obs,
    )
    assert text is not None
    # anchor del primer record
    assert "1960" in text
    # complemento con primera yoy disponible
    assert "first_yoy_period=1961" in text
    assert "first_yoy_pct=5.5" in text
    # instrucción explícita para que el LLM lo mencione (estructura 2 párrafos)
    assert "párrafo 2" in text.lower() or "parrafo 2" in text.lower()
    assert "interanual" in text.lower() or "a\u00f1o anterior" in text.lower()


def test_build_earliest_data_instruction_does_not_inject_first_yoy_when_first_record_has_yoy():
    """Si el primer record ya trae yoy_pct, no debe agregarse first_yoy_period
    como complemento redundante (la regla 3 dice 'menciónalo como secundario')."""
    obs = _make_obs_with_yoy(
        [
            {"period": "2010", "value": 100.0, "yoy_pct": 2.0},
            {"period": "2011", "value": 103.0, "yoy_pct": 3.0},
        ],
        freq="A",
    )
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "earliest", "indicator_ent": "pib"},
        observations=obs,
    )
    assert text is not None
    # value y yoy_pct del primer record están
    assert "value=100.0" in text or "value=100" in text
    assert "yoy_pct=2.0" in text or "yoy_pct=2" in text
    # NO debe inyectarse el extra dinámico first_yoy_period=YYYY (porque el
    # primer record ya trae yoy_pct). El literal "first_yoy_period" puede
    # aparecer en el texto explicativo del OBLIGATORIO, pero NO el patrón
    # "first_yoy_period=" con valor concreto.
    assert "first_yoy_period=2011" not in text
    assert "first_yoy_pct=3.0" not in text


# ===========================================================================
# BUG-A (calc_mode='prev_period'): "primer imacec desestacionalizado" debe
# reportar la PRIMERA variación pct disponible (no el level), porque el cuadro
# cargado es de Variación c/r al período anterior.
# ===========================================================================

def _make_obs_prev_period(records, freq="M", series_id="F032.IMC.IND.Z.Z.EP18.Z.Z.1.M",
                          short_title="IMACEC desestacionalizado"):
    """records: list of dicts {period, value, pct} para cuadro prev_period."""
    return {
        "series": [
            {
                "series_id": series_id,
                "short_title": short_title,
                "data": {freq: {"records": records}},
            }
        ],
        "latest_available": {freq: records[-1]["period"]},
        "classification": {"calc_mode": "prev_period", "seasonality": "sa"},
    }


def test_compute_first_available_includes_first_pct_record():
    """El helper debe exponer first_pct_period/first_pct_value para cuadros prev_period."""
    obs = _make_obs_prev_period(
        [
            {"period": "1996-01", "value": 42.0, "pct": None},
            {"period": "1996-02", "value": 42.5, "pct": 1.2},
            {"period": "1996-03", "value": 42.5, "pct": 0.0},
        ],
        freq="M",
    )
    result = response_module._compute_first_available_from_observations(obs)
    assert result["M"]["period"] == "1996-01"
    assert result["M"]["first_pct_period"] == "1996-02"
    assert result["M"]["first_pct_value"] == 1.2


def test_build_earliest_data_instruction_prev_period_uses_pct_not_value():
    """En cuadros calc_mode='prev_period', la instrucción debe anclar la respuesta
    a la PRIMERA variación pct disponible y prohibir reportar value (level)."""
    obs = _make_obs_prev_period(
        [
            {"period": "1996-01", "value": 42.0, "pct": None},
            {"period": "1996-02", "value": 42.5, "pct": 1.2},
        ],
        freq="M",
    )
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "earliest", "indicator_ent": "imacec"},
        observations=obs,
        calc_mode="prev_period",
    )
    assert text is not None
    # Debe exponer el primer pct disponible
    assert "first_pct_period=1996-02" in text
    assert "first_pct_value=1.2" in text
    # Debe instruir reportar variación respecto al período anterior
    assert "respecto al período anterior" in text or "respecto al periodo anterior" in text.lower()
    # Debe prohibir reportar value (level)
    assert "PROHIBIDO" in text and "value" in text


def test_build_earliest_data_instruction_original_mode_still_uses_value():
    """Cuando calc_mode='original' (o no se pasa), debe mantener el comportamiento
    histórico: reportar value del primer record."""
    obs = _make_obs_with_yoy(
        [
            {"period": "1996-01", "value": 42.49, "yoy_pct": None},
            {"period": "1997-01", "value": 45.13, "yoy_pct": 6.2},
        ],
        freq="M",
        series_id="F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
        short_title="IMACEC",
    )
    text = response_module._build_earliest_data_instruction(
        entities_ctx={"date_direction": "earliest", "indicator_ent": "imacec"},
        observations=obs,
        calc_mode="original",
    )
    assert text is not None
    # Mantiene la estructura de DOS PÁRRAFOS (value + first_yoy)
    assert "PÁRRAFO 1" in text
    assert "PÁRRAFO 2" in text
    assert "value=42.49" in text
    assert "first_yoy_period=1997-01" in text

