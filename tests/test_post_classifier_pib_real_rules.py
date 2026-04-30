from orchestrator.data._business_rules import ResolvedEntities, apply_business_rules


def test_pib_real_range_forces_co_and_original_from_20260430_log():
    ent = ResolvedEntities(
        question="cuales son los valores del pib real entre 2022 y 2024",
        indicator_ent="pib",
        frequency_ent="a",
        calc_mode_cls="original",
        req_form_cls="range",
        price_ent=None,
        period_ent=["2022-01-01", "2024-12-31"],
    )

    apply_business_rules(ent)

    assert ent.indicator_ent == "pib"
    assert ent.price == "co"
    assert ent.price_ent == "co"
    assert ent.calc_mode_cls == "original"


def test_pib_real_two_latest_years_disambiguates_imacec_from_20260430_log():
    ent = ResolvedEntities(
        question="Cual es el pib real de los dos ultimos anos",
        indicator_ent="imacec",
        frequency_ent="m",
        calc_mode_cls="original",
        req_form_cls="point",
        price_ent=None,
        period_ent=["2026-03-01", "2026-03-31"],
    )

    apply_business_rules(ent)

    assert ent.indicator_ent == "pib"
    assert ent.frequency_ent == "a"
    assert ent.price == "co"
    assert ent.price_ent == "co"
    assert ent.calc_mode_cls == "original"


def test_imacec_latest_query_keeps_existing_behavior_from_20260430_log():
    ent = ResolvedEntities(
        question="cual es el valor del ultimo imacec",
        indicator_ent="imacec",
        frequency_ent="m",
        seasonality_ent="nsa",
        calc_mode_cls="original",
        req_form_cls="latest",
        price_ent=None,
    )

    apply_business_rules(ent)

    assert ent.indicator_ent == "imacec"
    assert ent.frequency_ent == "m"
    assert ent.calc_mode_cls == "original"
    assert ent.price == "enc"


def test_variacion_trimestral_pib_keeps_prev_period_rule_from_20260430_log():
    ent = ResolvedEntities(
        question="cual es la variacion trimestral del pib",
        indicator_ent="pib",
        frequency_ent="q",
        seasonality_ent="nsa",
        calc_mode_cls="yoy",
        req_form_cls="point",
        price_ent=None,
    )

    apply_business_rules(ent)

    assert ent.calc_mode_cls == "prev_period"
    assert ent.seasonality_ent == "sa"


def test_pib_precios_corrientes_forces_co_when_price_ent_is_missing():
    ent = ResolvedEntities(
        question="cual es el valor del pib a precios corrientes",
        indicator_ent="pib",
        frequency_ent="q",
        calc_mode_cls="original",
        req_form_cls="latest",
        price_ent=None,
    )

    apply_business_rules(ent)

    assert ent.price == "co"
    assert ent.price_ent == "co"
