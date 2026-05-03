from orchestrator.normalizer.normalizer import normalize_entities, normalize_region


def test_normalize_region_accepts_bernardo_ohiggins_alias():
    normalized, failed = normalize_region("bernardo ohiggins")

    assert normalized == "ohiggins"
    assert failed == []


def test_normalize_entities_keeps_region_for_bernardo_ohiggins_alias():
    entities = {
        "indicator": ["pib"],
        "seasonality": ["nsa"],
        "frequency": ["q"],
        "activity": [],
        "region": ["bernardo ohiggins"],
        "investment": [],
        "period": ["2025-10-01", "2025-12-31"],
    }

    normalized = normalize_entities(
        entities,
        calc_mode="original",
        req_form="point",
        intents={"region": {"label": "specific"}},
    )

    assert normalized["region"] == ["ohiggins"]


def test_normalize_entities_forces_nsa_for_regional_pib_with_explicit_region():
    entities = {
        "indicator": ["pib"],
        "seasonality": ["desestacionalizado"],
        "frequency": ["q"],
        "activity": ["mineria"],
        "region": ["metropolitana"],
        "investment": [],
        "period": ["2025-10-01", "2025-12-31"],
    }

    normalized = normalize_entities(
        entities,
        calc_mode="original",
        req_form="point",
        intents={"region": {"label": "specific"}},
    )

    assert normalized["indicator"] == ["pib"]
    assert normalized["seasonality"] == ["nsa"]


def test_normalize_entities_forces_nsa_for_regional_pib_with_region_intent_only():
    entities = {
        "indicator": ["pib"],
        "seasonality": ["desestacionalizado"],
        "frequency": ["q"],
        "activity": ["bienes"],
        "region": [],
        "investment": [],
        "period": ["2025-10-01", "2025-12-31"],
    }

    normalized = normalize_entities(
        entities,
        calc_mode="original",
        req_form="point",
        intents={"region": {"label": "general"}},
    )

    assert normalized["indicator"] == ["pib"]
    assert normalized["seasonality"] == ["nsa"]


def test_normalize_entities_infers_pib_when_region_present_without_indicator():
    entities = {
        "indicator": [],
        "seasonality": [],
        "frequency": [],
        "activity": [],
        "region": ["metropolitana"],
        "investment": [],
        "period": ["2025-10-01", "2025-12-31"],
    }

    normalized = normalize_entities(
        entities,
        calc_mode="original",
        req_form="point",
    )

    assert normalized["indicator"] == ["pib"]


def test_normalize_entities_infers_pib_when_investment_present_without_indicator():
    entities = {
        "indicator": [],
        "seasonality": [],
        "frequency": [],
        "activity": [],
        "region": [],
        "investment": ["inversion"],
        "period": ["2025-10-01", "2025-12-31"],
    }

    normalized = normalize_entities(
        entities,
        calc_mode="original",
        req_form="point",
    )

    assert normalized["indicator"] == ["pib"]


def test_normalize_entities_treats_unknown_indicator_as_generic_imacec_case():
    entities = {
        "indicator": ["foo"],
        "seasonality": [],
        "frequency": [],
        "activity": ["mineria"],
        "region": [],
        "investment": [],
        "period": [],
    }

    normalized = normalize_entities(
        entities,
        calc_mode="prev_period",
        req_form="latest",
        intents={
            "activity": {"label": "specific"},
            "region": {"label": "none"},
            "investment": {"label": "none"},
        },
    )

    assert normalized["indicator"] == ["imacec"]
    assert normalized["frequency"] == ["m"]


def test_normalize_entities_treats_bde_as_generic_imacec_case():
    entities = {
        "indicator": ["BDE"],
        "seasonality": [],
        "frequency": [],
        "activity": ["mineria"],
        "region": [],
        "investment": [],
        "period": [],
    }

    normalized = normalize_entities(
        entities,
        calc_mode="prev_period",
        req_form="latest",
        intents={
            "activity": {"label": "specific"},
            "region": {"label": "none"},
            "investment": {"label": "none"},
        },
    )

    assert normalized["indicator"] == ["imacec"]
    assert normalized["frequency"] == ["m"]
