from orchestrator.normalizer.normalizer import normalize_entities


def test_pib_no_minero_is_not_normalized():
    entities = {
        "indicator": ["pib"],
        "activity": ["no minero"],
        "period": ["durante el 2024"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="range")

    assert normalized["indicator"] == ["pib"]
    assert normalized["activity"] == []


def test_imacec_no_minero_normalizes_to_no_mineria():
    entities = {
        "indicator": ["imacec"],
        "activity": ["no minero"],
        "period": ["durante el 2024"],
    }

    normalized = normalize_entities(entities, calc_mode="yoy", req_form="range")

    assert normalized["indicator"] == ["imacec"]
    assert normalized["activity"] == ["no_mineria"]


def test_pib_regional_activity_restricts_to_regional_catalog():
    entities = {
        "indicator": ["pib"],
        "region": ["metropolitana"],
        "activity": ["agropecuario"],
        "period": ["durante el 2024"],
    }

    normalized = normalize_entities(
        entities,
        calc_mode="yoy",
        req_form="range",
        intents={"region": {"label": "specific"}},
    )

    assert normalized["indicator"] == ["pib"]
    assert normalized["region"] == ["metropolitana"]
    assert normalized["activity"] == []


def test_pib_regional_activity_allows_bienes():
    entities = {
        "indicator": ["pib"],
        "region": ["metropolitana"],
        "activity": ["bienes"],
        "period": ["durante el 2024"],
    }

    normalized = normalize_entities(
        entities,
        calc_mode="yoy",
        req_form="range",
        intents={"region": {"label": "specific"}},
    )

    assert normalized["indicator"] == ["pib"]
    assert normalized["region"] == ["metropolitana"]
    assert normalized["activity"] == ["bienes"]
