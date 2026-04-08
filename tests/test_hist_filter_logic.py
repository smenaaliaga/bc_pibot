from orchestrator.data._business_rules import ResolvedEntities, apply_business_rules
from orchestrator.data.catalog_data_search import search_output_payloads


def test_imacec_before_1996_is_clamped_and_hist_is_zero():
    ent = ResolvedEntities(indicator_ent="imacec", period_ent=["1986"])
    apply_business_rules(ent)
    assert ent.hist == 0
    assert ent.period_ent == ["1996"]
    assert ent.historical_floor_instruction is not None


def test_pib_before_1960_is_clamped_to_1960_for_point():
    ent = ResolvedEntities(indicator_ent="pib", period_ent=["1955"])
    apply_business_rules(ent)
    assert ent.period_ent == ["1960"]
    assert ent.hist == 1
    assert ent.historical_floor_instruction is not None


def test_pib_range_before_1960_clamps_only_start():
    ent = ResolvedEntities(indicator_ent="pib", period_ent=["1950-01-01", "1970-12-31"])
    apply_business_rules(ent)
    assert ent.period_ent == ["1960-01-01", "1970-12-31"]
    assert ent.hist == 1


def test_hist_default_still_zero_when_no_period_is_provided():
    ent = ResolvedEntities(indicator_ent="pib", period_ent=[])
    apply_business_rules(ent)
    assert ent.hist == 0


def test_search_with_hist_zero_matches_non_historical_payloads():
    matches = search_output_payloads(
        "orchestrator/memory/data_store",
        indicator="imacec",
        hist=0,
    )
    assert len(matches) > 0


def test_search_with_hist_one_matches_historical_payloads():
    matches = search_output_payloads(
        "orchestrator/memory/data_store",
        indicator="pib",
        hist=1,
    )
    assert len(matches) > 0
