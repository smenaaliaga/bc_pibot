from orchestrator.normalizer.normalizer import ResolvedEntities, apply_business_rules
from orchestrator.data.catalog_data_search import search_output_payloads


def test_hist_is_one_when_period_is_1996_or_older():
    ent = ResolvedEntities(period_ent=["1996"])
    apply_business_rules(ent)
    assert ent.hist == 1


def test_hist_is_zero_when_period_is_newer_than_1996():
    ent = ResolvedEntities(period_ent=["1997"])
    apply_business_rules(ent)
    assert ent.hist == 0


def test_hist_is_zero_when_no_period_is_provided():
    ent = ResolvedEntities(period_ent=[])
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
