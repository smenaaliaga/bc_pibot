from __future__ import annotations

from orchestrator.catalog.catalog_lookup import (
    find_family_by_classification,
    select_target_series_by_classification,
)


def test_select_target_series_prefers_price_when_series_define_it():
    family_series = [
        {
            "id": "SER.ENC",
            "classification": {"indicator": "pib", "seasonality": "nsa", "price": "enc"},
        },
        {
            "id": "SER.CO",
            "classification": {"indicator": "pib", "seasonality": "nsa", "price": "co"},
        },
    ]

    selected = select_target_series_by_classification(
        family_series,
        eq={"indicator": "pib", "seasonality": "nsa", "price": "co"},
        fallback_to_first=True,
    )

    assert isinstance(selected, dict)
    assert selected.get("id") == "SER.CO"


def test_select_target_series_ignores_price_filter_when_not_present_in_rows():
    family_series = [
        {
            "id": "SER.MINERIA",
            "classification": {"indicator": "pib", "activity": "mineria"},
        },
        {
            "id": "SER.INDUSTRIA",
            "classification": {"indicator": "pib", "activity": "industria"},
        },
    ]

    selected = select_target_series_by_classification(
        family_series,
        eq={"indicator": "pib", "activity": "mineria", "price": "co"},
        fallback_to_first=True,
    )

    assert isinstance(selected, dict)
    assert selected.get("id") == "SER.MINERIA"


def test_find_family_by_classification_prefers_per_capita_activity_match():
    family = find_family_by_classification(
        "orchestrator/catalog/catalog.json",
        indicator="pib",
        activity_value="pib_percapita",
        region_value=None,
        investment_value=None,
        calc_mode="yoy",
        price=None,
        seasonality="nsa",
        frequency=None,
        hist=None,
    )

    assert isinstance(family, dict)
    assert "per capita" in str(family.get("family_name") or "").lower()
