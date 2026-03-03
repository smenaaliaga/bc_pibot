import json

from orchestrator.catalog.series_search import find_family_by_classification


def test_sparse_hist_family_matches_when_hist_requested(tmp_path):
    catalog = {
        "PIB standard": {
            "classification": {
                "indicator": "pib",
                "calc_mode": "original",
                "seasonality": "nsa",
                "price": "enc",
                "has_activity": 0,
                "has_region": 0,
                "has_investment": 0,
            },
            "source_url": "standard-url",
            "series": [{"id": "STD", "classification": {"indicator": "pib"}}],
        },
        "PIB historical": {
            "classification": {
                "indicator": "pib",
                "calc_mode": "original",
                "seasonality": "nsa",
                "frequency": "a",
                "price": "enc",
                "has_activity": 0,
                "has_region": 0,
                "has_investment": 0,
                "hist": 1,
            },
            "source_url": "hist-url",
            "series": [{"id": "HIST", "classification": {"indicator": "pib"}}],
        },
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    family = find_family_by_classification(
        catalog_path,
        indicator="pib",
        activity_value=None,
        region_value=None,
        investment_value=None,
        calc_mode="original",
        price=None,
        seasonality=None,
        frequency=None,
        hist=1,
    )

    assert family is not None
    assert family.get("family_name") == "PIB historical"
    assert family.get("source_url") == "hist-url"


def test_hist_none_prefers_families_without_hist_key(tmp_path):
    catalog = {
        "PIB standard": {
            "classification": {
                "indicator": "pib",
                "calc_mode": "original",
                "seasonality": "nsa",
                "price": "enc",
                "has_activity": 0,
                "has_region": 0,
                "has_investment": 0,
            },
            "source_url": "standard-url",
            "series": [{"id": "STD", "classification": {"indicator": "pib"}}],
        },
        "PIB historical": {
            "classification": {
                "indicator": "pib",
                "calc_mode": "original",
                "seasonality": "nsa",
                "frequency": "a",
                "price": "enc",
                "has_activity": 0,
                "has_region": 0,
                "has_investment": 0,
                "hist": 1,
            },
            "source_url": "hist-url",
            "series": [{"id": "HIST", "classification": {"indicator": "pib"}}],
        },
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    family = find_family_by_classification(
        catalog_path,
        indicator="pib",
        activity_value=None,
        region_value=None,
        investment_value=None,
        calc_mode="original",
        price=None,
        seasonality=None,
        frequency=None,
        hist=None,
    )

    assert family is not None
    assert family.get("family_name") == "PIB standard"
    assert family.get("source_url") == "standard-url"


def test_family_region_value_selects_matching_region(tmp_path):
    catalog = {
        "PIB región Arica": {
            "classification": {
                "indicator": "pib",
                "calc_mode": "original",
                "seasonality": "nsa",
                "frequency": "q",
                "price": "enc",
                "has_activity": 1,
                "has_region": 1,
                "has_investment": 0,
                "region": "arica_parinacota",
            },
            "source_url": "arica-url",
            "series": [{"id": "ARICA.SERIE", "classification": {"activity": "mineria"}}],
        },
        "PIB región Metropolitana": {
            "classification": {
                "indicator": "pib",
                "calc_mode": "original",
                "seasonality": "nsa",
                "frequency": "q",
                "price": "enc",
                "has_activity": 1,
                "has_region": 1,
                "has_investment": 0,
                "region": "metropolitana",
            },
            "source_url": "metro-url",
            "series": [{"id": "METRO.SERIE", "classification": {"activity": "mineria"}}],
        },
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    family = find_family_by_classification(
        catalog_path,
        indicator="pib",
        activity_value="mineria",
        region_value="metropolitana",
        investment_value=None,
        calc_mode="original",
        price="enc",
        seasonality="nsa",
        frequency="q",
        hist=None,
    )

    assert family is not None
    assert family.get("family_name") == "PIB región Metropolitana"
    assert family.get("source_url") == "metro-url"
