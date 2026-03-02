import orchestrator.data.get_data_serie as gds


def test_get_series_from_redis_fallback_applies_range_filter(monkeypatch):
    monkeypatch.setattr(gds, "_ensure_redis_client", lambda: None)

    def _fake_api(*, series_id, target_date=None, target_frequency=None, agg="avg"):
        return {
            "meta": {
                "series_id": series_id,
                "firstdate": "primer dato disponible",
                "lastdate": target_date or "último dato disponible",
            },
            "observations": [
                {"date": "1996-01-01", "value": 42.0},
                {"date": "2024-01-01", "value": 100.0},
                {"date": "2024-12-01", "value": 110.0},
                {"date": "2026-01-01", "value": 120.0},
            ],
            "observations_raw": [],
        }

    monkeypatch.setattr(gds, "get_series_api_rest_bcch", _fake_api)

    result = gds.get_series_from_redis(
        series_id="F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
        firstdate="2024-01-01",
        lastdate="2024-12-31",
        target_frequency="M",
        agg="avg",
        use_fallback=True,
    )

    assert result is not None
    assert [row["date"] for row in result["observations"]] == ["2024-01-01", "2024-12-01"]
    assert result["meta"]["firstdate"] == "2024-01-01"
    assert result["meta"]["lastdate"] == "2024-12-31"
