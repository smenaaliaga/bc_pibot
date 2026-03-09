import json

import orchestrator.data.get_data_serie as gds
import pandas as pd


def test_get_series_from_redis_fallback_applies_range_filter(monkeypatch):
    monkeypatch.setattr(gds, "_ensure_redis_client", lambda: None)

    def _fake_api(*, series_id, target_frequency=None, agg="avg"):
        return {
            "meta": {
                "series_id": series_id,
            },
            "observations": [
                {"date": "1996-01-31", "value": 42.0},
                {"date": "2024-01-31", "value": 100.0},
                {"date": "2024-12-31", "value": 110.0},
                {"date": "2026-01-31", "value": 120.0},
            ],
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
    assert [row["date"] for row in result["observations"]] == ["2024-01-31", "2024-12-31"]
    assert result["meta"]["period_start"] == "2024-01-31"
    assert result["meta"]["period_end"] == "2024-12-31"


def test_get_series_from_redis_uses_cache_when_requested_period_is_covered(monkeypatch):
    cached_payload = {
        "meta": {"cache_created_at": "2026-03-03T00:00:00+00:00"},
        "observations": [
            {"date": "2024-01-31", "value": 100.0},
            {"date": "2024-12-31", "value": 110.0},
        ],
    }

    class _FakeRedisClient:
        def get(self, key):
            return json.dumps(cached_payload)

    monkeypatch.setattr(gds, "_ensure_redis_client", lambda: _FakeRedisClient())

    def _should_not_call_api(**kwargs):
        raise AssertionError("No debería consultar API cuando el periodo está cubierto en cache")

    monkeypatch.setattr(gds, "get_series_api_rest_bcch", _should_not_call_api)

    result = gds.get_series_from_redis(
        series_id="F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
        firstdate="2024-01-01",
        lastdate="2024-12-31",
        target_frequency="M",
        agg="avg",
        use_fallback=True,
    )

    assert result is not None
    assert [row["date"] for row in result["observations"]] == ["2024-01-31", "2024-12-31"]
    assert result["meta"]["cache_resolution"] == "cache_covered"


def test_get_series_from_redis_refetches_when_requested_period_is_not_covered(monkeypatch):
    cached_payload = {
        "meta": {"cache_created_at": "2026-03-03T00:00:00+00:00"},
        "observations": [
            {"date": "2024-01-31", "value": 100.0},
            {"date": "2024-12-31", "value": 110.0},
        ],
    }

    class _FakeRedisClient:
        def get(self, key):
            return json.dumps(cached_payload)

    monkeypatch.setattr(gds, "_ensure_redis_client", lambda: _FakeRedisClient())

    api_calls = {"count": 0}

    def _fake_api(*, series_id, target_frequency=None, agg="avg"):
        api_calls["count"] += 1
        return {
            "meta": {"series_id": series_id},
            "observations": [
                {"date": "2027-01-31", "value": 140.0},
                {"date": "2027-12-31", "value": 160.0},
            ],
        }

    monkeypatch.setattr(gds, "get_series_api_rest_bcch", _fake_api)

    result = gds.get_series_from_redis(
        series_id="F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
        firstdate="2027-01-01",
        lastdate="2027-12-31",
        target_frequency="M",
        agg="avg",
        use_fallback=True,
    )

    assert result is not None
    assert api_calls["count"] == 1
    assert [row["date"] for row in result["observations"]] == ["2027-01-31", "2027-12-31"]
    assert result["meta"]["cache_resolution"] == "cache_missing_period"


def test_normalize_observations_empty_returns_expected_columns():
    df = gds._normalize_observations([])

    assert list(df.columns) == ["date", "value"]
    assert df.empty


def test_get_series_api_rest_bcch_empty_obs_returns_placeholder(monkeypatch):
    monkeypatch.setattr(gds, "_ensure_redis_client", lambda: None)

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "Codigo": 0,
                "Series": {
                    "seriesId": "F032.PIB.MINERIA.LOS_LAGOS.Q",
                    "descripEsp": "PIB Minería, Región de Los Lagos",
                    "Obs": [],
                },
            }

    monkeypatch.setattr(gds.requests, "get", lambda *args, **kwargs: _FakeResponse())

    result = gds.get_series_api_rest_bcch(
        series_id="F032.PIB.MINERIA.LOS_LAGOS.Q",
        target_frequency="Q",
        agg="sum",
    )

    assert isinstance(result, dict)
    assert result.get("meta", {}).get("descripEsp") == "PIB Minería, Región de Los Lagos"
    assert result.get("observations") == []


def test_resample_empty_dataframe_returns_empty_without_error():
    df = pd.DataFrame(columns=["date", "value", "status"])

    result = gds._resample(df, target_freq="Q", agg="sum", original_freq="M")

    assert result.empty
