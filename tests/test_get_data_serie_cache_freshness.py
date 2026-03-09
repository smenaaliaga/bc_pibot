import datetime
import json

import orchestrator.data.get_data_serie as gds


class _FakeRedisClient:
    def __init__(self, payload: str):
        self._payload = payload
        self.deleted_keys = []

    def get(self, key):
        return self._payload

    def delete(self, key):
        self.deleted_keys.append(key)
        return 1


def test_get_series_from_redis_uses_cache_when_source_not_newer(monkeypatch):
    cached_payload = {
        "meta": {
            "cache_created_at": "2026-03-03T00:00:00+00:00",
        },
        "observations": [
            {"date": "2026-02-01", "value": 111.0},
        ],
    }
    fake_client = _FakeRedisClient(json.dumps(cached_payload))
    monkeypatch.setattr(gds, "_ensure_redis_client", lambda: fake_client)
    monkeypatch.setattr(
        gds,
        "_get_series_source_updated_date",
        lambda _series_id: datetime.date(2026, 3, 2),
    )

    def _should_not_call_api(**kwargs):
        raise AssertionError("No debería consultar API cuando el cache está vigente")

    monkeypatch.setattr(gds, "get_series_api_rest_bcch", _should_not_call_api)

    result = gds.get_series_from_redis(
        series_id="F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
        firstdate=None,
        lastdate=None,
        target_frequency="M",
        agg="sum",
        use_fallback=True,
    )

    assert result is not None
    assert result["observations"][0]["value"] == 111.0
    assert fake_client.deleted_keys == []


def test_get_series_from_redis_refetches_when_source_updated_after_cache(monkeypatch):
    monkeypatch.setattr(gds, "SERIES_UPDATES_ENABLED", True)
    cached_payload = {
        "meta": {
            "cache_created_at": "2026-03-01T00:00:00+00:00",
        },
        "observations": [
            {"date": "2026-02-01", "value": 100.0},
        ],
    }
    fake_client = _FakeRedisClient(json.dumps(cached_payload))
    monkeypatch.setattr(gds, "_ensure_redis_client", lambda: fake_client)
    monkeypatch.setattr(
        gds,
        "_get_series_source_updated_date",
        lambda _series_id: datetime.date(2026, 3, 2),
    )

    api_calls = {"count": 0}

    def _fake_api(*, series_id, target_frequency=None, agg="avg"):
        api_calls["count"] += 1
        return {
            "meta": {"series_id": series_id},
            "observations": [
                {"date": "2026-03-31", "value": 222.0},
            ],
        }

    monkeypatch.setattr(gds, "get_series_api_rest_bcch", _fake_api)

    result = gds.get_series_from_redis(
        series_id="F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
        firstdate=None,
        lastdate=None,
        target_frequency="M",
        agg="sum",
        use_fallback=True,
    )

    expected_key = gds._make_cache_key("F032.IMC.IND.Z.Z.EP18.Z.Z.0.M", None, None, "M", "sum")

    assert result is not None
    assert result["observations"][0]["value"] == 222.0
    assert api_calls["count"] == 1
    assert fake_client.deleted_keys == [expected_key]
