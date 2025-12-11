import sys
import types

import pytest

from orchestrator.data import data_flow


@pytest.fixture(autouse=True)
def _clear_context():
    data_flow._last_data_context.clear()
    yield
    data_flow._last_data_context.clear()


def test_get_series_with_retry_records_error(monkeypatch):
    def _failing_fetch(**_kwargs):
        raise RuntimeError("boom")

    fake_module = types.SimpleNamespace(get_series_api_rest_bcch=_failing_fetch)
    monkeypatch.setitem(sys.modules, "orchestrator.data.get_series", fake_module)

    result = data_flow._get_series_with_retry(
        series_id="TEST.SERIE.M",
        firstdate="2024-01-01",
        lastdate="2024-12-31",
        target_frequency="M",
        retries=0,
    )

    assert result is None
    err = data_flow._last_data_context.get("fetch_error")
    assert err is not None
    assert err["series_id"] == "TEST.SERIE.M"
    assert "boom" in err["message"]


def test_build_fetch_failure_message_includes_details():
    data_flow._record_fetch_error(
        {
            "series_id": "F032.TEST.M",
            "firstdate": "2023-01-01",
            "lastdate": "2024-12-31",
            "target_frequency": "M",
            "message": "RuntimeError: boom",
        }
    )

    message = data_flow._build_fetch_failure_message()

    assert "serie F032.TEST.M" in message
    assert "rango 2023-01-01→2024-12-31" in message
    assert "detalle RuntimeError: boom" in message


def test_load_defaults_returns_real_series_code():
    defaults = data_flow._load_defaults_for_domain("IMACEC")
    assert defaults is not None
    assert defaults["cod_serie"].startswith("F032.")


def test_fetch_series_for_year_uses_cod_serie(monkeypatch):
    captured = {}

    def _fake_retry(series_id, firstdate, lastdate, target_frequency, agg="avg"):
        captured.update(
            {
                "series_id": series_id,
                "firstdate": firstdate,
                "lastdate": lastdate,
                "target_frequency": target_frequency,
            }
        )
        return {"meta": {"series_id": series_id}, "observations": [{"date": "2024-01-01"}]}

    monkeypatch.setattr(data_flow, "_get_series_with_retry", _fake_retry)

    data_flow._fetch_series_for_year("IMACEC", 2024)

    assert captured["series_id"].startswith("F032.")
    assert captured["firstdate"] == "2023-01-01"
    assert captured["lastdate"] == "2024-12-31"


def test_stream_phase_skips_duplicate_chunks(monkeypatch):
    class _FakeChain:
        def stream(self, _vars):
            yield types.SimpleNamespace(content="Hola final.")
            yield types.SimpleNamespace(content=".")

    class _FakePrompt:
        def __or__(self, _llm):
            return _FakeChain()

    classification = types.SimpleNamespace(
        query_type=None,
        data_domain="IMACEC",
        is_generic=False,
        default_key=None,
        imacec=None,
        pibe=None,
    )

    monkeypatch.setattr(data_flow, "_data_prompt", _FakePrompt())

    chunks = list(data_flow.stream_phase(classification, "Pregunta metodológica", ""))

    assert chunks == ["Hola final."]


def test_stream_data_flow_full_deduplicates_chunks(monkeypatch):
    classification = types.SimpleNamespace(
        data_domain="IMACEC",
        default_key=None,
        query_type=None,
        is_generic=False,
        imacec=None,
        pibe=None,
    )

    monkeypatch.setattr(
        data_flow,
        "_fetch_series_for_year",
        lambda *_args, **_kwargs: {"meta": {"series_id": "S"}, "observations": [1]},
    )

    def _fake_stream(*_args, **_kwargs):
        yield "Tabla base"
        yield "Duplicado final."
        yield "."

    monkeypatch.setattr(data_flow, "_stream_data_phase_with_table", _fake_stream)

    chunks = list(data_flow.stream_data_flow_full(classification, "Muéstrame el PIB 2024", ""))

    assert chunks == ["Tabla base", "Duplicado final."]


def test_stream_data_phase_emits_csv_marker_once(monkeypatch):
    data = {
        "meta": {"series_id": "S", "freq_effective": "M"},
        "observations": [
            {"date": "2024-01-01", "yoy_pct": 1.2},
            {"date": "2024-02-01", "yoy_pct": 1.0},
        ],
    }

    monkeypatch.setattr(data_flow, "_build_year_table", lambda *_: "Tabla\nA|B\n1|2")
    monkeypatch.setattr(data_flow, "_format_series_metadata_block", lambda *_: "")
    monkeypatch.setattr(data_flow, "_summarize_with_llm", lambda *_: "Resumen final.")

    marker_calls = {"count": 0}

    def _fake_marker(*_args, **_kwargs):
        marker_calls["count"] += 1
        return "##CSV_DOWNLOAD_START\npath=/tmp/fake.csv\nlabel=Descargar CSV\n##CSV_DOWNLOAD_END"

    monkeypatch.setattr(data_flow, "_emit_csv_download_marker", _fake_marker)

    classification = types.SimpleNamespace(data_domain="IMACEC")

    chunks = list(
        data_flow._stream_data_phase_with_table(
            classification,
            "Pregunta de datos",
            "",
            "IMACEC",
            2024,
            data,
        )
    )

    assert marker_calls["count"] == 1
    assert any("##CSV_DOWNLOAD_START" in chunk for chunk in chunks)
