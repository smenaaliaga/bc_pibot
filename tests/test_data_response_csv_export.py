import csv
from pathlib import Path

import orchestrator.data.response as response_module


def test_flatten_observations_for_csv_includes_all_series_rows():
    observations = {
        "SERIE.A": {
            "meta": {
                "descripEsp": "Serie A",
                "original_frequency": "M",
            },
            "observations": [
                {"date": "2025-01-31", "value": 10.0, "yoy_pct": 1.0, "pct": 0.2},
                {"date": "2025-02-28", "value": 11.0, "yoy_pct": 1.5, "pct": 0.3},
            ],
        },
        "SERIE.B": {
            "meta": {
                "descripEsp": "Serie B",
                "original_frequency": "Q",
            },
            "observations": {
                "Q": [
                    {"date": "2025-03-31", "value": 20.0, "yoy_pct": 2.0, "pct": 0.4},
                ],
                "A": [
                    {"date": "2025-12-31", "value": 80.0, "yoy_pct": 3.0, "pct": 0.5},
                ],
            },
        },
    }

    rows = response_module._flatten_observations_for_csv(observations)

    assert len(rows) == 4
    assert {row["series_id"] for row in rows} == {"SERIE.A", "SERIE.B"}
    assert sum(1 for row in rows if row["series_id"] == "SERIE.A") == 2
    assert sum(1 for row in rows if row["series_id"] == "SERIE.B") == 2


def test_build_csv_marker_from_payload_exports_all_rows(monkeypatch, tmp_path):
    observations = {
        "SERIE.A": {
            "meta": {
                "descripEsp": "Serie A",
                "original_frequency": "M",
            },
            "observations": [
                {"date": "2025-01-31", "value": 10.0},
            ],
        },
        "SERIE.B": {
            "meta": {
                "descripEsp": "Serie B",
                "original_frequency": "M",
            },
            "observations": [
                {"date": "2025-01-31", "value": 20.0},
            ],
        },
    }

    captured = {}

    def _fake_export(rows, *, root_dir=None):
        captured["rows"] = list(rows)
        out = Path(tmp_path) / "export.csv"
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["series_id", "series_title", "frequency", "date", "value", "yoy_pct", "pct"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return out

    monkeypatch.setenv("DATA_RESPONSE_CSV_EXPORT_ENABLED", "1")
    monkeypatch.setattr(response_module, "_export_observations_csv", _fake_export)

    marker = response_module._build_csv_marker_from_payload({"observations": observations})

    assert marker is not None
    assert "##CSV_DOWNLOAD_START" in marker
    assert "##CSV_DOWNLOAD_END" in marker
    assert "filename=export.csv" in marker
    assert "path=" in marker
    assert len(captured["rows"]) == 2
