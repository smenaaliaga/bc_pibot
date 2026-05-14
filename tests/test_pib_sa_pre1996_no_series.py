"""Tests para el mensaje informativo cuando se solicita PIB/IMACEC SA pre-1996."""
from orchestrator.data.response import build_no_series_message


def test_no_series_message_pib_sa_pre1996():
    msg = build_no_series_message(
        seasonality_unavailable={"indicator": "pib", "year": 1961, "floor_year": 1996}
    )
    low = msg.lower()
    assert "pib desestacionalizada" in low
    assert "1996" in msg
    assert "1961" in msg
    # Debe ofrecer la alternativa histórica sin desestacionalización
    assert "sin ajuste estacional" in low or "sin desestacionalizar" in low


def test_no_series_message_default_unchanged():
    msg = build_no_series_message(indicator_label="PIB")
    # El mensaje genérico NO debe mencionar la limitación SA pre-1996
    assert "desestacionalizada solo está disponible" not in msg.lower()


def test_no_series_message_imacec_sa_pre1996():
    msg = build_no_series_message(
        seasonality_unavailable={"indicator": "imacec", "year": 1990, "floor_year": 1996}
    )
    low = msg.lower()
    assert "imacec desestacionalizada" in low
    assert "1990" in msg
    assert "1996" in msg
