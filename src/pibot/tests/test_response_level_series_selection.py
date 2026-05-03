from orchestrator.data import response as response_module


def _sample_observations():
    return {
        "series": [
            {
                "series_id": "F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T",
                "short_title": "PIB a precios corrientes",
                "classification": {"indicator": "pib"},
            },
            {
                "series_id": "F033.CTO.FLU.N.CLP.EP18.0.T",
                "short_title": "Consumo total",
                "classification": {},
            },
            {
                "series_id": "F033.CPR.FLU.N.CLP.EP18.0.T",
                "short_title": "Consumo de hogares e IPSFL",
                "classification": {"investment": "consumo"},
            },
        ]
    }


def test_pick_level_target_series_prefers_household_consumption_with_hogares_hint():
    selected = response_module._pick_level_target_series(
        "Cual fue el consumo de hogares nominal en 2025?",
        {"indicator_ent": "pib", "investment_ent": "consumo"},
        _sample_observations(),
    )

    assert selected is not None
    assert selected["series_id"] == "F033.CPR.FLU.N.CLP.EP18.0.T"


def test_pick_level_target_series_prefers_household_consumption_with_cosnumo_typo():
    selected = response_module._pick_level_target_series(
        "Cual fue el cosnumo de hogares nominal en 2025?",
        {"indicator_ent": "pib", "investment_ent": None},
        _sample_observations(),
    )

    assert selected is not None
    assert selected["series_id"] == "F033.CPR.FLU.N.CLP.EP18.0.T"
