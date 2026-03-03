import orchestrator.data.response as response_module


def test_specific_response_range_single_period_avoids_redundant_from_to(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            yield "texto de prueba"

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.PIB.TEST",
            series_title="PIB",
            req_form="range",
            obs_to_show=[
                {
                    "date": "2025-03-31",
                    "value": 52975,
                    "yoy": 2.6,
                }
            ],
            parsed_point=None,
            parsed_range=("2025-01-01", "2025-03-31"),
            final_indicator_name="PIB",
            indicator_context_val="pib",
            component_context_val=None,
            seasonality_context_val="nsa",
            metric_type_val="yoy",
            calc_mode_cls="yoy",
            intent_cls="value",
            display_period_label="desde 1er trimestre 2025 hasta 1er trimestre 2025",
            freq="q",
            date_range_label="desde 1er trimestre 2025 hasta 1er trimestre 2025",
            reference_period=None,
            is_contribution=False,
            is_specific_activity=False,
            all_series_data=None,
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "desde 1er trimestre 2025 hasta 1er trimestre 2025" not in output
    assert "En 1er trimestre 2025, según los datos de la BDE, PIB registró un valor de" in output
    assert "una variación de 2,6%" in output


def test_specific_response_table_value_includes_asterisk(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            yield "texto de prueba"

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.PIB.MINERIA.TEST",
            series_title="PIB minería",
            req_form="point",
            obs_to_show=[
                {
                    "date": "2025-09-30",
                    "value": 3731,
                    "yoy": -6.5,
                }
            ],
            parsed_point="2025-09-30",
            parsed_range=None,
            final_indicator_name="PIB minería",
            indicator_context_val="pib",
            component_context_val="mineria",
            seasonality_context_val="nsa",
            metric_type_val="yoy",
            calc_mode_cls="yoy",
            intent_cls="value",
            display_period_label="3er trimestre 2025",
            freq="q",
            date_range_label="3er trimestre 2025",
            reference_period=None,
            is_contribution=False,
            is_specific_activity=True,
            all_series_data=None,
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    normalized_output = output.replace("\\", "").replace(" ", "")
    assert "3.731*" in normalized_output
    assert "3ertrimestre2025|3.731*|-6,5%" in normalized_output


def test_specific_response_contribution_announces_missing_requested_period(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            yield "texto de prueba"

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.PIB.CONTRIB.TEST",
            series_title="PIB",
            req_form="point",
            obs_to_show=[
                {
                    "date": "2024-12-31",
                    "value": 2.6,
                }
            ],
            parsed_point="2025-12-31",
            parsed_range=None,
            final_indicator_name="PIB",
            indicator_context_val="pib",
            component_context_val=None,
            seasonality_context_val="nsa",
            metric_type_val="contribution",
            calc_mode_cls="contribution",
            intent_cls="value",
            display_period_label="el año 2025",
            freq="a",
            date_range_label="el año 2025",
            reference_period=None,
            is_contribution=True,
            is_specific_activity=False,
            all_series_data=[
                {"activity": "total", "title": "PIB", "date": "2024-12-31", "value": 2.6},
                {"activity": "mineria", "title": "Minería", "date": "2024-12-31", "value": 0.5},
                {"activity": "servicios", "title": "Servicios", "date": "2024-12-31", "value": 0.3},
            ],
            used_latest_fallback_for_point=True,
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "No hay datos disponibles para el año 2025" in output
    assert "el PIB del año 2024 creció 2,6%" in output
    assert "La mayor contribución provino de la minería, con 0,5%." in output


def test_specific_response_original_general_renders_breakdown_table_with_max_growth_highlight(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            yield "texto de prueba"

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.PIB.TARGET",
            series_title="PIB",
            req_form="latest",
            obs_to_show=[
                {
                    "series_id": "SERIE.PIB.TARGET",
                    "title": "PIB",
                    "date": "2025-09-30",
                    "value": 1000.0,
                    "yoy": 1.6,
                }
            ],
            parsed_point="2025-09-30",
            parsed_range=None,
            final_indicator_name="PIB",
            indicator_context_val="pib",
            component_context_val=None,
            seasonality_context_val="nsa",
            metric_type_val="original",
            calc_mode_cls="original",
            intent_cls="value",
            display_period_label="3er trimestre 2025",
            freq="q",
            date_range_label="3er trimestre 2025",
            reference_period="2025-09-30",
            is_contribution=False,
            is_specific_activity=False,
            all_series_data=[
                {
                    "series_id": "SERIE.PIB.TARGET",
                    "title": "PIB",
                    "activity": "pib",
                    "date": "2025-09-30",
                    "value": 1000.0,
                    "yoy": 1.6,
                    "comparison_value": 1.6,
                },
                {
                    "series_id": "SERIE.COMERCIO",
                    "title": "Comercio",
                    "activity": "comercio",
                    "date": "2025-09-30",
                    "value": 250.0,
                    "yoy": 4.2,
                    "comparison_value": 4.2,
                },
                {
                    "series_id": "SERIE.INDUSTRIA",
                    "title": "Industria",
                    "activity": "industria",
                    "date": "2025-09-30",
                    "value": 180.0,
                    "yoy": 2.1,
                    "comparison_value": 2.1,
                },
            ],
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "Actividad | Valor* | Variación" in output
    assert "**Comercio** | **250**" in output
    assert "**4,2%**" in output


def test_specific_response_latest_non_contribution_mentions_top_variation_from_breakdown(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            raise RuntimeError("force fallback intro")

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.IMACEC.TARGET",
            series_title="Imacec",
            req_form="latest",
            obs_to_show=[
                {
                    "series_id": "SERIE.IMACEC.TARGET",
                    "title": "Imacec",
                    "activity": "imacec",
                    "date": "2025-12-31",
                    "value": 128,
                    "yoy": 1.7,
                }
            ],
            parsed_point="2025-12-31",
            parsed_range=None,
            final_indicator_name="IMACEC",
            indicator_context_val="imacec",
            component_context_val=None,
            seasonality_context_val="nsa",
            metric_type_val="yoy",
            calc_mode_cls="yoy",
            intent_cls="value",
            display_period_label="el último período disponible",
            freq="m",
            date_range_label="Diciembre 2025",
            reference_period="2025-12-31",
            is_contribution=False,
            is_specific_activity=False,
            all_series_data=[
                {
                    "series_id": "SERIE.IMACEC.TARGET",
                    "title": "Imacec",
                    "activity": "imacec",
                    "date": "2025-12-31",
                    "value": 128,
                    "yoy": 1.7,
                    "comparison_value": 1.7,
                },
                {
                    "series_id": "SERIE.COMERCIO",
                    "title": "Comercio",
                    "activity": "comercio",
                    "date": "2025-12-31",
                    "value": 156,
                    "yoy": 6.6,
                    "comparison_value": 6.6,
                },
                {
                    "series_id": "SERIE.SERVICIOS",
                    "title": "Servicios",
                    "activity": "servicios",
                    "date": "2025-12-31",
                    "value": 125,
                    "yoy": 2.2,
                    "comparison_value": 2.2,
                },
            ],
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "La variación mensual de la serie Imacec" in output
    assert "La mayor variación en el desglose del IMACEC la registró Comercio, con 6,6%." in output


def test_specific_response_latest_non_contribution_bypasses_llm_and_mentions_top_variation(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            yield "texto llm que no menciona desglose"

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.IMACEC.TARGET",
            series_title="Imacec",
            req_form="latest",
            obs_to_show=[
                {
                    "series_id": "SERIE.IMACEC.TARGET",
                    "title": "Imacec",
                    "activity": "imacec",
                    "date": "2025-12-31",
                    "value": 128,
                    "yoy": 1.7,
                }
            ],
            parsed_point="2025-12-31",
            parsed_range=None,
            final_indicator_name="IMACEC",
            indicator_context_val="imacec",
            component_context_val=None,
            seasonality_context_val="nsa",
            metric_type_val="yoy",
            calc_mode_cls="yoy",
            intent_cls="value",
            display_period_label="el último período disponible",
            freq="m",
            date_range_label="Diciembre 2025",
            reference_period="2025-12-31",
            is_contribution=False,
            is_specific_activity=False,
            all_series_data=[
                {
                    "series_id": "SERIE.IMACEC.TARGET",
                    "title": "Imacec",
                    "activity": "imacec",
                    "date": "2025-12-31",
                    "value": 128,
                    "yoy": 1.7,
                    "comparison_value": 1.7,
                },
                {
                    "series_id": "SERIE.COMERCIO",
                    "title": "Comercio",
                    "activity": "comercio",
                    "date": "2025-12-31",
                    "value": 156,
                    "yoy": 6.6,
                    "comparison_value": 6.6,
                },
            ],
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "texto llm que no menciona desglose" not in output
    assert "La mayor variación en el desglose del IMACEC la registró Comercio, con 6,6%." in output


def test_specific_response_latest_non_contribution_mentions_top_region_variation(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            raise RuntimeError("force deterministic intro")

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.PIB.REGIONAL.TARGET",
            series_title="PIB",
            req_form="latest",
            obs_to_show=[
                {
                    "series_id": "SERIE.PIB.REGIONAL.TARGET",
                    "title": "PIB",
                    "date": "2025-09-30",
                    "value": 1000,
                    "yoy": 1.6,
                }
            ],
            parsed_point="2025-09-30",
            parsed_range=None,
            final_indicator_name="PIB",
            indicator_context_val="pib",
            component_context_val=None,
            seasonality_context_val="nsa",
            metric_type_val="yoy",
            calc_mode_cls="yoy",
            intent_cls="value",
            display_period_label="el último período disponible",
            freq="q",
            date_range_label="3er trimestre 2025",
            reference_period="2025-09-30",
            is_contribution=False,
            is_specific_activity=False,
            all_series_data=[
                {
                    "series_id": "SERIE.PIB.REG.MAULE",
                    "title": "Maule",
                    "region": "maule",
                    "date": "2025-09-30",
                    "value": 95,
                    "yoy": 2.1,
                    "comparison_value": 2.1,
                },
                {
                    "series_id": "SERIE.PIB.REG.NUBLE",
                    "title": "Ñuble",
                    "region": "nuble",
                    "date": "2025-09-30",
                    "value": 88,
                    "yoy": 3.4,
                    "comparison_value": 3.4,
                },
            ],
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "La mayor variación en el desglose del PIB regional la registró Ñuble, con 3,4%." in output


def test_specific_response_point_fallback_non_contribution_mentions_top_region_variation(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            yield "texto llm irrelevante"

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.PIB.REGIONAL.TARGET",
            series_title="PIB",
            req_form="point",
            obs_to_show=[
                {
                    "series_id": "SERIE.PIB.REGIONAL.TARGET",
                    "title": "PIB",
                    "date": "2025-09-30",
                    "value": 1000,
                    "yoy": 1.6,
                }
            ],
            parsed_point="2025-12-31",
            parsed_range=None,
            final_indicator_name="PIB",
            indicator_context_val="pib",
            component_context_val=None,
            seasonality_context_val="nsa",
            metric_type_val="yoy",
            calc_mode_cls="yoy",
            intent_cls="value",
            display_period_label="4to trimestre 2025",
            freq="q",
            date_range_label="4to trimestre 2025",
            reference_period="2025-09-30",
            is_contribution=False,
            is_specific_activity=False,
            all_series_data=[
                {
                    "series_id": "SERIE.PIB.REG.MAULE",
                    "title": "Maule",
                    "region": "maule",
                    "date": "2025-09-30",
                    "value": 95,
                    "yoy": 2.1,
                    "comparison_value": 2.1,
                },
                {
                    "series_id": "SERIE.PIB.REG.NUBLE",
                    "title": "Ñuble",
                    "region": "nuble",
                    "date": "2025-09-30",
                    "value": 88,
                    "yoy": 3.4,
                    "comparison_value": 3.4,
                },
            ],
            used_latest_fallback_for_point=True,
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "No hay datos disponibles para 4to trimestre 2025" in output
    assert "último valor disponible corresponde a 3er trimestre 2025" in output
    assert "La mayor variación en el desglose del PIB regional la registró Ñuble, con 3,4%." in output


def test_specific_response_point_non_fallback_mentions_region_and_top_variation(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            yield "texto llm irrelevante"

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.PIB.REGIONAL.TARGET",
            series_title="PIB",
            req_form="point",
            obs_to_show=[
                {
                    "series_id": "SERIE.PIB.REGIONAL.TARGET",
                    "title": "PIB",
                    "date": "2025-09-30",
                    "value": 1000,
                    "yoy": 1.6,
                }
            ],
            parsed_point="2025-09-30",
            parsed_range=None,
            final_indicator_name="PIB de la región de Magallanes y de la Antártica Chilena",
            indicator_context_val="pib",
            component_context_val=None,
            seasonality_context_val="nsa",
            metric_type_val="yoy",
            calc_mode_cls="yoy",
            intent_cls="value",
            display_period_label="3er trimestre 2025",
            freq="q",
            date_range_label="3er trimestre 2025",
            reference_period="2025-09-30",
            is_contribution=False,
            is_specific_activity=False,
            all_series_data=[
                {
                    "series_id": "SERIE.PIB.REG.MAULE",
                    "title": "Maule",
                    "region": "maule",
                    "date": "2025-09-30",
                    "value": 95,
                    "yoy": 2.1,
                    "comparison_value": 2.1,
                },
                {
                    "series_id": "SERIE.PIB.REG.NUBLE",
                    "title": "Ñuble",
                    "region": "nuble",
                    "date": "2025-09-30",
                    "value": 88,
                    "yoy": 3.4,
                    "comparison_value": 3.4,
                },
            ],
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "el PIB de la región de Magallanes y de la Antártica Chilena presentó una variación de 1,6%" in output
    assert "La mayor variación en el desglose del PIB regional la registró Ñuble, con 3,4%." in output


def test_specific_response_point_fallback_sa_mentions_desestacionalizado(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            yield "texto llm irrelevante"

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.PIB.REGIONAL.SA.TARGET",
            series_title="PIB",
            req_form="point",
            obs_to_show=[
                {
                    "series_id": "SERIE.PIB.REGIONAL.SA.TARGET",
                    "title": "PIB",
                    "date": "2025-09-30",
                    "value": 1000,
                    "prev_period": -0.1,
                }
            ],
            parsed_point="2025-12-31",
            parsed_range=None,
            final_indicator_name="PIB desestacionalizado",
            indicator_context_val="pib",
            component_context_val=None,
            seasonality_context_val="sa",
            metric_type_val="prev_period",
            calc_mode_cls="prev_period",
            intent_cls="value",
            display_period_label="4to trimestre 2025",
            freq="q",
            date_range_label="4to trimestre 2025",
            reference_period="2025-09-30",
            is_contribution=False,
            is_specific_activity=False,
            all_series_data=[
                {
                    "series_id": "SERIE.PIB.REG.MAULE",
                    "title": "Maule",
                    "region": "maule",
                    "date": "2025-09-30",
                    "value": 95,
                    "prev_period": 2.1,
                    "comparison_value": 2.1,
                },
                {
                    "series_id": "SERIE.PIB.REG.NUBLE",
                    "title": "Ñuble",
                    "region": "nuble",
                    "date": "2025-09-30",
                    "value": 88,
                    "prev_period": 3.4,
                    "comparison_value": 3.4,
                },
            ],
            used_latest_fallback_for_point=True,
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "No hay datos disponibles para 4to trimestre 2025" in output
    assert "donde el PIB desestacionalizado presentó una variación de -0,1% respecto al período anterior." in output
    assert "La mayor variación en el desglose del PIB regional desestacionalizado la registró Ñuble, con 3,4%." in output


def test_specific_response_point_missing_variation_explains_no_comparable_prior_period(monkeypatch):
    class _FakeLLM:
        def stream(self, prompt, history=None, intent_info=None):
            yield "texto llm irrelevante"

    monkeypatch.setattr(
        response_module,
        "build_llm",
        lambda **kwargs: _FakeLLM(),
    )

    output = "".join(
        response_module.specific_response(
            series_id="SERIE.PIB.1960",
            series_title="PIB",
            req_form="point",
            obs_to_show=[
                {
                    "series_id": "SERIE.PIB.1960",
                    "title": "PIB",
                    "date": "1960-12-31",
                    "value": 19142,
                    "yoy": None,
                }
            ],
            parsed_point="1960-12-31",
            parsed_range=None,
            final_indicator_name="PIB",
            indicator_context_val="pib",
            component_context_val=None,
            seasonality_context_val="nsa",
            metric_type_val="yoy",
            calc_mode_cls="yoy",
            intent_cls="value",
            display_period_label="el año 1960",
            freq="a",
            date_range_label="el año 1960",
            reference_period="1960-12-31",
            is_contribution=False,
            is_specific_activity=False,
            all_series_data=None,
            source_urls=["https://si3.bcentral.cl/siete"],
        )
    )

    assert "En el año 1960, según los datos de la BDE" in output
    assert "No se reporta variación respecto al mismo período del año anterior" in output
    assert "porque no hay dato de referencia en la serie histórica." in output
