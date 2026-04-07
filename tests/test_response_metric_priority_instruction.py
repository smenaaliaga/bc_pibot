import orchestrator.data.response as response_module
from orchestrator.data._helpers import build_target_series_url
from pathlib import Path


def test_build_metric_priority_instruction_for_yoy():
    text = response_module._build_metric_priority_instruction("yoy")
    assert text is not None
    assert "'yoy_pct'" in text
    assert "PERIODO analizado" in text
    assert "No comiences con 'value'" in text


def test_build_metric_priority_instruction_for_prev_period():
    text = response_module._build_metric_priority_instruction("prev_period")
    assert text is not None
    assert "'pct'" in text
    assert "PERIODO analizado" in text
    assert "No comiences con 'value'" in text


def test_build_metric_priority_instruction_for_original():
    text = response_module._build_metric_priority_instruction("original")
    assert text is not None
    assert "'value'" in text
    assert "PERIODO analizado" in text
    assert "No comiences con 'yoy_pct'" in text


def test_build_metric_priority_instruction_for_unknown_mode_returns_none():
    assert response_module._build_metric_priority_instruction("share") is None
    assert response_module._build_metric_priority_instruction("") is None


def test_build_metric_priority_instruction_for_contribution_enforces_neutral_negative_wording():
    text = response_module._build_metric_priority_instruction("contribution")
    assert text is not None
    assert "MATRIZ DE SINÓNIMOS ESTILO INFORME IMACEC" in text
    assert "NUNCA 'pp'" in text
    assert "valor absoluto" in text
    assert "CHEQUEO FINAL OBLIGATORIO" in text
    assert "'-X,X%'" in text
    assert "contribuciones negativas de" in text
    assert "PLANTILLA OBLIGATORIA POR ACTIVIDAD" in text


def test_contribution_activity_focus_instruction_forbids_generic_negative_summary():
    text = response_module._build_contribution_activity_focus_instruction(
        question="que actividad afecto al aumento del pib de la region del bio bio",
        entities_ctx={"calc_mode_cls": "contribution"},
        observations={"latest_available": {"T": "2025-Q3"}},
    )

    assert text is not None
    assert "PROHIBIDO resumir como 'contribuciones negativas de varias actividades'" in text
    assert "plantilla obligatoria" in text.lower()


def test_prevalidated_missing_specific_activity_instruction_when_activity_normalized_empty():
    text = response_module._build_prevalidated_missing_specific_activity_instruction(
        entities_ctx={
            "activity_cls": "specific",
            "indicator_ent": "pib",
            "activity_ent": "",
        },
        observations={
            "series": [
                {"classification_series": {"activity": "Producción de bienes"}},
                {"classification_series": {"activity": "Servicios"}},
            ]
        },
    )

    assert text is not None
    assert "VALIDACIÓN PREVIA DEL SISTEMA" in text
    assert "En [PERIODO], esta actividad no se encuentra disponible en la Base de datos estadísticos para el indicador PIB" in text
    assert "debe aparecer una sola vez" in text
    assert "actividad más similar disponible" in text
    assert "Producción de bienes" in text
    assert "Servicios" in text
    assert "NO escribas una introducción que afirme contribución" in text


def test_prevalidated_missing_specific_activity_instruction_none_when_activity_present():
    text = response_module._build_prevalidated_missing_specific_activity_instruction(
        entities_ctx={
            "activity_cls": "specific",
            "indicator_ent": "pib",
            "activity_ent": "mineria",
        },
        observations={"series": []},
    )

    assert text is None


def test_humanize_activity_label_removes_underscores_and_hyphens():
    assert response_module._humanize_activity_label("admin_publica") == "admin publica"
    assert response_module._humanize_activity_label("servicios-empresariales") == "servicios empresariales"


def test_missing_activity_instruction_requests_natural_activity_names():
    text = response_module._build_missing_activity_instruction(
        entities_ctx={"activity_ent": "mineria", "activity_cls": "specific", "indicator_ent": "pib"},
        observations={
            "series": [
                {"classification_series": {"activity": "admin_publica"}},
                {"classification_series": {"activity": "comercio"}},
            ]
        },
    )

    assert text is not None
    assert "admin publica" in text
    assert "admin_publica" not in text
    assert "sin guiones bajos" in text


def test_build_incomplete_period_instruction_for_incomplete_quarterly_year():
    text = response_module._build_incomplete_period_instruction(
        question="cuanto crecio el pib de la region metropolitana en 2025",
        entities_ctx={"frequency_ent": "q", "period_ent": ["2025"]},
        observations={"latest_available": {"T": "2025-Q3"}, "classification": {"frequency": "q"}},
    )
    assert text is not None
    assert "NO está cerrado" in text
    assert "Hasta 3er trimestre de 2025" in text or "hasta 3er trimestre de 2025" in text
    assert "No escribas frases como 'En 2025, ...'" in text


def test_build_incomplete_period_instruction_for_incomplete_first_semester():
    text = response_module._build_incomplete_period_instruction(
        question="como estuvo el primer semestre de 2025",
        entities_ctx={"frequency_ent": "q", "period_ent": []},
        observations={"latest_available": {"T": "2025-Q1"}, "classification": {"frequency": "q"}},
    )
    assert text is not None
    assert "semestre solicitado NO está completo" in text
    assert "primer semestre de 2025" in text
    assert "Hasta 1er trimestre de 2025" in text or "hasta 1er trimestre de 2025" in text


def test_build_incomplete_period_instruction_returns_none_for_complete_year():
    text = response_module._build_incomplete_period_instruction(
        question="cuanto crecio el pib en 2024",
        entities_ctx={"frequency_ent": "q", "period_ent": ["2024"]},
        observations={"latest_available": {"T": "2024-Q4"}, "classification": {"frequency": "q"}},
    )
    assert text is None


def test_build_incomplete_period_instruction_uses_frequency_from_entities_ctx():
    text = response_module._build_incomplete_period_instruction(
        question="como estuvo el primer semestre de 2025",
        entities_ctx={"frequency_ent": "q", "period_ent": []},
        observations={"latest_available": {"T": "2025-Q1"}, "classification": {"frequency": "a"}},
    )
    assert text is not None
    assert "semestre solicitado NO está completo" in text


def test_relative_period_fallback_instruction_when_last_year_has_no_data():
    year = response_module.date.today().year
    text = response_module._build_relative_period_fallback_instruction(
        question="cuanto crecio la mineria el año pasado",
        entities_ctx={"frequency_ent": "a"},
        observations={"latest_available": {"A": str(year - 2)}},
    )
    assert text is not None
    assert "sin datos disponibles" in text
    assert str(year - 1) in text
    assert str(year - 2) in text


def test_relative_period_fallback_instruction_none_when_last_year_exists():
    year = response_module.date.today().year
    text = response_module._build_relative_period_fallback_instruction(
        question="cuanto crecio la mineria el año pasado",
        entities_ctx={"frequency_ent": "a"},
        observations={"latest_available": {"A": str(year - 1)}},
    )
    assert text is None


def test_missing_activity_instruction_when_requested_activity_not_available():
    text = response_module._build_missing_activity_instruction(
        entities_ctx={"activity_ent": "mineria", "activity_cls": "specific"},
        observations={
            "series": [
                {"classification_series": {"activity": "industria"}},
                {"classification_series": {"activity": "servicios"}},
                {"classification_series": {"indicator": "pib"}},
            ]
        },
    )
    assert text is not None
    assert "actividad solicitada no está disponible" in text
    assert "debe aparecer una sola vez" in text
    assert "actividad más similar disponible" in text
    assert "NO reemplaces por la serie agregada" in text


def test_missing_activity_instruction_none_when_activity_exists():
    text = response_module._build_missing_activity_instruction(
        entities_ctx={"activity_ent": "industria"},
        observations={
            "series": [
                {"classification_series": {"activity": "industria"}},
                {"classification_series": {"activity": "servicios"}},
            ]
        },
    )
    assert text is None


def test_missing_activity_instruction_none_for_imacec_indicator_token():
    text = response_module._build_missing_activity_instruction(
        entities_ctx={"activity_ent": "imacec", "activity_cls": "specific"},
        observations={
            "series": [
                {"classification_series": {"activity": "industria"}},
                {"classification_series": {"activity": "servicios"}},
            ]
        },
    )
    assert text is None


def test_missing_activity_instruction_none_for_pib_indicator_token():
    text = response_module._build_missing_activity_instruction(
        entities_ctx={"activity_ent": "pib", "activity_cls": "specific"},
        observations={
            "series": [
                {"classification_series": {"activity": "industria"}},
                {"classification_series": {"activity": "servicios"}},
            ]
        },
    )
    assert text is None


def test_no_explicit_period_instruction_when_question_has_no_date():
    text = response_module._build_no_explicit_period_latest_instruction(
        question="dame la contribucion del sector minero a la economia",
        entities_ctx={"frequency_ent": "m"},
        observations={"latest_available": {"M": "2026-01"}},
    )
    assert text is not None
    assert "no especifica fecha" in text
    assert "ultimo periodo disponible" in text
    assert "NO menciones falta de datos" in text


def test_no_explicit_period_instruction_none_when_relative_date_is_explicit():
    text = response_module._build_no_explicit_period_latest_instruction(
        question="dame la contribucion del sector minero el mes pasado",
        entities_ctx={"frequency_ent": "m"},
        observations={"latest_available": {"M": "2026-01"}},
    )
    assert text is None


def test_no_explicit_period_instruction_when_req_form_latest_even_with_relative_date():
    text = response_module._build_no_explicit_period_latest_instruction(
        question="dame la contribucion del sector minero el mes pasado",
        entities_ctx={"frequency_ent": "m", "req_form_cls": "latest"},
        observations={"latest_available": {"M": "2026-01"}},
    )
    assert text is not None
    assert "req_form_cls='latest'" in text


def test_no_explicit_period_instruction_none_for_current_prices_original_query():
    text = response_module._build_no_explicit_period_latest_instruction(
        question="Cual es el valor del pib en pesos",
        entities_ctx={"frequency_ent": "q", "price_ent": "co"},
        observations={"latest_available": {"T": "2025-Q4"}},
    )
    assert text is None


def test_no_explicit_period_instruction_none_for_per_capita_original_query():
    text = response_module._build_no_explicit_period_latest_instruction(
        question="Cual es el pib per capita",
        entities_ctx={"frequency_ent": "a", "indicator_ent": "pib_per_capita"},
        observations={"latest_available": {"A": "2024"}},
    )
    assert text is None


def test_relative_period_fallback_instruction_none_when_req_form_latest():
    year = response_module.date.today().year
    text = response_module._build_relative_period_fallback_instruction(
        question="dame la contribucion del sector minero el mes pasado",
        entities_ctx={"frequency_ent": "m", "req_form_cls": "latest"},
        observations={"latest_available": {"M": f"{year}-01"}},
    )
    assert text is None


def test_build_csv_markers_returns_single_marker_for_multiple_fetched_series():
    marker_block = response_module._build_csv_markers(
        [
            {
                "series_id": "SERIE_A",
                "short_title": "Serie A",
                "frequency": "M",
                "records": [
                    {"period": "2025-01", "value": 10, "pct": "1.0%", "yoy_pct": "2.0%"},
                ],
            },
            {
                "series_id": "SERIE_B",
                "short_title": "Serie B",
                "frequency": "M",
                "records": [
                    {"period": "2025-01", "value": 20, "pct": "3.0%", "yoy_pct": "4.0%"},
                ],
            },
        ],
        cuadro_name="Cuadro de prueba",
    )

    assert marker_block.count("##CSV_DOWNLOAD_START") == 1
    assert "filename=serie_SERIE_A.csv" in marker_block
    assert "title=Serie A" in marker_block

    path_line = next(line for line in marker_block.splitlines() if line.startswith("path="))
    export_path = Path(path_line.split("=", 1)[1])
    export_text = export_path.read_text(encoding="utf-8-sig")

    assert "SERIE_A" in export_text
    assert "SERIE_B" not in export_text
    assert "Cuadro de prueba" in export_text


def test_list_series_exposes_available_frequencies():
    payload = {
        "series": [
            {
                "series_id": "SERIE.T",
                "short_title": "Serie trimestral",
                "long_title": "Serie trimestral larga",
                "classification_series": {"seasonality": "nsa"},
                "series_freq": "T",
                "data": {"T": {"records": []}, "A": {"records": []}},
            }
        ]
    }

    result = response_module.handle_tool_call("list_series", {}, payload)
    parsed = response_module.json.loads(result)

    assert parsed[0]["series_id"] == "SERIE.T"
    assert parsed[0]["available_frequencies"] == ["T", "A"]


def test_get_series_data_unknown_series_returns_hint_and_catalog():
    payload = {
        "series": [
            {
                "series_id": "SERIE.T",
                "short_title": "Serie trimestral",
                "series_freq": "T",
                "data": {"T": {"records": []}, "A": {"records": []}},
            }
        ]
    }

    result = response_module.handle_tool_call(
        "get_series_data",
        {"series_id": "SERIE.A", "frequency": "A"},
        payload,
    )
    parsed = response_module.json.loads(result)

    assert "no encontrada en este cuadro" in parsed["error"]
    assert "Usa list_series" in parsed["hint"]
    assert parsed["available_series"][0]["series_id"] == "SERIE.T"
    assert parsed["available_series"][0]["available_frequencies"] == ["T", "A"]


def test_build_filtered_source_url_uses_none_for_original_current_prices_query():
    observations = {
        "source_url": "https://example.test/series",
        "classification": {"calc_mode": "original"},
        "series": [
            {
                "series_id": "SERIE.PIB.CO",
                "short_title": "PIB precios corrientes",
                "data": {
                    "A": {
                        "records": [
                            {"period": "2024", "value": 100.0, "pct": 1.2, "yoy_pct": 3.4},
                        ]
                    }
                },
            }
        ],
    }
    entities_ctx = {
        "calc_mode_cls": "original",
        "price_ent": "co",
        "question": "cuanto fue el pib en pesos",
        "period_ent": ["2024-01-01", "2024-12-31"],
    }
    selected_series_ctx = {"series_id": "SERIE.PIB.CO", "frequency": "A"}

    url = response_module._build_filtered_source_url(observations, entities_ctx, selected_series_ctx)

    assert isinstance(url, str)
    assert "cbCalculo=NONE" in url


def test_build_target_series_url_supports_explicit_none_mode():
    url = response_module.build_target_series_url(
        source_url="https://example.test/series",
        series_id="SERIE.PIB.TEST",
        period=["2024-01-01", "2024-12-31"],
        observations=[{"date": "2024-12-31", "value": 100.0}],
        frequency="a",
        calc_mode="none",
    )

    assert isinstance(url, str)
    assert "cbCalculo=NONE" in url


def test_build_target_series_url_original_defaults_to_ytypct():
    url = build_target_series_url(
        source_url="https://example.test/series",
        series_id="SERIE.PIB.TEST",
        period=["2024-01-01", "2024-12-31"],
        observations=[{"date": "2024-12-31", "value": 100.0, "yoy_pct": 3.4}],
        frequency="a",
        calc_mode="original",
    )

    assert isinstance(url, str)
    assert "cbCalculo=YTYPCT" in url
    assert "cbCalculo=PCT" not in url


def test_build_target_series_url_prev_period_explicit_uses_pct():
    url = build_target_series_url(
        source_url="https://example.test/series",
        series_id="SERIE.PIB.TEST",
        period=["2024-01-01", "2024-12-31"],
        observations=[{"date": "2024-12-31", "value": 100.0, "pct": 1.2}],
        frequency="a",
        calc_mode="prev_period",
    )

    assert isinstance(url, str)
    assert "cbCalculo=PCT" in url


def test_build_target_series_url_unknown_mode_defaults_to_ytypct():
    url = build_target_series_url(
        source_url="https://example.test/series",
        series_id="SERIE.PIB.TEST",
        period=["2024-01-01", "2024-12-31"],
        observations=[{"date": "2024-12-31", "value": 100.0, "yoy_pct": 3.4}],
        frequency="a",
        calc_mode="share",
    )

    assert isinstance(url, str)
    assert "cbCalculo=YTYPCT" in url


def test_build_target_series_url_contribution_homologates_link_format():
    url = build_target_series_url(
        source_url="https://example.test/cuadro",
        series_id="SERIE.CONTRIB.TEST",
        period=["2020-01-01", "2020-12-31"],
        observations=[{"date": "2020-12-31", "value": 1.2}],
        frequency="q",
        calc_mode="contribution",
    )

    assert isinstance(url, str)
    assert "cbFechaInicio=2020" in url
    assert "cbFechaTermino=2025" in url
    assert "cbFrecuencia=QUARTERLY" in url
    assert "cbCalculo=NONE" in url
    assert "cbFechaBase=" in url
    assert "id5=SI" not in url
    assert "idSerie=" not in url


def test_build_target_series_url_contribution_keeps_requested_period_when_observed_differs():
    url = build_target_series_url(
        source_url="https://example.test/cuadro",
        series_id="SERIE.CONTRIB.TEST",
        period=["2019-01-01", "2020-12-31"],
        observations=[{"date": "2024-12-31", "value": 1.2}],
        frequency="q",
        calc_mode="contribution",
        req_form="latest",
    )

    assert isinstance(url, str)
    assert "cbFechaInicio=2019" in url
    assert "cbFechaTermino=2020" in url


def test_build_target_series_url_contribution_same_start_end_forces_end_2025():
    url = build_target_series_url(
        source_url="https://example.test/cuadro",
        series_id="SERIE.CONTRIB.TEST",
        period=["2020-01-01", "2020-12-31"],
        observations=[{"date": "2020-12-31", "value": 1.2}],
        frequency="q",
        calc_mode="contribution",
        req_form="range",
    )

    assert isinstance(url, str)
    assert "cbFechaInicio=2020" in url
    assert "cbFechaTermino=2025" in url


def test_build_target_series_url_inverted_range_forces_end_year_2025():
    url = build_target_series_url(
        source_url="https://example.test/series",
        series_id="SERIE.PIB.TEST",
        period=["2027-01-01", "2024-12-31"],
        observations=[{"date": "2024-12-31", "value": 100.0, "yoy_pct": 3.4}],
        frequency="a",
        calc_mode="yoy",
    )

    assert isinstance(url, str)
    assert "cbFechaTermino=2025" in url
    assert "cbFechaInicio=2015" in url


def test_build_filtered_source_url_uses_none_for_original_per_capita_query():
    observations = {
        "source_url": "https://example.test/series",
        "classification": {"calc_mode": "original"},
        "series": [
            {
                "series_id": "SERIE.PIB.PERCAPITA",
                "short_title": "PIB per cápita",
                "data": {
                    "A": {
                        "records": [
                            {"period": "2024", "value": 16586.0, "pct": 0.9, "yoy_pct": 2.1},
                        ]
                    }
                },
            }
        ],
    }
    entities_ctx = {
        "calc_mode_cls": "original",
        "indicator_ent": "pib_per_capita",
        "question": "cual fue el pib per capita",
        "period_ent": ["2024-01-01", "2024-12-31"],
    }
    selected_series_ctx = {"series_id": "SERIE.PIB.PERCAPITA", "frequency": "A"}

    url = response_module._build_filtered_source_url(observations, entities_ctx, selected_series_ctx)

    assert isinstance(url, str)
    assert "cbCalculo=NONE" in url


def test_special_query_mapping_instruction_for_per_capita_forces_single_original_value():
    text = response_module._build_special_query_mapping_instruction(
        question="cual fue el pib per capita",
        entities_ctx={},
    )

    assert text is not None
    assert "SIEMPRE valor original (value)" in text
    assert "nunca yoy_pct ni pct" in text
    assert "REGLA DE UN SOLO VALOR" in text


def test_special_query_mapping_instruction_for_current_prices_forces_single_original_value():
    text = response_module._build_special_query_mapping_instruction(
        question="cual fue el pib a precios corrientes",
        entities_ctx={},
    )

    assert text is not None
    assert "CASO PIB EN PESOS / PRECIOS CORRIENTES" in text
    assert "SIEMPRE valor original (value)" in text
    assert "nunca yoy_pct ni pct" in text
    assert "REGLA DE UN SOLO VALOR" in text


def test_special_query_mapping_instruction_detects_current_prices_from_entities_ctx():
    text = response_module._build_special_query_mapping_instruction(
        question="cual fue el pib",
        entities_ctx={"price_ent": "co"},
    )

    assert text is not None
    assert "CASO PIB EN PESOS / PRECIOS CORRIENTES" in text
    assert "SIEMPRE valor original (value)" in text


def test_value_no_rescale_instruction_forces_original_value_for_current_prices_ctx():
    text = response_module._build_value_no_rescale_instruction(
        question="cual fue el pib",
        entities_ctx={"price_ent": "co"},
    )

    assert text is not None
    assert "solo value (serie original)" in text
    assert "no uses yoy_pct ni pct" in text


def test_original_series_force_instruction_for_current_prices_by_entities_ctx():
    text = response_module._build_original_series_force_instruction(
        question="cual fue el pib",
        entities_ctx={"price_ent": "co"},
    )

    assert text is not None
    assert "SERIE ORIGINAL OBLIGATORIA" in text
    assert "SIEMPRE del campo value" in text
    assert "PROHIBIDO usar yoy_pct o pct" in text


def test_original_series_force_instruction_for_per_capita_by_indicator_ctx():
    text = response_module._build_original_series_force_instruction(
        question="cual fue el pib",
        entities_ctx={"indicator_ent": "pib_per_capita"},
    )

    assert text is not None
    assert "SERIE ORIGINAL OBLIGATORIA" in text
    assert "OBLIGATORIO: entrega una sola cifra principal" in text


def test_specific_contribution_directness_instruction_uses_absolute_percent_rules():
    text = response_module._build_specific_contribution_directness_instruction(
        "cuanto contribuyo mineria al pib"
    )

    assert text is not None
    assert "cayó **X,X%** respecto al mismo período del año anterior" in text
    assert "PROHIBIDO usar 'pp'" in text
    assert "PROHIBIDO mostrar signo negativo" in text
    assert "CHEQUEO FINAL OBLIGATORIO" in text


def test_sanitize_contribution_tool_result_rank_series_uses_absolute_value_and_direction():
    raw = {
        "frequency": "T",
        "period": "2025-Q3",
        "metric": "value",
        "order": "desc",
        "ranking": [
            {"series_id": "A", "short_title": "Servicios", "value": 0.6},
            {"series_id": "B", "short_title": "Industria", "value": -1.8},
        ],
    }

    out = response_module._sanitize_contribution_tool_result(
        "rank_series",
        {"frequency": "T", "period": "2025-Q3", "metric": "value", "calc_mode": "contribution"},
        response_module.json.dumps(raw, ensure_ascii=False),
    )
    parsed = response_module.json.loads(out)
    assert parsed["ranking"][0]["value"] == 0.6
    assert parsed["ranking"][0]["contribution_direction"] == "alza"
    assert parsed["ranking"][1]["value"] == 1.8
    assert parsed["ranking"][1]["value_signed"] == -1.8
    assert parsed["ranking"][1]["contribution_direction"] == "baja"
    assert parsed["ranking"][1]["contribution_sign"] == "negative"


def test_sanitize_contribution_tool_result_get_series_data_total_negative_to_absolute():
    raw = {
        "series_id": "F032.PIB.REGION.BIOBIO",
        "frequency": "T",
        "records": [{"period": "2025-Q3", "value": -4.2}],
    }

    out = response_module._sanitize_contribution_tool_result(
        "get_series_data",
        {"series_id": "F032.PIB.REGION.BIOBIO", "frequency": "T", "calc_mode": "contribution"},
        response_module.json.dumps(raw, ensure_ascii=False),
    )
    parsed = response_module.json.loads(out)
    rec = parsed["records"][0]
    assert rec["value"] == 4.2
    assert rec["value_signed"] == -4.2
    assert rec["contribution_direction"] == "baja"


def test_sanitize_contribution_tool_result_no_change_for_non_contribution_modes():
    raw = {
        "series_id": "F032.IMACEC.X",
        "frequency": "M",
        "records": [{"period": "2025-01", "value": -0.1}],
    }

    out = response_module._sanitize_contribution_tool_result(
        "get_series_data",
        {"series_id": "F032.IMACEC.X", "frequency": "M", "calc_mode": "yoy"},
        response_module.json.dumps(raw, ensure_ascii=False),
    )
    parsed = response_module.json.loads(out)
    assert parsed["records"][0]["value"] == -0.1
    assert "contribution_direction" not in parsed["records"][0]
