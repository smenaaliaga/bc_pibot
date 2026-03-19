from orchestrator.graph.send_response import (
    _normalize_percentage_decimals,
    _normalize_contribution_decimals,
    _postprocess_response_sections,
    _requires_original_value_response,
)


def _build_generation_logic(*, question: str, entities: dict, observations: dict) -> dict:
    return {
        "mode": "data",
        "input": {
            "input": {
                "question": question,
                "entities": entities,
                "observations": observations,
            }
        },
    }


def test_requires_original_value_response_is_false_for_imacec_value_query() -> None:
    question = "cual es el valor del ultimo imacec"
    entities = {"calc_mode_cls": "value", "indicator_ent": "imacec"}
    observations = {"classification": {"indicator": "imacec", "price": "co"}}
    generation_logic = _build_generation_logic(
        question=question,
        entities=entities,
        observations=observations,
    )

    assert _requires_original_value_response(question, entities, generation_logic) is False


def test_requires_original_value_response_is_true_for_pib_current_price_query() -> None:
    question = "cual es el pib a precios corrientes del ultimo trimestre"
    entities = {"calc_mode_cls": "value", "indicator_ent": "pib"}
    observations = {"classification": {"indicator": "pib", "price": "co"}}
    generation_logic = _build_generation_logic(
        question=question,
        entities=entities,
        observations=observations,
    )

    assert _requires_original_value_response(question, entities, generation_logic) is True


def test_postprocess_adds_indicator_and_period_context_for_latest_imacec() -> None:
    question = "cual es el valor del ultimo imacec"
    entities = {"calc_mode_cls": "yoy", "indicator_ent": "imacec"}
    observations = {
        "frequency": "M",
        "latest_available": {"M": "2026-01"},
        "classification": {"indicator": "imacec", "price": "co"},
        "series": [
            {
                "series_id": "SERIE.IMACEC",
                "short_title": "IMACEC",
                "data": {
                    "M": {
                        "records": [
                            {"period": "2025-12", "yoy_pct": "-0.8"},
                            {"period": "2026-01", "yoy_pct": "-0.6"},
                        ]
                    }
                },
            }
        ],
    }
    generation_logic = _build_generation_logic(
        question=question,
        entities=entities,
        observations=observations,
    )

    sections = {
        "introduccion": "Con base en los datos disponibles.",
        "respuesta": "La variación anual fue -0,6% respecto al mismo período del año anterior.",
        "sugerencias": "",
        "csv": "",
    }

    _composed, updated = _postprocess_response_sections(sections, generation_logic)
    intro = updated["introduccion"]

    assert "IMACEC" in intro
    assert "enero de 2026" in intro


def test_normalize_contribution_decimals_handles_bare_contribution_values() -> None:
    text = "Comercio contribuyó con **0,53** al PIB y servicios aportó con **0,39**."

    normalized = _normalize_contribution_decimals(text)

    assert "0,5" in normalized
    assert "0,4" in normalized


def test_contribution_decimal_pipeline_preserves_bold_percent_format() -> None:
    text = "Comercio contribuyó con **0,53%** y minería con **-0,61%**."

    normalized = _normalize_contribution_decimals(_normalize_percentage_decimals(text))

    assert "**0,5%**" in normalized
    assert "**-0,6%**" in normalized
    assert "****%" not in normalized
