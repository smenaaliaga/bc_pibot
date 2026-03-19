from orchestrator.graph.send_response import _enforce_original_value_narrative, _split_sentences


def test_split_sentences_keeps_numeric_thousands_separator() -> None:
    text = (
        "En el 4to trimestre de 2025, el PIB fue de **57.246,76** miles de millones de pesos encadenados. "
        "La variacion anual fue **1,55%** respecto al mismo periodo del ano anterior."
    )

    sentences = _split_sentences(text)

    assert len(sentences) == 2
    assert "57.246,76" in sentences[0]
    assert "1,55%" in sentences[1]


def test_enforce_original_value_narrative_preserves_whole_number_sentence() -> None:
    text = (
        "En el 4to trimestre de 2025, el PIB fue de **57.246,76** miles de millones de pesos encadenados. "
        "La variacion anual fue **1,55%** respecto al mismo periodo del ano anterior."
    )

    cleaned = _enforce_original_value_narrative(text)

    assert "57.246,76" in cleaned
    assert "57.\n\n246,76" not in cleaned
    assert "variacion anual" not in cleaned.lower()
