from main import _missing_followup_blocks


def test_missing_followup_blocks_returns_state_followups_when_stream_missing_them():
    streamed_text = (
        "Respuesta base\n"
        "##CSV_DOWNLOAD_START\n"
        "path=/tmp/series.csv\n"
        "##CSV_DOWNLOAD_END\n"
    )
    state_output = (
        streamed_text
        + "\n##FOLLOWUP_START\n"
        + "suggestion_1=Pregunta uno\n"
        + "suggestion_2=Pregunta dos\n"
        + "##FOLLOWUP_END"
    )

    missing = _missing_followup_blocks(streamed_text, state_output)

    assert len(missing) == 1
    assert "##FOLLOWUP_START" in missing[0]
    assert "suggestion_1=Pregunta uno" in missing[0]
    assert "##FOLLOWUP_END" in missing[0]


def test_missing_followup_blocks_ignores_followups_already_streamed():
    streamed_text = (
        "Respuesta base\n"
        "##FOLLOWUP_START\n"
        "suggestion_1=Pregunta uno\n"
        "##FOLLOWUP_END\n"
    )

    missing = _missing_followup_blocks(streamed_text, streamed_text)

    assert missing == []
