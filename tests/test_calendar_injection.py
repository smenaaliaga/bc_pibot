"""Tests for calendar publication date injection and URL fix."""

from orchestrator.llm.llm_adapter import (
    _is_calendar_question,
    CALENDAR_2026_TEXT,
    CALENDAR_URL,
    CALENDAR_LINK_MD,
    LLMAdapter,
)
from orchestrator.graph.nodes.llm import _fix_calendar_url, _dedup_calendar_refs, make_rag_node


# ── _is_calendar_question detection ──────────────────────────────


def test_detects_cuando_se_publica_pib():
    assert _is_calendar_question("cuando se publica el pib") is True


def test_detects_cuando_se_publica_imacec():
    assert _is_calendar_question("cuando se publica el imacec") is True


def test_detects_proxima_publicacion():
    assert _is_calendar_question("cual es la próxima publicación del pib") is True


def test_detects_calendario_de_publicaciones():
    assert _is_calendar_question("dame el calendario de publicaciones del imacec") is True


def test_detects_fecha_publicacion():
    assert _is_calendar_question("cual es la fecha de publicación del pib trimestral") is True


def test_detects_cuando_sale():
    assert _is_calendar_question("cuando sale el imacec") is True


def test_detects_cuando_publican():
    assert _is_calendar_question("cuando publican el pib regional") is True


def test_does_not_detect_regular_pib_question():
    assert _is_calendar_question("cuanto crecio el pib en 2025") is False


def test_does_not_detect_methodology_question():
    assert _is_calendar_question("que es el imacec") is False


# ── Calendar text content ────────────────────────────────────────


def test_calendar_text_contains_imacec_dates():
    assert "04-05-2026: IMACEC - marzo 2026" in CALENDAR_2026_TEXT
    assert "01-12-2026: IMACEC - octubre 2026" in CALENDAR_2026_TEXT


def test_calendar_text_contains_pib_dates():
    assert "18-05-2026: Cuentas Nacionales Trimestrales - 1er trim 2026" in CALENDAR_2026_TEXT
    assert "18-11-2026: Cuentas Nacionales Trimestrales - 3er trim 2026" in CALENDAR_2026_TEXT


def test_calendar_text_contains_pib_regional_dates():
    assert "23-04-2026: PIB Regional trimestral" in CALENDAR_2026_TEXT
    assert "23-12-2026: PIB Regional trimestral - 3er trim 2026" in CALENDAR_2026_TEXT


def test_calendar_text_contains_correct_url():
    assert CALENDAR_URL in CALENDAR_2026_TEXT


def test_calendar_text_forbids_old_url():
    assert "calendario-de-publicaciones" in CALENDAR_2026_TEXT  # in the NUNCA instruction


# ── _fix_calendar_url ────────────────────────────────────────────


def test_fix_calendar_url_replaces_old_url_in_markdown_link():
    old = "https://www.bcentral.cl/web/banco-central/areas/estadisticas/calendario-de-publicaciones"
    text = f"Visita [Calendario]({old}) para más info."
    result = _fix_calendar_url(text)
    assert old not in result
    assert CALENDAR_URL in result


def test_fix_calendar_url_no_change_when_url_absent():
    text = "El PIB creció 3,5% en el tercer trimestre."
    assert _fix_calendar_url(text) == text


def test_fix_calendar_url_replaces_raw_url_with_markdown_link():
    """A raw CALENDAR_URL (not in markdown) should become a markdown link."""
    text = f"Puedes revisar el calendario oficial aquí:\n{CALENDAR_URL}\n"
    result = _fix_calendar_url(text)
    assert CALENDAR_LINK_MD in result
    # No raw URL leftover outside markdown
    assert result.count(CALENDAR_URL) == 1  # only inside the [text](url)


def test_fix_calendar_url_preserves_existing_markdown_link():
    """A properly formatted markdown link should NOT be double-wrapped."""
    text = f"Revisa el [Calendario de Publicaciones 2026]({CALENDAR_URL}) para detalles."
    result = _fix_calendar_url(text)
    assert result == text  # unchanged


def test_fix_calendar_url_replaces_old_url_raw():
    old = "https://www.bcentral.cl/web/banco-central/areas/estadisticas/calendario-de-publicaciones"
    text = f"Link 1: {old} y Link 2: {old}"
    result = _fix_calendar_url(text)
    assert old not in result
    assert CALENDAR_URL in result


# ── Calendar injection into system prompt ────────────────────────


class _DummyRetriever:
    def invoke(self, payload):
        return []


def test_calendar_injected_into_system_prompt_for_calendar_question():
    adapter = LLMAdapter(streaming=False, retriever=_DummyRetriever(), mode="rag")
    msgs = adapter._build_messages(
        "cuando se publica el proximo pib",
        history=[],
        intent_info={},
    )
    system_text = msgs[0].content if hasattr(msgs[0], "content") else str(msgs[0])
    assert "CALENDARIO DE PUBLICACIONES ESTADÍSTICAS 2026" in system_text
    assert "18-05-2026" in system_text
    assert CALENDAR_URL in system_text


def test_calendar_not_injected_for_regular_question():
    adapter = LLMAdapter(streaming=False, retriever=_DummyRetriever(), mode="rag")
    msgs = adapter._build_messages(
        "cuanto crecio el pib en 2025",
        history=[],
        intent_info={},
    )
    system_text = msgs[0].content if hasattr(msgs[0], "content") else str(msgs[0])
    assert "CALENDARIO DE PUBLICACIONES" not in system_text


# ── rag_node applies URL fix ─────────────────────────────────────


def test_rag_node_fixes_calendar_url_in_output():
    old = "https://www.bcentral.cl/web/banco-central/areas/estadisticas/calendario-de-publicaciones"

    class _FakeAdapter:
        def stream(self, question, history=None, intent_info=None):
            yield f"Consulta el [Calendario]({old})"

        def get_last_rag_sources(self):
            return []

    node = make_rag_node(_FakeAdapter())
    result = node(
        {"question": "cuando se publica el pib", "conversation_history": [], "intent": {"intent": "methodology"}},
    )
    assert old not in result["output"]
    assert CALENDAR_URL in result["output"]


def test_rag_node_hides_raw_calendar_url():
    """If the LLM outputs the raw long URL, rag_node must not leak it.

    Under the new contract the calendar link is surfaced exclusively via the
    methodology footer (never in the body). For calendar-related questions the
    footer is auto-injected with the calendar reference even if the adapter
    returned no sources.
    """

    class _FakeAdapter:
        def stream(self, question, history=None, intent_info=None):
            yield f"Puedes verlo aquí:\n{CALENDAR_URL}\nMás datos próximamente."

        def get_last_rag_sources(self):
            return []

    node = make_rag_node(_FakeAdapter())
    result = node(
        {"question": "cuando se publica el imacec", "conversation_history": [], "intent": {"intent": "methodology"}},
    )
    output = result["output"]
    # Raw URL must not appear unwrapped in the body
    assert f"\n{CALENDAR_URL}\n" not in output
    # The calendar reference must be present exactly once, inside the footer
    assert output.count(CALENDAR_URL) == 1
    assert CALENDAR_LINK_MD in output
    # And it must live inside the references footer
    footer_marker = "Para mayor información"
    assert footer_marker in output
    assert output.index(CALENDAR_LINK_MD) > output.index(footer_marker)


# ── _dedup_calendar_refs ─────────────────────────────────────────


def test_dedup_removes_calendar_from_footer_when_in_body():
    body = (
        f"El próximo IMACEC es el 04-05-2026.\n"
        f"Puedes revisar el calendario aquí: {CALENDAR_LINK_MD}\n\n"
    )
    footer = (
        "Para mayor información, puedes consultar los documentos disponibles en la web oficial del Banco Central de Chile:\n"
        f"- [Calendario de Publicaciones 2026]({CALENDAR_URL})\n"
        f"- [Índice Mensual de Actividad Económica - IMACEC](https://si3.bcentral.cl/imacec.pdf)\n"
    )
    text = body + footer
    result = _dedup_calendar_refs(text)
    # Calendar removed from footer, IMACEC kept
    assert "Índice Mensual de Actividad Económica" in result
    # Calendar link appears only ONCE (in body)
    assert result.count("Calendario") == 1 + 1  # body link + "Para mayor información" or just body
    # Verify body link is preserved
    assert CALENDAR_LINK_MD in result


def test_dedup_no_change_when_calendar_only_in_footer():
    text = (
        "El IMACEC creció 3,5%.\n\n"
        "Para mayor información:\n"
        f"- [Calendario de Publicaciones 2026]({CALENDAR_URL})\n"
    )
    result = _dedup_calendar_refs(text)
    assert result == text  # unchanged — no calendar in body


def test_dedup_no_change_when_no_footer():
    text = f"Próximo IMACEC: 04-05-2026. Ver {CALENDAR_LINK_MD}."
    result = _dedup_calendar_refs(text)
    assert result == text


def test_rag_node_deduplicates_calendar_in_footer():
    """rag_node should remove duplicate calendar from LLM-generated footer."""

    class _FakeAdapter:
        def stream(self, question, history=None, intent_info=None):
            yield (
                f"El próximo IMACEC es el 04-05-2026.\n"
                f"Puedes revisar el calendario aquí: [Calendario de Publicaciones 2026]({CALENDAR_URL})\n\n"
                f"Para mayor información, puedes consultar los documentos disponibles en la web oficial del Banco Central de Chile:\n"
                f"- [Calendario de Publicaciones 2026]({CALENDAR_URL})\n"
                f"- [Índice Mensual de Actividad Económica - IMACEC](https://si3.bcentral.cl/imacec.pdf)\n"
            )

        def get_last_rag_sources(self):
            return []

    node = make_rag_node(_FakeAdapter())
    result = node(
        {"question": "cuando se publica el imacec", "conversation_history": [], "intent": {"intent": "methodology"}},
    )
    output = result["output"]
    # Calendar link appears only in body, not duplicated in footer
    assert "Índice Mensual de Actividad Económica" in output
    # Count occurrences of markdown calendar links
    import re
    calendar_links = re.findall(r"\[.*?[Cc]alendario.*?\]\(", output)
    assert len(calendar_links) == 1, f"Expected 1 calendar link, found {len(calendar_links)}: {calendar_links}"
