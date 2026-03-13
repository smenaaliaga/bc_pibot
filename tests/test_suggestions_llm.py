from __future__ import annotations

import types

import orchestrator.graph.suggestions as suggestions_module


class _DummyMessage:
    def __init__(self, content: str):
        self.content = content


class _DummyChat:
    captured_messages = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def invoke(self, messages):
        _DummyChat.captured_messages = messages
        return types.SimpleNamespace(
            content=(
                "1. ¿Quieres comparar ese dato con el trimestre anterior?\n"
                "2. ¿Te muestro la serie del PIB en precios encadenados?\n"
                "3. ¿Quieres desagregar el PIB por actividad económica?"
            )
        )


def test_extract_followups_from_llm_text_parses_numbered_lines():
    raw = (
        "1. ¿Quieres comparar con el mes anterior?\n"
        "2) ¿Te muestro un gráfico por trimestre?\n"
        "- ¿Quieres ver el detalle por actividad?"
    )

    parsed = suggestions_module._extract_followups_from_llm_text(raw)

    assert len(parsed) == 3
    assert parsed[0].startswith("¿Quieres comparar")
    assert parsed[1].startswith("¿Te muestro")
    assert parsed[2].startswith("¿Quieres ver")


def test_generate_suggested_questions_uses_llm_and_limits_to_three(monkeypatch):
    monkeypatch.setenv("FOLLOWUP_SUGGESTIONS_USE_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setattr(suggestions_module, "ChatOpenAI", _DummyChat)
    monkeypatch.setattr(suggestions_module, "SystemMessage", _DummyMessage)
    monkeypatch.setattr(suggestions_module, "HumanMessage", _DummyMessage)

    state = {
        "session_id": "s-test",
        "question": "cual es el valor del pib",
        "output": "El PIB tuvo una variación interanual de 1,58% en el último trimestre.",
        "entities": [{"indicator": "pib", "seasonality": "nsa"}],
        "classification": types.SimpleNamespace(intent="value"),
    }

    suggestions = suggestions_module.generate_suggested_questions(state, intent_store=None)

    assert 2 <= len(suggestions) <= 3
    assert len(suggestions) == 3
    assert "Pregunta del usuario" in str(_DummyChat.captured_messages[1].content)
    assert all(text.endswith("?") for text in suggestions)


def test_generate_suggested_questions_fills_minimum_with_fallback(monkeypatch):
    class _SingleLineChat:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, messages):
            return types.SimpleNamespace(content="1. ¿Quieres comparar ese dato con el periodo anterior?")

    monkeypatch.setenv("FOLLOWUP_SUGGESTIONS_USE_LLM", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setattr(suggestions_module, "ChatOpenAI", _SingleLineChat)
    monkeypatch.setattr(suggestions_module, "SystemMessage", _DummyMessage)
    monkeypatch.setattr(suggestions_module, "HumanMessage", _DummyMessage)

    state = {
        "session_id": "s-test-2",
        "question": "cual es el valor del imacec",
        "output": "El IMACEC mostró una variación interanual de -0,09%.",
        "entities": [{"indicator": "imacec", "seasonality": "nsa"}],
        "classification": types.SimpleNamespace(intent="value"),
    }

    suggestions = suggestions_module.generate_suggested_questions(state, intent_store=None)

    assert 2 <= len(suggestions) <= 3
