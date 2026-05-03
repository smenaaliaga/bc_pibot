"""Test para prompts del sistema."""
from orchestrator.llm.system_prompt import build_system_message, GuardrailMode


def test_system_prompt_builds():
    """Verifica que el prompt del sistema se construye correctamente."""
    prompt = build_system_message(mode="rag")
    assert "PIBot" in prompt
    assert "español" in prompt


def test_guardrail_prompt_changes_with_mode():
    """Verifica que el prompt cambia según el modo."""
    rag_prompt = build_system_message(mode="rag")
    fallback_prompt = build_system_message(mode="fallback")
    assert "RAG" in rag_prompt
    assert "FALLBACK" in fallback_prompt
    assert rag_prompt != fallback_prompt


def test_guards_can_be_disabled():
    """Verifica que las guardas se pueden deshabilitar."""
    with_guards = build_system_message(include_guards=True)
    without_guards = build_system_message(include_guards=False)
    assert "inventes" in with_guards
    assert len(without_guards) < len(with_guards)

