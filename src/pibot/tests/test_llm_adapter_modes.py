from orchestrator.llm.llm_adapter import build_llm


def test_build_llm_sets_custom_mode():
    adapter = build_llm(streaming=False, mode="fallback")
    assert getattr(adapter, "mode") == "fallback"


def test_build_llm_defaults_to_rag_mode():
    adapter = build_llm(streaming=False)
    assert getattr(adapter, "mode") == "rag"
