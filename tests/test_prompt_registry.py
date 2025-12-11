from orchestrator.prompts import registry


def test_classifier_prompt_mentions_tool_name():
    prompt = registry.build_classifier_prompt()
    assert "classify_economic_query" in prompt
    assert "query_type" in prompt


def test_data_method_prompt_contains_placeholders():
    system_msg, human_msg = registry.build_data_method_prompt()
    assert "modo de respuesta orientada a DATOS" in system_msg
    assert "{history}" in human_msg
    assert "{mode_instruction}" in human_msg


def test_data_summary_prompt_is_parameterized():
    system_msg, human_msg = registry.build_data_summary_prompt()
    assert "FASE 2" in system_msg
    for placeholder in (
        "{domain}",
        "{year}",
        "{table_description}",
        "{table_excerpt}",
        "{latest_yoy_summary}",
    ):
        assert placeholder in human_msg
    assert "hasta tres" in system_msg.lower()
    assert "último valor" in human_msg.lower()
    assert "variación anual" in human_msg.lower()


def test_guardrail_prompt_changes_with_mode():
    rag_prompt = registry.build_guardrail_prompt(mode="rag")
    fallback_prompt = registry.build_guardrail_prompt(mode="fallback")
    assert "base de conocimiento" in rag_prompt
    assert "no dispongas" in fallback_prompt
    assert rag_prompt != fallback_prompt
