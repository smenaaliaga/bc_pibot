from orchestrator.normalizer.standalone_rules import resolve_standalone_route


def test_macro_zero_always_fallback():
    assert (
        resolve_standalone_route(
            normalized_intent="value",
            context_label="standalone",
            macro_label=0,
        )
        == "fallback"
    )


def test_standalone_other_routes_fallback():
    assert (
        resolve_standalone_route(
            normalized_intent="other",
            context_label="standalone",
            macro_label=1,
        )
        == "fallback"
    )


def test_value_routes_data():
    assert (
        resolve_standalone_route(
            normalized_intent="value",
            context_label="standalone",
            macro_label=1,
        )
        == "data"
    )


def test_method_routes_rag():
    assert (
        resolve_standalone_route(
            normalized_intent="method",
            context_label="standalone",
            macro_label=1,
        )
        == "rag"
    )


def test_unknown_routes_fallback():
    assert (
        resolve_standalone_route(
            normalized_intent="unknown",
            context_label="standalone",
            macro_label=1,
        )
        == "fallback"
    )
