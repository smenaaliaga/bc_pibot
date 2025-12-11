from orchestrator.graph.agent_graph import route_decider


class DummyMeth:
    query_type = "METHODOLOGICAL"
    data_domain = "PIB"


class DummyData:
    query_type = "DATA"
    data_domain = "PIB"


def test_route_decider_goes_rag_for_methodological():
    state = {"classification": DummyMeth()}
    decision = route_decider(state)
    assert decision == "rag"


def test_route_decider_goes_data_for_data():
    state = {"classification": DummyData()}
    decision = route_decider(state)
    assert decision == "data"


def test_route_decider_respects_existing_route_decision():
    state = {"classification": DummyData(), "route_decision": "fallback"}
    decision = route_decider(state)
    assert decision == "fallback"
