from orchestrator.graph import agent_graph
from orchestrator.routes import intent_router


class DummyMemory:
    def __init__(self, turns=None):
        self._turns = turns or []

    def get_turn_metadata(self, session_id, turn=-1):
        if not self._turns:
            return None
        try:
            entry = self._turns[turn]
        except IndexError:
            return None
        return entry.get("metadata")


def test_chart_followup_sets_metadata():
    memory = DummyMemory(
        turns=[{"role": "assistant", "content": "[chart::PIB]", "metadata": {"chart_domain": "PIB"}}]
    )
    response = intent_router._handle_chart_followup(
        "¿puedes mostrar un gráfico del PIB?",
        "PIB",
        memory,
        session_id="sess-1",
    )

    assert response is not None
    chunks = list(response)
    assert chunks and "[CHART:PIB]" in chunks[0]
    assert response.metadata.get("chart_domain") == "PIB"
    assert response.metadata.get("chart_request") is True


def test_chart_followup_skips_mismatched_domain():
    turns = [
        {"role": "assistant", "content": "[chart::PIB]", "metadata": {"chart_domain": "PIB"}},
    ]
    memory = DummyMemory(turns=turns)
    result = intent_router._handle_chart_followup(
        "¿puedes mostrar un gráfico del IMACEC?",
        "IMACEC",
        memory,
        session_id="sess-2",
    )

    assert result is None


def test_chart_followup_requires_recent_chart_turn():
    turns = [
        {"role": "assistant", "content": "[chart::PIB]", "metadata": {"chart_domain": "PIB"}},
        {"role": "assistant", "content": "tabla", "metadata": {}},
    ]
    memory = DummyMemory(turns=turns)

    result = intent_router._handle_chart_followup(
        "muestra un gráfico del PIB",
        "PIB",
        memory,
        session_id="sess-3",
    )

    assert result is None


def test_chart_followup_skips_mentions_without_command():
    turns = [
        {"role": "assistant", "content": "[chart::PIB]", "metadata": {"chart_domain": "PIB"}},
    ]
    memory = DummyMemory(turns=turns)

    result = intent_router._handle_chart_followup(
        "Ese gráfico se ve raro, ¿por qué?",
        "PIB",
        memory,
        session_id="sess-4",
    )

    assert result is None


def test_chart_followup_accepts_explicit_command():
    turns = [
        {"role": "assistant", "content": "[chart::PIB]", "metadata": {"chart_domain": "PIB"}},
    ]
    memory = DummyMemory(turns=turns)

    response = intent_router._handle_chart_followup(
        "Otro gráfico del PIB, por favor",
        "PIB",
        memory,
        session_id="sess-5",
    )

    assert response is not None
    chunks = list(response)
    assert chunks and "[CHART:PIB]" in chunks[0]


def test_chart_followup_uses_context_without_recent_chart():
    turns = [
        {"role": "assistant", "content": "Tabla reciente", "metadata": {}},
    ]
    memory = DummyMemory(turns=turns)

    response = intent_router._handle_chart_followup(
        "puedes generar un gráfico?",
        "PIB",
        memory,
        session_id="sess-ctx",
    )

    assert response is not None
    chunks = list(response)
    assert chunks and "[CHART:PIB]" in chunks[0]


def test_memory_node_records_chart_metadata(monkeypatch):
    class RecorderMemory:
        def __init__(self):
            self.turns = []
            self.saved = []

        def on_assistant_turn(self, session_id, message, *, metadata=None):
            self.turns.append((session_id, message, metadata or {}))

        def save_checkpoint(self, session_id, checkpoint, *, metadata=None):
            self.saved.append((session_id, checkpoint, metadata))
            return True

    dummy_memory = RecorderMemory()
    monkeypatch.setattr(agent_graph, "_MEMORY", dummy_memory)

    chart_output = (
        "Tabla reciente\n"
        "##CHART_START\n"
        "type=line\n"
        "title=PIB - Variación anual (%) 2025\n"
        "data_path=/tmp/pib_chart.csv\n"
        "columns=date,yoy_pct\n"
        "domain=PIB\n"
        "##CHART_END\n"
    )

    class _Cls:
        query_type = "DATA"

    state = {
        "session_id": "sess-chart",
        "output": chart_output,
        "question": "puedes generar un gráfico?",
        "route_decision": "data",
        "classification": _Cls(),
    }

    result = agent_graph.memory_node(state)

    assert result["output"].startswith(chart_output)
    assert dummy_memory.turns
    assert dummy_memory.turns[-1][2].get("chart_domain") == "PIB"