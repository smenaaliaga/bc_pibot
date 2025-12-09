import datetime

import pytest

from orchestrator.graph import agent_graph
from orchestrator.routes import intent_router


class DummyMemory:
    def __init__(self, turns=None):
        self.facts = {}
        self._turns = turns or []

    def set_facts(self, session_id, facts):
        self.facts.setdefault(session_id, {}).update(facts)

    def get_recent_turns(self, session_id, limit=6):
        return list(self._turns)


class DummyDF:
    def __init__(self, domain="PIB"):
        self._last_data_context = {"domain": domain, "data_full": {"observations": [1]}}

    def _emit_chart_marker(self, domain, data):
        return f"[chart::{domain}]"


@pytest.fixture()
def patch_df(monkeypatch):
    dummy = DummyDF()
    monkeypatch.setattr(intent_router, "_df", dummy)
    return dummy


def test_chart_followup_sets_metadata_and_facts(patch_df):
    memory = DummyMemory(
        turns=[{"role": "assistant", "content": "[chart::PIB]", "metadata": {"chart_domain": "PIB"}}]
    )
    response = intent_router._handle_chart_followup(
        "¿puedes mostrar un gráfico del PIB?",
        "PIB",
        memory,
        session_id="sess-1",
        facts={},
    )

    assert response is not None
    chunks = list(response)
    assert chunks == ["[chart::PIB]"]
    assert response.metadata.get("chart_domain") == "PIB"
    assert "chart_last_domain" in memory.facts["sess-1"]


def test_chart_followup_skips_mismatched_domain(monkeypatch, patch_df):
    turns = [
        {"role": "assistant", "content": "[chart::PIB]", "metadata": {"chart_domain": "PIB"}},
    ]
    memory = DummyMemory(turns=turns)
    result = intent_router._handle_chart_followup(
        "¿puedes mostrar un gráfico del IMACEC?",
        "IMACEC",
        memory,
        session_id="sess-2",
        facts={"chart_last_domain": "PIB", "chart_last_ts": datetime.datetime.now(datetime.timezone.utc).isoformat()},
    )

    assert result is None


def test_chart_followup_requires_recent_chart_turn(patch_df):
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
        facts={"chart_last_domain": "PIB"},
    )

    assert result is None


def test_chart_followup_skips_mentions_without_command(patch_df):
    turns = [
        {"role": "assistant", "content": "[chart::PIB]", "metadata": {"chart_domain": "PIB"}},
    ]
    memory = DummyMemory(turns=turns)

    result = intent_router._handle_chart_followup(
        "Ese gráfico se ve raro, ¿por qué?",
        "PIB",
        memory,
        session_id="sess-4",
        facts={"chart_last_domain": "PIB"},
    )

    assert result is None


def test_chart_followup_accepts_explicit_command(patch_df):
    turns = [
        {"role": "assistant", "content": "[chart::PIB]", "metadata": {"chart_domain": "PIB"}},
    ]
    memory = DummyMemory(turns=turns)

    response = intent_router._handle_chart_followup(
        "Otro gráfico del PIB, por favor",
        "PIB",
        memory,
        session_id="sess-5",
        facts={"chart_last_domain": "PIB"},
    )

    assert response is not None
    assert list(response) == ["[chart::PIB]"]


def test_chart_followup_uses_context_without_recent_chart(monkeypatch, patch_df):
    turns = [
        {"role": "assistant", "content": "Tabla reciente", "metadata": {}},
    ]
    memory = DummyMemory(turns=turns)

    response = intent_router._handle_chart_followup(
        "puedes generar un gráfico?",
        "PIB",
        memory,
        session_id="sess-ctx",
        facts={},
    )

    assert response is not None
    assert list(response) == ["[chart::PIB]"]


def test_memory_node_records_chart_metadata(monkeypatch):
    class RecorderMemory:
        def __init__(self):
            self.turns = []
            self.facts = {}
            self.saved = []
            self.saver = None

        def on_assistant_turn(self, session_id, message, *, metadata=None):
            self.turns.append((session_id, message, metadata or {}))

        def set_facts(self, session_id, facts):
            self.facts.setdefault(session_id, {}).update(facts)

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
        "facts": {"chart_last_domain": ""},
    }

    result = agent_graph.memory_node(state)

    assert result["output"] == chart_output
    assert dummy_memory.turns
    assert dummy_memory.turns[-1][2].get("chart_domain") == "PIB"
    assert dummy_memory.facts["sess-chart"]["chart_last_domain"] == "PIB"