import pytest

from orchestrator.memory.memory_adapter import MemoryAdapter


@pytest.fixture()
def memory_adapter(monkeypatch):
    """Return a MemoryAdapter that always uses the local fallback paths."""
    monkeypatch.delenv("PG_DSN", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PG_LOCALHOST_DSN", raising=False)
    return MemoryAdapter(pg_dsn="")


def test_window_returns_recent_turns(memory_adapter):
    session_id = "session-a"
    memory_adapter.on_user_turn(session_id, "hola")
    memory_adapter.on_assistant_turn(session_id, "respuesta")

    window = memory_adapter.get_window_for_llm(session_id, max_turns=4)

    assert [turn["role"] for turn in window] == ["user", "assistant"]
    assert window[-1]["content"] == "respuesta"


def test_recent_turn_limit(memory_adapter):
    session_id = "session-b"
    for idx in range(5):
        memory_adapter.on_user_turn(session_id, f"turno {idx}")

    recent = memory_adapter.get_recent_turns(session_id, limit=3)

    assert len(recent) == 3
    assert [t["content"] for t in recent] == ["turno 2", "turno 3", "turno 4"]


def test_checkpoint_roundtrip(memory_adapter):
    session_id = "session-c"
    payload = {"output": "respuesta", "question": "?"}

    saved = memory_adapter.save_checkpoint(session_id, payload, metadata={"route": "rag"})

    assert saved is True

    restored = memory_adapter.load_checkpoint(session_id)

    assert restored is not None
    assert restored["checkpoint"]["output"] == "respuesta"
    assert restored["metadata"].get("route") == "rag"


def test_recent_turns_include_metadata(memory_adapter):
    session_id = "session-metadata"
    memory_adapter.on_assistant_turn(session_id, "chart output", metadata={"chart_domain": "PIB"})

    turns = memory_adapter.get_recent_turns(session_id, limit=1)

    assert turns
    assert turns[0]["metadata"].get("chart_domain") == "PIB"
