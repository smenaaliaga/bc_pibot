"""Compare graph.stream outputs for different stream modes (with debug logging)."""

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "DEBUG")
logging.basicConfig(level=logging.DEBUG)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.graph.agent_graph import build_graph  # noqa: E402


def _dump(mode: str) -> None:
    graph = build_graph()
    state = {
        "question": "que es el pib?",
        "history": [{"role": "assistant", "content": "Hola"}],
        "context": {"session_id": f"test-session-{mode}"},
    }
    cfg = {
        "configurable": {"thread_id": f"test-session-{mode}", "checkpoint_ns": "memory"},
    }
    print(f"--- stream_mode={mode} ---")
    for idx, event in enumerate(graph.stream(state, config=cfg, stream_mode=mode), start=1):
        print(idx, event)


def main() -> None:
    _dump("updates")
    _dump("values")
    _dump(["updates", "custom"])


if __name__ == "__main__":
    main()
