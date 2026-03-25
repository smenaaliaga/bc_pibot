"""Debug graph execution: invoke or stream with different modes.

Usage:
    python tools/debug_graph.py                       # invoke (default)
    python tools/debug_graph.py --stream              # stream mode=updates
    python tools/debug_graph.py --stream -m values    # stream mode=values
    python tools/debug_graph.py --stream -m updates -m custom  # multiple modes
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug graph execution")
    parser.add_argument("--stream", action="store_true", help="Use graph.stream instead of graph.invoke")
    parser.add_argument("-m", "--mode", action="append", default=None,
                        help="Stream mode(s): updates, values, custom (repeatable). Default: updates")
    args = parser.parse_args()

    graph = build_graph()

    if args.stream:
        modes = args.mode or ["updates"]
        stream_mode = modes[0] if len(modes) == 1 else modes
        session_id = f"test-session-{stream_mode}"
        state = {
            "question": "que es el pib?",
            "history": [{"role": "assistant", "content": "Hola"}],
            "context": {"session_id": session_id},
        }
        cfg = {"configurable": {"thread_id": session_id, "checkpoint_ns": "memory"}}
        print(f"--- stream_mode={stream_mode} ---")
        for idx, event in enumerate(graph.stream(state, config=cfg, stream_mode=stream_mode), start=1):
            print(idx, event)
    else:
        state = {
            "question": "que es el pib?",
            "history": [{"role": "assistant", "content": "Hola"}],
            "context": {"session_id": "invoke-test"},
        }
        cfg = {"configurable": {"thread_id": "invoke-test", "checkpoint_ns": "memory"}}
        result = graph.invoke(state, config=cfg)
        print(result)


if __name__ == "__main__":
    main()
