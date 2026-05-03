from orchestrator.graph.agent_graph import build_graph


def main() -> None:
    graph = build_graph()
    state = {
        "question": "que es el pib?",
        "history": [{"role": "assistant", "content": "Hola"}],
        "context": {"session_id": "invoke-test"},
    }
    cfg = {
        "configurable": {"thread_id": "invoke-test", "checkpoint_ns": "memory"},
    }
    result = graph.invoke(state, config=cfg)
    print(result)


if __name__ == "__main__":
    main()
