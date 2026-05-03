from typing import TypedDict, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter


class State(TypedDict, total=False):
    output: str


def foo_node(state: State, *, writer: Optional[StreamWriter] = None):
    if writer:
        writer({"chunk": "A"})
        writer({"chunk": "B"})
        writer({"chunk": "C"})
    return {"output": "ABC"}


def main() -> None:
    builder = StateGraph(State)
    builder.add_node("foo", foo_node)
    builder.add_edge(START, "foo")
    builder.add_edge("foo", END)
    graph = builder.compile()
    print("--- custom ---")
    for idx, event in enumerate(graph.stream({}, stream_mode="custom"), start=1):
        print(idx, event)


if __name__ == "__main__":
    main()
