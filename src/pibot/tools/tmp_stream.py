from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import get_runtime


class S(TypedDict, total=False):
    output: str


def node(state: S):
    get_runtime().stream_writer({'chunk': 'hello'})
    return {'output': 'done'}

graph = StateGraph(S)
graph.add_node('node', node)
graph.add_edge(START, 'node')
graph.add_edge('node', END)
compiled = graph.compile()
for event in compiled.stream({}, stream_mode=['updates', 'custom']):
    print(event)
