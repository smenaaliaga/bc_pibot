from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.channels.topic import Topic

class State(TypedDict, total=False):
    output: str
    stream_chunks: Annotated[List[str], Topic(str)]

def node(state: State):
    yield {"stream_chunks": "hola"}
    yield {"stream_chunks": " mundo"}
    return {"output": "hola mundo"}

graph = StateGraph(State)
graph.add_node("node", node)
graph.add_edge(START, "node")
graph.add_edge("node", END)
compiled = graph.compile()
for event in compiled.stream({}, stream_mode=["updates"]):
    print(event)
