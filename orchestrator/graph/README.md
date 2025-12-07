# Grafo LangGraph

Controla la orquestación completa de un turno de conversación. Cada nodo actualiza el `AgentState`
compartido y puede emitir chunks en streaming (LangGraph Topics + `StreamWriter`).

## Componentes
- `agent_graph.py`
	- `AgentState`: `TypedDict` que centraliza `question`, `history`, `facts`, `classification`, `output`
		y `stream_chunks` (Topic acumulativo usado para debugging/tests).
	- `_ensure_backends`: inicializa memoria, retriever y adaptadores LLM una sola vez.
	- `_emit_stream_chunk`: escribe el chunk tanto en el writer explícito como en el `runtime.stream_writer`
		(esto habilita los eventos `custom` consumidos por Streamlit).
	- Nodos: `ingest`, `classify`, `intent_shortcuts`, `data`, `rag`, `fallback`, `direct`, `memory`.

## Flujo resumido
1. `ingest`: normaliza texto, asegura `session_id` y precarga facts.
2. `classify`: llama a `classifier_agent` y construye `intent_info`.
3. `intent_shortcuts`: ejecuta intents deterministas (puede cerrar la conversación aquí mismo).
4. `data` / `rag` / `fallback`: se elige según `route_decision` o `ClassificationResult`.
5. `memory`: persiste la respuesta y actualiza hechos.

## Streaming
- Todos los nodos relevantes usan `_emit_stream_chunk` antes de `yield` para que cada fragmento llegue
	al front.
- `graph.stream(..., stream_mode=["updates","custom"])` envía tanto los cambios de estado como los
	chunks personalizados (`{"stream_chunks": "texto"}`).

## Extender
- Agrega nodos usando `StateGraph` en `build_agent_graph()`. Mantén la cantidad de llaves en el estado
	lo más reducida posible para evitar colisiones con LangGraph.
- Si necesitas emitir nuevos canales, crea un `Topic` en `AgentState` o define claves de streaming
	personalizadas en `_emit_stream_chunk`.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
- [README de Docker](../../docker/README.md)
- [README de scripts](../../readme/README.md)
