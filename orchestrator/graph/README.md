# Grafo LangGraph

Controla la orquestación completa de cada turno. Todos los nodos comparten un `AgentState`
centralizado y emiten chunks en vivo mediante LangGraph Topics + `StreamWriter`.

-## Componentes principales (`agent_graph.py`)
- `AgentState`: `TypedDict` con `question`, `history`, `context`,
  `classification`/`intent_info`, `route_decision`, `output` y
  `stream_chunks: Topic[str]` para
  depurar desde `tools/debug_graph_stream.py`.
- `_ensure_backends()`: inicializa una sola vez `MemoryAdapter`, `LLMAdapter` y el retriever RAG.
- `_emit_stream_chunk()`: escribe el fragmento both en el writer recibido y en
  `get_runtime().stream_writer`, habilitando los eventos `custom` consumidos por Streamlit.
- `build_agent_graph()`: registra los nodos `ingest`, `classify`, `intent`, `router`,
  `data`, `rag`, `fallback` y `memory`, definiendo transiciones y terminales (`END`).

## Ciclo de ejecución
| Nodo | Entrada clave | Salida/efecto |
| --- | --- | --- |
| `ingest` | `question`, `history` | Normaliza texto, crea `session_id`, obtiene la ventana reciente desde memoria y arma `context`. |
| `classify` | Estado previo | Llama a `intents/classifier_agent.classify_question_with_history` que usa `prompts/query_classifier.py` y produce `ClassificationResult` + `history_text`. |
| `intent` | `classification`, historial de entidades | Invoca el IntentRouter clásico, rellena entidades faltantes y propone `route_decision`. |
| `router` | `route_decision` | Fan-out hacia `data`/`rag`/`fallback` y valida follow-ups de gráficos. |
| `data` / `rag` / `fallback` | Dependiendo de la ruta | Emiten los chunks de cada flujo; `data` y `rag` también adjuntan markers y follow-ups. |
| `memory` | `output` | Persiste la respuesta y el checkpoint del agente a través de `MemoryAdapter`. |

## Streaming
- Los nodos productores llaman `_emit_stream_chunk(chunk, writer)` antes de `yield`; Streamlit recibe
  `{"stream_chunks": "texto"}` en el canal `custom` y lo pinta tal cual llega (sin deduplicación automática).
- `graph.stream(..., stream_mode=["updates","custom"])` envía tanto cambios de estado como eventos
  personalizados. Úsalo también en herramientas CLI (`tools/debug_graph_stream.py`).
- Para nuevos canales, agrega un `Topic` adicional en `AgentState` o incluye claves específicas en el
  diccionario emitido por `_emit_stream_chunk`.

## Extender el grafo
1. Agrega nodos en `build_agent_graph()` usando `StateGraph.add_node` y `add_edge`.
2. Mantén los efectos secundarios contenidos: los nodos deberían leer/escribir en `state` y emitir
   chunks, evitando IO descontrolado.
3. Si necesitas backends adicionales (p. ej. otro retriever), extiende `_ensure_backends()` y actualiza
   los nodos que los usan.
4. Revisa `tests/test_agent_graph_streaming.py` y `tools/test_orch2_chunk.py` después de cambios
   estructurales.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de datos](../data/README.md)
- [README del clasificador](../classifier/README.md)
- [README de pruebas](../../tests/README.md)
