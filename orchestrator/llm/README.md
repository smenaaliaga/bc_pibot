# Capa LLM

Abstracción única para todo lo que interactúa con modelos de lenguaje (respuestas metodológicas,
RAG, fallback). Mantiene la dependencia con LangChain/OpenAI confinada en un solo lugar para facilitar
cambios de backend.

## Archivos
- `llm_adapter.py`: implementación de `LLMAdapter` con soporte de streaming, inyección de contexto RAG,
  historial y facts. Se encarga de `_build_messages`, del `ChatOpenAI`/`init_chat_model` y de la lógica
  de `stream()`/`generate()`.
- `system_prompt.py`: construye el mensaje de sistema común. Define guardrails (tono, transparencia,
  citas), instrucciones adicionales para PIB/IMACEC y toggles para prompts extendidos.

## Responsabilidades
- Normalizar contexto: combina `intent_info`, facts (`memory`), clasificación, follow-ups pendientes y
  contexto RAG antes de llamar al modelo.
- RAG opt-in: cuando se le entrega un retriever, el adapter decide los filtros (`topic`, `sector` o
  `seasonality`) y limita el tamaño del contexto (`RAG_CONTEXT_MAX_CHARS`).
- Streaming: `LLMAdapter.stream` produce chunks respetando SSE y marca cada pedazo con
  `[LLM_STREAM_CHUNK]` para logging.
- Fallback seguro: si LangChain o las credenciales no están presentes, devuelve una respuesta stub para
  que la UI siga operativa durante pruebas locales.

## Extender
- **Nuevos modelos/backends**: modifica `_build_messages` y el bloque donde se inicializa `self._chat`. No
  importes SDKs fuera de esta clase para mantener el aislamiento.
- **Más contexto**: añade funciones helper en `system_prompt.py` y llámalas desde `_build_messages`.
- **Nuevos modos** (ej. evaluación): crea métodos específicos (`stream_eval`) que reutilicen la lógica
  de construcción de mensajes para garantizar consistencia.

## Validación
- `pytest tests/test_llm_adapter_modes.py`
- `tests/test_agent_graph_streaming.py` para asegurarse de que los chunks lleguen correctamente.
- `tools/debug_llm_stream.py` permite inspeccionar en consola cómo se emiten los eventos `custom`.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
