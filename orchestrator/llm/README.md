# Capa LLM

Abstracción única para todo lo que interactúa con modelos de lenguaje (fallback, respuestas
metodológicas y modo RAG). El objetivo es mantener la interacción con LangChain/OpenAI confinada
en este módulo para facilitar los cambios de backend.

## Archivos
- `llm_adapter.py`: clase `LLMAdapter` con soporte de streaming, inyección de contexto RAG,
	historial y facts persistidos. Encapsula la construcción de mensajes (`_build_messages`), la
	selección del backend (`ChatOpenAI` o `init_chat_model`) y la lógica de `stream()/generate()`.
- `system_prompt.py`: builder del mensaje de sistema. Incluye guardrails, formato de cita,
	instrucciones de transparencia y toggles para prompts extendidos.

## Responsabilidades
- Normalizar contexto: combina `intent_info`, facts de memoria y contexto RAG antes de llegar al
	modelo.
- RAG opt-in: cuando se provee un retriever, el adapter decide el filtro (`topic`, `sector`,
	`seasonality`) y limita el tamaño del contexto (`RAG_CONTEXT_MAX_CHARS`).
- Streaming real: `LLMAdapter.stream` produce chunks de texto respetando el modo SSE y delega el
	logging (`[LLM_STREAM_CHUNK]`).
- Compatibilidad: cuando LangChain no está disponible, emite respuestas stub para pruebas locales.

## Extender
- Nuevos backends se incorporan modificando `_build_messages` y el bloque que inicializa `self._chat`.
	Evita importar SDKs directamente fuera de esta clase.
- Si necesitas adjuntar más contexto (p. ej. resúmenes), agrega la lógica en `_build_messages` para
	mantener toda la ingeniería de prompts en un solo lugar.
- Para prompts específicos crea helpers en `system_prompt.py` y consúmelos desde el adapter.

## Recomendaciones
- Usa `build_llm(streaming=True, retriever=...)` desde el grafo para mantener las opciones
	alineadas con LangGraph.
- Revisa `tests/test_orchestrator2.py::test_rag_branch` cuando toques la construcción de mensajes;
	garantiza que los filtros esperados lleguen al retriever.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
- [README de Docker](../../docker/README.md)
- [README de scripts](../../readme/README.md)
