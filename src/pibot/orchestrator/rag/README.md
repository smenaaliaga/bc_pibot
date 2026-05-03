# RAG (Retrieval-Augmented Generation)

Construye el retriever compartido que el grafo inyecta en `LLMAdapter`. Maneja feature flags, prioridad
de backends y configuración vía variables de entorno.

## Componentes
- `rag_factory.py`
	- `_embeddings()`: inicializa `OpenAIEmbeddings` validando modelo (`OPENAI_EMBEDDINGS_MODEL` o
		`OPENAI_MODEL`).
	- `_pgvector_retriever`, `_faiss_retriever`, `_chroma_retriever`: adaptadores por backend que exponen
		`.as_retriever()`.
	- `create_retriever()`: función pública que respeta `RAG_ENABLED`, `RAG_BACKEND`, `RAG_TOP_K` y retorna
		`None` si el feature flag está apagado.

## Flujo en runtime
1. `agent_graph._ensure_backends()` invoca `create_retriever()` apenas se inicia la sesión.
2. Se crea un embedder, se intenta cada backend disponible en orden de preferencia y se guarda el
   retriever resultante.
3. Si todo falla, se loggea la advertencia `[RAG_DISABLED]` y el grafo responde usando solo datos/LLM.

## Configuración útil
- `RAG_ENABLED` (default `true`).
- `RAG_BACKEND` = `pgvector` | `faiss` | `chroma` (vacío → auto fallback).
- `RAG_PGVECTOR_URL`, `RAG_PGVECTOR_COLLECTION`, `RAG_PGVECTOR_USE_JSONB`.
- `RAG_FAISS_PATH` / `RAG_CHROMA_PATH` para despliegues locales.
- `RAG_TOP_K`, `RAG_SCORE_THRESHOLD`, `RAG_VECTOR_OPTS` (JSON opcional con filtros).
- `RAG_CONTEXT_MAX_CHARS` limita el contexto que se inyecta en el prompt del LLM.

### Relación con los loaders
Los documentos se cargan mediante `docker/postgres/load_txt_rag.py` (manifest-driven, staging tables).
Cuando hagas cambios en esquemas o colecciones, actualiza también las variables anteriores para que el
retriever apunte a los índices correctos.

## Validación
- `pytest tests/test_rag_factory.py` y `tests/test_orchestrator2.py::test_rag_branch`.
- `tools/test_orch2_chunk.py --mode rag` te permite ver los markers RAG en streaming.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de Docker](../../docker/README.md)
- [README de pruebas](../../tests/README.md)
