# RAG (Retrieval-Augmented Generation)

Este módulo arma el retriever compartido que el grafo inyecta al `LLMAdapter`. Incluye lógica de
feature flag, prioridad de backends y configuración declarada vía variables de entorno.

## Componentes
- `rag_factory.py`
	- `_embeddings()`: inicializa `OpenAIEmbeddings` y valida modelo (usa
		`OPENAI_EMBEDDINGS_MODEL` o `OPENAI_MODEL`).
	- `_pgvector_retriever`, `_faiss_retriever`, `_chroma_retriever`: adaptadores para cada backend.
	- `create_retriever()`: función pública que respeta `RAG_ENABLED`, `RAG_BACKEND`, `RAG_TOP_K` y
		otorga un objeto compatible con LangChain `.as_retriever()`.

## Cómo funciona
1. Al levantar el orquestador, `agent_graph._ensure_backends()` llama `create_retriever()`.
2. Se crea un embedders y luego se recorre un orden de preferencia configurable.
3. Si ningún backend está disponible, se registra una advertencia y el grafo seguirá respondiendo sin
	 contexto documental (el LLM avisará que no hay datos RAG).

## Configuración útil
- `RAG_ENABLED` (default `true`).
- `RAG_BACKEND` (`pgvector`, `faiss`, `chroma` o vacío para auto fallback).
- `RAG_PGVECTOR_URL`, `RAG_PGVECTOR_COLLECTION`, `RAG_PGVECTOR_USE_JSONB`.
- `RAG_FAISS_PATH` / `RAG_CHROMA_PATH` para despliegues locales sin Postgres.
- `RAG_TOP_K` para ajustar el número de fragmentos que se envían al prompt.

## Validación
- `tests/test_orchestrator2.py::test_rag_branch` cubre la inyección del retriever en el grafo.
- `tools/test_orch2_chunk.py` muestra cómo se integran los chunks RAG en la respuesta final.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
- [README de Docker](../../docker/README.md)
- [README de scripts](../../readme/README.md)
