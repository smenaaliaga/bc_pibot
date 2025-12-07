# Memoria conversacional

Maneja la persistencia de hechos y el historial corto que se pasa al LLM. Implementa una interfaz
única tanto para despliegues con Postgres (LangGraph Checkpoints + tablas `session_facts`) como para
entornos locales sin infraestructura.

## Piezas principales
- `memory_adapter.py`
	- `MemoryAdapter`: expone `set_facts`, `get_facts`, `on_user_turn`, `on_assistant_turn`,
		`get_history_for_llm` y `get_backend_status`.
	- Integra `langgraph.checkpoint.postgres.PostgresSaver` o cae en `MemorySaver`/fallback in-process.
	- Administra el esquema `session_facts` (JSONB o KV) y aplica logging con `throttled_pg_log` para
		evitar spam.
	- Gestiona Response Diversity (opcional) cuando está disponible `memory_handler.response_diversity`.

## Flujo resumido
1. `agent_graph.ingest` solicita facts mediante `MemoryAdapter.get_facts(session_id)`.
2. Cada nodo puede actualizar `state.facts`; al llegar al nodo `memory`, se escribe la respuesta en
	 LangGraph + Postgres.
3. Las llamadas siguientes leen el historial reciente (hasta `MEMORY_MAX_TURNS_PROMPT`).

## Configuración relevante
- `PG_DSN`, `REQUIRE_PG_MEMORY`, `LANGGRAPH_CHECKPOINT_NS`, `LANGGRAPH_AUTO_SETUP`.
- `MEMORY_FACTS_LAYOUT` (`json` o `kv`), `MEMORY_MAX_TURNS_PROMPT`, `MEMORY_SUMMARY_EVERY`.
- `DIVERSITY_*` para ResponseDiversityManager.

## Operación y pruebas
- `MemoryAdapter` registra métricas básicas (`memory_fallback_used`, `diversity_hits`). Úsalas para
	health checks via `get_backend_status`.
- Ejecuta `pytest tests/test_short_term_memory.py tests/test_session.py` antes de tocar la lógica de
	escritura/lectura.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
- [README de Docker](../../docker/README.md)
- [README de scripts](../../readme/README.md)
