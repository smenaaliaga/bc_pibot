# Memoria conversacional

Gestiona la persistencia de hechos (`session_facts`) y el historial corto que se envía al LLM. Brinda
una interfaz única tanto para despliegues con Postgres (LangGraph Checkpoints + tablas JSONB) como
para ambientes locales sin infraestructura.

## Piezas principales
- `memory_adapter.py`
	- `MemoryAdapter`: expone `set_facts`, `get_facts`, `on_user_turn`, `on_assistant_turn`,
		`get_history_for_llm`, `get_recent_turns`, `get_window_for_llm`, `save_checkpoint`,
		`load_checkpoint` y `get_backend_status`.
	- Integra `langgraph.checkpoint.postgres.PostgresSaver`; si Postgres no está disponible, cae en
		`MemorySaver` o en el fallback in-process `_InProcessMemoryAdapter`.
	- Administra los esquemas `session_facts` (JSONB o KV) y `session_turns` (historial estructurado)
		y usa `utils/pg_logging.throttled_pg_log` para
		controlar el spam de errores.
	- Puede habilitar Response Diversity (cuando `memory_handler.response_diversity` está presente).
	- `session_turns` guarda rol, contenido, metadata y timestamp para cada turno, y permite recuperar
		ventanas eficientes para el LLM sin depender únicamente del fallback en memoria.

## Flujo resumido
1. `ingest` pide los facts (`get_facts(session_id)`) y los agrega a `state.context`.
2. Durante la conversación, los nodos pueden actualizar `state.facts`.
3. El nodo `memory` registra la respuesta final (turno de assistant) y almacena un checkpoint liviano
	del estado (`question`, `output`, `route_decision`, `facts`). En local, se usan las estructuras in-process.
4. Para el próximo turno, `get_history_for_llm` delega en `get_window_for_llm`, que obtiene hasta
	`MEMORY_MAX_TURNS_PROMPT` mensajes directamente desde Postgres (`session_turns`) o el fallback local.

## Configuración relevante
- `PG_DSN`, `REQUIRE_PG_MEMORY`, `LANGGRAPH_CHECKPOINT_NS`, `LANGGRAPH_AUTO_SETUP`.
- `MEMORY_FACTS_LAYOUT` (`json` | `kv`), `MEMORY_MAX_TURNS_PROMPT`, `MEMORY_SUMMARY_EVERY`.
- `MEMORY_MAX_TURNS_STORE` controla cuántos turnos se conservan en el fallback in-process; `MEMORY_MAX_CHECKPOINTS`
	delimita los checkpoints retenidos cuando `PostgresSaver` no está disponible.
- `MEMORY_FACTS_COMPRESSION`, `DIVERSITY_*` (si usas Response Diversity).

Consulta `docs/README_memory.md` para ver el estado de las migraciones (`session_facts` JSONB) y los
pasos para ejecutar los scripts en `docker/postgres/migrations`.

## Operación y pruebas
- `MemoryAdapter.get_backend_status()` expone banderas (`using_pg`, `fallback_sessions`, métricas) que puedes
	mostrar en la UI o logs.
- Tests útiles: `pytest tests/test_memory_adapter.py tests/test_session.py`.
- Las rutas de gráficos almacenan `chart_last_domain` y `chart_last_ts` como facts para evitar reusar gráficos
	antiguos cuando el usuario cambia de intención; estos facts se limpian automáticamente cuando se envía una
	respuesta que no corresponde a un gráfico.
- Para validar manualmente, ejecuta `tools/debug_graph_stream.py --no-pg` y revisa que el fallback se
  informe correctamente.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
