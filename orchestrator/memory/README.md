# Memoria conversacional

Gestiona el historial corto que se envía al LLM y los checkpoints que LangGraph necesita para retomar
una sesión. El viejo concepto de `facts` quedó en desuso: hoy solo se almacenan turnos, metadata del
assistant y snapshots compactos del `AgentState`. El módulo sigue exponiendo los métodos legacy para
compatibilidad, pero el grafo ya no los invoca.

## Piezas principales
- `memory_adapter.py`
	- `MemoryAdapter`: expone `on_user_turn`, `on_assistant_turn`, `get_history_for_llm`,
		`get_recent_turns`, `get_window_for_llm`, `save_checkpoint`, `load_checkpoint` y
		`get_backend_status`. Los métodos `set_facts`/`get_facts` siguen presentes para legacy, pero el grafo
		moderno no los utiliza.
	- Integra `langgraph.checkpoint.postgres.PostgresSaver`; si Postgres no está disponible, cae en
		`MemorySaver` o en el fallback in-process `_InProcessMemoryAdapter`.
	- Administra los esquemas `session_turns` (historial estructurado) y, para compatibilidad, `session_facts`
		(JSONB o KV); usa `utils/pg_logging.throttled_pg_log` para controlar el spam de errores.
	- Puede habilitar Response Diversity (cuando `memory_handler.response_diversity` está presente).
	- `session_turns` guarda rol, contenido, metadata y timestamp para cada turno, y permite recuperar
		ventanas eficientes para el LLM sin depender únicamente del fallback en memoria.

## Flujo resumido
1. `ingest` pide una ventana corta con `get_window_for_llm(session_id)` y, si existe, fusiona el último
	checkpoint del agente.
2. Durante la conversación, los nodos pueden adjuntar metadata al turno del assistant (ej. dominio del
	gráfico detectado en la salida) usando `memory_adapter.on_assistant_turn`.
3. El nodo `memory` registra la respuesta final y almacena un checkpoint liviano (`question`, `output`,
	`route_decision`, entidades relevantes). En local, se usan las estructuras in-process.
4. Para el próximo turno, `get_history_for_llm` delega en `get_window_for_llm`, que obtiene hasta
	`MEMORY_MAX_TURNS_PROMPT` mensajes directamente desde Postgres (`session_turns`) o el fallback local.

## Configuración relevante
- `PG_DSN`, `REQUIRE_PG_MEMORY`, `LANGGRAPH_CHECKPOINT_NS`, `LANGGRAPH_AUTO_SETUP`.
- `MEMORY_MAX_TURNS_PROMPT`, `MEMORY_SUMMARY_EVERY` controlan la ventana enviada al LLM.
- `MEMORY_MAX_TURNS_STORE` regula cuántos turnos se conservan en el fallback in-process;
	`MEMORY_MAX_CHECKPOINTS` delimita los checkpoints retenidos cuando `PostgresSaver` no está disponible.
- `DIVERSITY_*` (si usas Response Diversity) sigue funcionando igual.

Consulta `docs/README_memory.md` para ver el estado de las migraciones (`session_facts` JSONB) y los
pasos para ejecutar los scripts en `docker/postgres/migrations`.

## Operación y pruebas
- `MemoryAdapter.get_backend_status()` expone banderas (`using_pg`, `fallback_sessions`, métricas) que puedes
	mostrar en la UI o logs.
- Tests útiles: `pytest tests/test_memory_adapter.py tests/test_session.py`.
- El nodo `memory` deduce metadata de gráficos leyendo los markers `##CHART_START/END` en la salida y la adjunta
	a los turnos más recientes, evitando depender de facts globales.
- Para validar manualmente, ejecuta `tools/debug_graph_stream.py --no-pg` y revisa que el fallback se
  informe correctamente.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
