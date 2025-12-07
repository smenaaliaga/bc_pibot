# Rutas deterministas

Traducen intents y clasificaciones en decisiones concretas del grafo. Aquí se decide si la respuesta
se resuelve con lógica específica, con el flujo de datos o con rutas legacy.

## Archivos
- `intent_router.py`
	- Helpers para IMACEC/PIB: detección de mes específico, intervalos, toggles de métrica (m/m vs a/a),
		cambio de frecuencia, solicitud de gráficos y respuestas metodológicas breves.
	- `route_intents`: punto de entrada que combina intents configurables (`catalog/intents.json`) con
		heurísticas codificadas.
	- Handlers especializados como `handle_pib_quarter_year`, `_toggle_imacec_metric`, `_chart_marker_from_context`.
- `data_router.py`
	- `stream_data_flow`: envoltorio resiliente que delega en `data_flow.stream_data_flow_full` y cae en
		mensajes metodológicos cuando el fetch falla.
	- Expone `can_handle_data`/`_default_data_reply` para que otros nodos sepan si deben usar el flujo.

## Interacción con el grafo
1. Nodo `intent_shortcuts`: invoca `intent_router.route_intents`. Si el handler produce texto, el grafo
	 responde inmediatamente (ruta `direct`).
2. Nodo `data`: llama a `data_router.stream_data_flow` y maneja los chunks/markers resultantes.
3. Otros nodos pueden reutilizar helpers (por ejemplo, `data_router.DATA_BANNER`).

## Buenas prácticas
- Siempre que un handler acceda al contexto compartido `_last_data_context`, verifica que el dominio
	corresponda para evitar respuestas obsoletas.
- Usa logging estructurado (`[IMACEC_FETCH_ERROR]`, `[DATA_DELEGATE]`, etc.) para facilitar el tracing.
- Cuando agregues un handler nuevo, añade pruebas en `tests/test_orchestrator2.py` o crea un caso en
	`tools/test_orch2_chunk.py`.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README de catálogo](../catalog/README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
- [README de Docker](../../docker/README.md)
- [README de scripts](../../readme/README.md)
