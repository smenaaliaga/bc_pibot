# Rutas deterministas

Traducen intents y clasificaciones en decisiones concretas del grafo. Aquí se decide si la respuesta
se resuelve con lógica específica, con el flujo de datos o con RAG/fallback.

## Archivos
- `intent_router.py`
	- Helpers IMACEC/PIB: detección de mes puntual, intervalos (`marzo 2023 a mayo 2023`), toggles m/m vs
		y/y, cambio de frecuencia (T ↔ A), solicitud de gráficos, respuestas metodológicas cortas.
	- `route_intents`: combina `catalog/intents.json` + heurísticas codificadas y devuelve iterables para el
		nodo `intent_shortcuts`.
	- Handlers destacados: `handle_pib_quarter_year`, `_toggle_imacec_metric`, `_chart_marker_from_context`,
		`_fetch_imacec_series` (usa `orchestrator.data.get_series`).
- `data_router.py`
	- `stream_data_flow`: envoltorio resiliente sobre `data_flow.stream_data_flow_full`. Captura errores y
		devuelve mensajes metodológicos cuando la API falla.
	- Helpers `can_handle_data`, `DATA_BANNER`, `_default_data_reply` y `_build_retry_message` que otros
		módulos pueden reutilizar.

## Interacción con el grafo
1. El nodo `intent_shortcuts` invoca `route_intents`. Si el handler produce texto, el grafo responde
   inmediatamente por la rama `direct`.
2. Si no hubo respuesta determinista, la decisión cae en `route` → `data`/`rag`/`fallback`.
3. El nodo `data` llama `data_router.stream_data_flow` y consume sus markers.

## Buenas prácticas
- Antes de usar `_last_data_context`, comprueba que `domain` coincide; evita usar datos viejos para otro
  indicador.
- Los handlers deben ser iterables (generadores) para no bloquear el streaming. Usa `yield from` si
  necesitas delegar en otro generador.
- Registra logs con prefijos consistentes (`[IMACEC_*]`, `[DATA_ROUTER]`) para facilitar el tracing.
- Cuando agregues un handler nuevo, añade casos en `tests/test_routing.py` o en
  `tools/test_orch2_chunk.py`.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README de catálogo](../catalog/README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
