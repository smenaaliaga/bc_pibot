# Rutas deterministas

Traducen intents y clasificaciones en decisiones concretas del grafo. Aquí se centralizan los
handlers que atienden follow-ups contextuales (ej. gráficos) y los envoltorios que preparan el flujo
de datos antes de invocar a LangGraph.

## Archivos
- `intent_router.py`
	- `_handle_chart_followup` usa solo metadata de los últimos turnos (sin facts persistidos) para detectar
		solicitudes como “otro gráfico del PIB”.
	- `route_intents` mantiene la capa declarativa (`catalog/intents.json`) para herramientas legacy; hoy
		el grafo principal confía en el IntentRouter ML del nodo `intent` (macro/intent/context), pero los handlers siguen
		estando disponibles para flujos CLI/tests.
	- Los helpers de IMACEC/PIB comparten utilidades con `data_router` para no duplicar llamadas al
		catálogo ni a `_last_data_context`.
- `data_router.py`
	- `stream_data_flow`: envoltorio resiliente sobre `data_flow.stream_data_flow_full`. Captura errores y
		devuelve mensajes metodológicos cuando la API falla.
	- Helpers `can_handle_data`, `DATA_BANNER`, `_default_data_reply` y `_build_retry_message` que otros
		módulos pueden reutilizar.

## Interacción con el grafo
1. El grafo decide la ruta en el nodo `intent` y valida follow-ups (gráficos) en `router`; los helpers
	de este módulo son reutilizables desde ahí o desde herramientas CLI.
2. Si no hubo respuesta determinista, la decisión cae en `route` → `data`/`rag`/`fallback`.
3. El nodo `data` llama `data_router.stream_data_flow` y consume sus markers.

## Buenas prácticas
- Antes de usar `_last_data_context`, comprueba que `domain` coincide; evita usar datos viejos para otro
  indicador.
- Los handlers deben ser iterables (generadores) para no bloquear el streaming. Usa `yield from` si
  necesitas delegar en otro generador.
- Registra logs con prefijos consistentes (`[IMACEC_*]`, `[DATA_ROUTER]`) para facilitar el tracing.
- Cuando agregues un handler nuevo, añade casos en `tests/test_routing.py` o en
	`tools/test_orch2_chunk.py`, aun si el handler solo se consume desde herramientas externas.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README de catálogo](../catalog/README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
