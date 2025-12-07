# Datos (series BCCh)

Implementa el flujo orientado a datos: responder metodología mínima (fase 1), buscar series por
ID/catálogo, construir tablas comparativas, emitir markers para CSV/gráficos y generar follow-ups.

## Archivos
- `data_flow.py`
	- `stream_data_flow_full`: pipeline streaming dividido en fases.
		1. Prompt metodológico con `_data_prompt` (explica qué se entregaría incluso si fallan los datos).
		2. Detección de dominio/serie/año (`_load_defaults_for_domain`, `_extract_year`).
		3. Fetch con reintentos (`_get_series_with_retry`).
		4. Construcción de tabla `_build_year_table`, exportación CSV (`_export_table_to_csv`) y marcador
			 `##CSV_DOWNLOAD_START`.
		5. Marker de gráfico `_emit_chart_marker` y follow-ups (`build_followups`).
	- `_last_data_context`: cachea dominio, serie, frecuencia y datos crudos para que otras rutas
		(ej. `intent_router`) puedan responder toggles sin volver a pegarle a la API.
- `method_router.py`: wrapper opcional para seguir usando el flujo legacy metodológico.

## Señales para la UI
- `##CSV_DOWNLOAD_START/END`: la UI genera el botón de descarga.
- `##CHART_START/END`: alimenta gráficos en Streamlit.
- Markdown enriquecido (tablas) + bloques con `series_id`, `freq`, etc.

## Extender
- Nuevos dominios: agrega defaults en `series/config_default.json` y reutiliza `_load_defaults_for_domain`.
- Nuevas salidas: emite markers con prefijo `##` para que la UI los detecte sin romper compatibilidad.
- Ajustes de follow-ups: modifica `orchestrator/utils/followups.py`.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README de catálogo](../catalog/README.md)
- [README raíz del proyecto](../../README.md)
- [README de Docker](../../docker/README.md)
- [README de pruebas](../../tests/README.md)
- [README de scripts](../../readme/README.md)
