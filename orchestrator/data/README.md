# Datos (series BCCh)

Pipeline encargado de: (1) responder la parte metodológica mínima, (2) consultar la API oficial del
Banco Central, (3) construir tablas comparativas, (4) emitir markers de CSV/gráfico y (5) proponer
follow-ups coherentes.

## Archivos
- `data_flow.py`
	- `stream_data_flow_full`: orquesta todas las fases en streaming.
	- `_get_series_with_retry`: envoltorio resiliente sobre `orchestrator.data.get_series.get_series_api_rest_bcch`.
	- `_last_data_context`: cachea dominio, serie, frecuencia, datos crudos y errores para que otros
		módulos (`intent_router`, follow-ups, toggles) reaccionen sin repetir llamadas a la API.
- `method_router.py`: compatibilidad con el flujo legacy metodológico (permite togglear según flags).

## Fases del pipeline
| Paso | Descripción | Señales |
| --- | --- | --- |
| 1. Metodología | `_data_prompt` (LangChain) explica qué se mostrará y cita al BCCh aunque el fetch falle. | Chunks Markdown simples. |
| 2. Selección serie/año | `_load_defaults_for_domain`, `_extract_year`, `_detect_chart_request`; se combinan defaults de `series/config_default.json` con pistas del usuario. | Actualiza `_last_data_context`. |
| 3. Fetch | `_get_series_with_retry` arma `firstdate/lastdate`, llama a la API y registra errores con `_record_fetch_error`. | Logs `[DATA_FETCH_*]`, `fetch_error` en contexto. |
| 4. Tabla + CSV | `_build_year_table` (usa `get_series.build_year_comparison_table_text`), `_export_table_to_csv`, `_emit_csv_download_marker`. | Markers `##CSV_DOWNLOAD_START/END`, ruta a `/logs/exports/*.csv`. |
| 5. Visualización | `_emit_chart_marker` agrega datos estructurados para la UI (serie, puntos, etiquetas). | Marker `##CHART_START/END`. |
| 6. Follow-ups | `utils.followups.build_followups` crea hasta tres sugerencias específicas. | Texto Markdown enumerado. |

## Contrato con la UI
- `##CSV_DOWNLOAD_START/END` → Streamlit muestra botón de descarga.
- `##CHART_START/END` → se construyen los gráficos (puntos + metadata).
- Bloque de metadatos (código, frecuencia, unidad, URL) se imprime antes del resumen.
- Cuando `_record_fetch_error` guarda un fallo, se expone un mensaje amigable y el banner
  `data_router.DATA_BANNER` avisa que se están procesando datos reales.

## Extender
- **Nuevos dominios**: añade defaults en `series/config_default.json` y, si necesitas más lógica, crea
  helpers `_load_defaults_for_<dominio>` que reutilicen `_get_series_with_retry`.
- **Markers adicionales**: respeta el prefijo `##` y encapsula el bloque entre `START/END` para que la UI
  siga siendo backwards-compatible.
- **Follow-ups a medida**: ajusta `utils/followups.py` o pasa flags adicionales en `_last_data_context`.
- **Errores controlados**: usa `_record_fetch_error` para que el usuario vea qué falló sin inundar logs.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README de catálogo](../catalog/README.md)
- [README raíz del proyecto](../../README.md)
- [README de Docker](../../docker/README.md)
- [README de pruebas](../../tests/README.md)
