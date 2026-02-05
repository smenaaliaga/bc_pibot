# Utilidades transversales

Helpers pequeños reutilizados en varias capas del orquestador.

## Archivos
- `pg_logging.py`: implementación de `throttled_pg_log` para agrupar errores de Postgres/Redis.
	Acepta metadata (`session_id`, `op`, `table`, `pool`) y mantiene un cache temporal para no imprimir
	el mismo error en cada ciclo.
- `followups.py`: `build_followups(context)` genera hasta tres sugerencias accionables considerando
	dominio, `has_table`, `series_id`, `year` y la metadata que emite el nodo de datos (sin depender de facts).

## Buenas prácticas
- Usa `throttled_pg_log` en vez de `logger.exception` directo cuando manejes reintentos de DB; te ahorra
	ruido en producción.
- Al construir follow-ups, pasa `has_table=True` solo si realmente se mostró una tabla para evitar
	recomendar acciones imposibles.
- Si agregas nuevas claves al contexto, actualiza los tests y la documentación en este archivo.

## Validación
- `pytest tests/test_utils_followups.py` (si existe) o añade casos en `tests/test_data_flow_fetch.py`.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
