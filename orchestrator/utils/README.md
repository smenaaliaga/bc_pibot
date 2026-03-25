# Utilidades transversales

Helpers pequeños reutilizados en varias capas del orquestador.

## Archivos
- `pg_logging.py`: implementación de `throttled_pg_log` para agrupar errores de Postgres/Redis.
	Acepta metadata (`session_id`, `op`, `table`, `pool`) y mantiene un cache temporal para no imprimir
	el mismo error en cada ciclo.
- `http_client.py`: cliente HTTP compartido (`get_json`) usado por `bde_client`, `classifier_agent`, etc.
- `period_normalizer.py`: normalización de referencias temporales (meses, trimestres, años) para IMACEC.
- `indicator_normalizer.py`: detección y estandarización de indicadores (IMACEC vs PIB).
- `component_normalizer.py`: normalización de componentes del gasto.

## Buenas prácticas
- Usa `throttled_pg_log` en vez de `logger.exception` directo cuando manejes reintentos de DB; te ahorra
	ruido en producción.
- Si agregas nuevas claves al contexto, actualiza los tests y la documentación en este archivo.

## Validación
- `pytest tests/test_period_normalizer_quarters.py tests/test_normalizers_standard_names.py`

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
