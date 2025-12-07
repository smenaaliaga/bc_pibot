# Utilidades transversales

Helpers pequeños que se reutilizan en múltiples módulos del orquestador.

## Archivos
- `pg_logging.py`: `throttled_pg_log` evita inundar los logs con errores repetidos de Postgres.
	Accepta meta (`session_id`, `op`, `table`, `pool`) y mantiene estado entre llamadas.
- `followups.py`: `build_followups(context)` genera hasta tres sugerencias accionables en base al
	dominio actual, métricas vistas y hechos almacenados.

## Buenas prácticas
- Cuando registres errores de DB, usa `throttled_pg_log` en vez de `logger.exception` directo para
	mantener los logs legibles.
- Al llamar `build_followups`, siempre provee `has_table=True` sólo si realmente se mostró una tabla;
	así evitamos sugerencias fuera de contexto.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
- [README de Docker](../../docker/README.md)
- [README de scripts](../../readme/README.md)
