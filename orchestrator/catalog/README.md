# Catálogo de intents

Capas deterministas que complementan al clasificador LLM. Aquí viven los intents declarativos
que se evalúan **antes** de llegar al grafo para cubrir preguntas frecuentes (último IMACEC,
PIB trimestral, ventanas de meses, etc.).

## Componentes
- `intents.json`: cada intent define `name`, `priority`, `requiresDomain`, `patterns`, `handler` y
  parámetros opcionales. Las plantillas `{MONTHS}` / `{YEARS}` se expanden automáticamente.
- `intent_router.py`: implementa los handlers mencionados en el catálogo y la lógica para decidir
  cuándo un intent corto responde directamente, cuándo enriquece el contexto y cuándo delega al
  flujo de datos/RAG.

## Flujo
1. El nodo `classify` etiqueta la consulta con `ClassificationResult`.
2. `intent_router.route_intents` carga `intents.json` (cacheado con `lru_cache`).
3. Cada intent ejecuta sus `patterns` (regex) y, si coincide, llama al handler indicado.
4. El handler puede:
	- devolver una respuesta directa (`yield` de texto/markers),
	- fijar contexto compartido (`_last_data_context`),
	- o pedir que el grafo continúe por `data` / `rag`.

## Añadir o modificar intents
1. Declara la entrada en `intents.json` (usa `priority` bajo para que se evalúe primero).
2. Implementa/actualiza el handler en `intent_router.py`. Procura que devuelva iterables de
	textos para streaming.
3. Si el intent necesita datos de catálogo externo (series), añade los imports al inicio del
	handler y reuse `_load_*` helpers.
4. Ejecuta `tools/run_small_tests.py intent` o `pytest tests/test_orchestrator2.py::test_intent_shortcuts`
	para validar que no rompiste el ruteo determinista.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README de intents](../intents/README.md)
- [README de rutas](../routes/README.md)
- [README raíz del proyecto](../../README.md)
- [README de Docker](../../docker/README.md)
- [README de pruebas](../../tests/README.md)
