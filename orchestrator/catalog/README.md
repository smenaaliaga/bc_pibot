# Catálogo de intents

Capa declarativa que cubre los atajos más frecuentes antes de consumir recursos LLM. Aquí se describen
preguntas estándar (último IMACEC, ventanas de meses, toggles de frecuencia) y el handler que debe
atenderlas.

## Componentes
- `intents.json`: cada intent define `name`, `priority`, `requiresDomain`, `patterns`, `handler` y
  parámetros específicos. Los placeholders `{MONTHS}`, `{YEARS}` y `{RANGE_X_Y}` se expanden al cargar
  el archivo, por lo que puedes mantener patrones legibles.
- `intent_router.py`: registra los handlers nombrados en el JSON, resuelve dependencias (por ejemplo,
  lecturas desde `_last_data_context`) y decide si devuelve respuesta directa o delega al grafo.

## Flujo
1. `intent_router.route_intents(…)` carga `intents.json` (cacheado con `functools.lru_cache`).
2. Cada intent evalúa sus `patterns` (regex) y, si hay match, invoca el handler.
3. El handler puede:
   - **Responder directo**: `yield` de texto/markers para que el consumidor (CLI, herramientas o un nodo
      personalizado) transmita la respuesta tal cual.
   - **Actualizar contexto**: escribir en `_last_data_context` para que otras rutas sepan qué serie o
      frecuencia usar.
   - **Forzar ruta**: retornar `None` pero establecer flags (`force_data`, `force_rag`) en el estado que
      maneje quien invoque al router.

## Añadir o modificar intents
1. Agrega la entrada en `intents.json`. Usa `priority` bajo para intents que deben evaluarse antes que
   el resto.
2. Crea o ajusta el handler en `routes/intent_router.py`; devuelve un iterable de strings para mantener
   el streaming, y documenta los logs (`[INTENT_*]`).
3. Si un intent requiere datos auxiliares (catálogo de series, defaults), reutiliza los helpers del
   módulo en vez de reimplementar lecturas JSON.
4. Valida con `pytest tests/test_chart_followups.py` (para handlers de gráficos) o con utilidades como
   `tools/run_small_tests.py intent`.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README del clasificador](../classifier/README.md)
- [README de rutas](../routes/README.md)
- [README raíz del proyecto](../../README.md)
