# Catálogo de intents

Capa declarativa que cubre los atajos más frecuentes antes de consumir recursos LLM. Aquí se describen
preguntas estándar (último IMACEC, ventanas de meses, toggles de frecuencia) y el handler que debe
atenderlas.

## Componentes
- `intents.json`: cada intent define `name`, `priority`, `requiresDomain`, `patterns`, `handler` y
  parámetros específicos. Los placeholders `{MONTHS}`, `{YEARS}` y `{RANGE_X_Y}` se expanden al cargar
  el archivo, por lo que puedes mantener patrones legibles.


## Flujo
1. Los nodos del grafo cargan `intents.json` para mapear patrones y metadata declarativa.
2. Cada intent evalúa sus `patterns` (regex) y resuelve handler dentro del flujo actual del orquestador.
3. El handler puede responder directo o actualizar contexto para etapas posteriores del pipeline.

## Añadir o modificar intents
1. Agrega la entrada en `intents.json`. Usa `priority` bajo para intents que deben evaluarse antes que
   el resto.
2. Crea o ajusta el handler en el flujo actual del orquestador; devuelve un iterable de strings para mantener
   el streaming, y documenta los logs (`[INTENT_*]`).
3. Si un intent requiere datos auxiliares (catálogo de series, defaults), reutiliza los helpers del
   módulo en vez de reimplementar lecturas JSON.
4. Valida con pruebas de ruteo/follow-up vigentes y utilidades como `tools/run_small_tests.py`.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README del clasificador](../classifier/README.md)
- [README raíz del proyecto](../../README.md)
