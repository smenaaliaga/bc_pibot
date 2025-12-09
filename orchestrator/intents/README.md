# Clasificación de intents

Aquí convergen las dos capas que definen el rumbo inicial de cada pregunta: clasificación LLM y
heurísticas deterministas. La salida combinada (`ClassificationResult` + `intent_info`) alimenta los
nodos `classify` e `intent_shortcuts` del grafo.

## Archivos
- `classifier_agent.py`: punto de entrada. Construye el prompt usando
	`prompts/query_classifier.py`, ejecuta el modelo (por defecto `gpt-4o-mini`) y parsea la función
	`classify_economic_query`.
- `intent_classifier.py`: reglas adicionales (regex, palabras clave) para casos comunes (IMACEC, PIB,
	frecuencia) antes de llamar al LLM.
- `joint_intent_classifier.py`: clasificador BIO opcional que detecta entidades (indicador, sector,
	frecuencia) para enriquecer `intent_info`.
- `intent_store.py` / `intent_memory.py`: cache ligero por sesión que evita repetir decisiones cuando
	el usuario hace follow-ups inmediatos.

## Flujo resumido
1. `classify_question_with_history(question, history)` aplica heurísticas deterministas.
2. Si aún se necesita LLM, se dispara `LLMAdapter` con el prompt generado por `query_classifier.py`.
3. Se devuelve `ClassificationResult` (tipado) y `history_text` (un string concatenado para ruteos
	 posteriores).
4. `build_intent_info(result)` empaqueta la información para `intent_shortcuts` y para los prompts de
	 datos/RAG.

## Buenas prácticas
- Cuando añadas campos nuevos en `ClassificationResult`, actualiza tanto las dataclasses como el schema
	enviado al LLM y las pruebas que mockean la respuesta.
- Extiende `intent_classifier.py` para cubrir heurísticas muy frecuentes antes de gastar tokens.
- Usa el logging existente (`[CLASSIFIER]...`) para auditar salidas; está replicado en consola y en el
	archivo `logs/run_main.log`.

## Testing y validación
- `pytest tests/test_orchestrator2.py::test_classifier_branch`
- `pytest tests/test_routing.py::test_intent_shortcuts_short_circuit`
- `tools/run_small_tests.py classifier` ejecuta un subconjunto orientado a esta capa.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README de catálogo](../catalog/README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
