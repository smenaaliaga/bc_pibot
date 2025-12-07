# Clasificación de intents

Esta carpeta reúne clasificadores LLM + heurísticas tradicionales. Su salida (`ClassificationResult`
e `intent_info`) alimenta el grafo y los módulos deterministas.

## Archivos
- `classifier_agent.py`: wrapper principal que prepara el prompt del clasificador, llama al modelo
	(`gpt-4o-mini` por defecto) e interpreta la respuesta JSON (`classify_economic_query`).
- `intent_classifier.py`: heurísticas adicionales y shortcuts basados en regex para cobertura rápida.
- `joint_intent_classifier.py`: modelo BIO que detecta entidades (indicador, sector, frecuencia) para
	enriquecer `intent_info`.
- `intent_store.py` / `intent_memory.py`: almacenamiento ligero para intents previamente detectados
	(permite reusar decisiones dentro de una sesión).

## Flujo
1. El grafo junta `question` + `history` y llama `classify_question_with_history`.
2. El resultado incluye la estructura `ClassificationResult` y un `history_text` resumido.
3. `build_intent_info` consolida la información para el nodo posterior (`intent_shortcuts`).

## Testing y ajustes
- Usa `pytest tests/test_orchestrator2.py::test_classifier_branch` para validar cambios.
- Si agregas nuevos campos a la clasificación, actualiza los `TypedDict` y el `prompt` en `classifier_agent.py`.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README de catálogo](../catalog/README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
- [README de Docker](../../docker/README.md)
- [README de scripts](../../readme/README.md)
