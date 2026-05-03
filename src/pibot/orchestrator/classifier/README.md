# Clasificación de intents

Aquí convergen las dos capas que definen el rumbo inicial de cada pregunta: clasificación JointBERT y
heurísticas deterministas. La salida combinada (`ClassificationResult` + `intent_info`) alimenta los
nodos `classify` e `intent_shortcuts` del grafo.

## Archivos
- `classifier_agent.py`: punto de entrada. Ejecuta JointBERT (modelo local) + normalización de entidades y
	construye `intent_info` para el resto del grafo.
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
4. `build_intent_info(result)` empaqueta `intent`, `score`, `entities`, `normalized`, `indicator` y
	`spans` para `intent_shortcuts` y para los prompts de datos/RAG.

## Contrato AgentState (INTENT_URL / PREDICT_URL)

El nodo `classify` deja dos superficies de consumo para el resto del grafo:

- `state["classification"]`: objeto `ClassificationResult` tipado.
- `state["intent_info"]`: dict serializable para nodos y prompts.

### 1) Datos de `INTENT_URL` (autoritativo para ruteo)

Respuesta esperada del servicio:

```json
{
	"macro": { "label": 1, "confidence": 0.97 },
	"intent": { "label": "value", "confidence": 0.94 },
	"context": { "label": "standalone", "confidence": 0.89 }
}
```

Se guarda en:

- `classification.intent_raw`
- `intent_info["intent_raw"]`

Y además se proyecta en campos normalizados para consumo directo:

- `classification.intent`, `classification.macro`, `classification.context`
- `intent_info["intent"]`, `intent_info["macro"]`, `intent_info["context"]`

### 2) Datos de `PREDICT_URL` (señales para nodo DATA)

Respuesta esperada del servicio:

```json
{
	"text": "...",
	"routing": { ... },
	"interpretation": {
		"intents": {
			"calc_mode": { "label": "yoy", "confidence": 0.99 },
			"activity": { "label": "specific", "confidence": 0.99 },
			"region": { "label": "none", "confidence": 0.99 },
			"investment": { "label": "none", "confidence": 0.99 },
			"req_form": { "label": "range", "confidence": 0.99 }
		},
		"entities_normalized": {
			"indicator": ["imacec"],
			"seasonality": ["sa"],
			"frequency": ["m"]
		}
	}
}
```

Se guarda completo en:

- `classification.predict_raw`
- `intent_info["predict_raw"]`

Además, el clasificador expone atajos ya extraídos:

- `classification.calc_mode`, `classification.activity`, `classification.region`,
	`classification.investment`, `classification.req_form`
- `classification.normalized` / `intent_info["normalized"]` (incluye `entities_normalized`)

### 3) Patrón recomendado de consumo en nodos

```python
classification = state.get("classification")
intent_info = state.get("intent_info") or {}

predict_raw = getattr(classification, "predict_raw", None) or intent_info.get("predict_raw") or {}
payload_root = predict_raw.get("interpretation") if isinstance(predict_raw.get("interpretation"), dict) else predict_raw

predict_intents = payload_root.get("intents") or {}
entities_normalized = payload_root.get("entities_normalized") or {}

intent_raw = getattr(classification, "intent_raw", None) or intent_info.get("intent_raw") or {}
```

Regla operativa:

- Usa `INTENT_URL` (`intent_raw` / campos `intent-macro-context`) para decidir ruteo.
- Usa `PREDICT_URL` (`predict_raw.interpretation.intents` y `entities_normalized`) para parametrizar
	el nodo `data` (calc mode, req form, actividad, etc.).

## Buenas prácticas
- Cuando añadas campos nuevos en `ClassificationResult`, actualiza tanto las dataclasses como el schema
	enviado al LLM y las pruebas que mockean la respuesta.
- Extiende `intent_classifier.py` para cubrir heurísticas muy frecuentes antes de gastar tokens.
- Usa el logging existente (`[CLASSIFIER]...`) para auditar salidas; está replicado en consola y en el
	archivo `logs/run_main.log`.

## Testing y validación
- `pytest tests/test_orchestrator2.py::test_classifier_branch`
- `pytest tests/test_followup_routing.py`
- `tools/run_small_tests.py classifier` ejecuta un subconjunto orientado a esta capa.

## Documentación relacionada
- [README del orquestador](../README.md)
- [README de catálogo](../catalog/README.md)
- [README raíz del proyecto](../../README.md)
- [README de pruebas](../../tests/README.md)
