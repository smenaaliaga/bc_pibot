# pibot_intent_router

Clasificador de intencion y modo de contexto para consultas en espanol.

## Descripcion

Este modelo predice 3 dimensiones:
- `macro_cls`: `"1"` (in-domain) | `"0"` (out-of-domain)
- `intent_cls`: `value` (quiere datos) | `method` (quiere explicacion) | `other`
- `context_cls`: `standalone` (pregunta nueva) | `followup` (continuacion)

## Descargar modelo

Sin descargar, funciona en modo heuristico. Para mayor precision (ML):

```bash
hf login

hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir ./models/pibot_intent_router/all-MiniLM-L6-v2

hf download smenaaliaga/pibot-intent-router \
  --local-dir ./models/pibot_intent_router/pibot-intent-router
```
Esto deja los pesos en `pibot-intent-router/`:
```
pibot_intent_router/
 model.py
 README.md
 pibot-intent-router/
     intent_clf.joblib
     context_clf.joblib
     label_maps.json
```

## Uso

```python
from models.pibot_intent_router import IntentRouter

# Heuristico (sin modelo)
router = IntentRouter()
res = router.predict("Cual es el ultimo Imacec?")
print(res.intent_cls.label)         # value (ej.)
print(res.context_cls.label)        # standalone
print(res.macro_cls.label)          # 1
print(res.intent_cls.confidence)    # None (heuristico)

# Con modelo ML descargado
router = IntentRouter(model_path="pibot-intent-router")
res = router.predict("Cual es el ultimo Imacec?")
print(res.intent_cls.label)         # value (ej.)
print(res.intent_cls.confidence)    # 0.95
print(res.context_cls.label)        # standalone
print(res.context_cls.confidence)   # 0.87
print(res.macro_cls.label)          # 1
```

## Inspeccion del modelo (modo ML)

```python
# Path del modelo
router.model_path  # "pibot-intent-router"

# Labels/clases disponibles
router.model["label_maps"]
# → {"intent_cls": {"0": "value", "1": "method", "2": "other"}, 
#     "context_cls": {"0": "standalone", "1": "followup"}}

# Coeficientes de LogisticRegression (interpretabilidad)
intent_clf = router.model["intent_clf"]
intent_clf.coef_        # Pesos de las 384 dimensiones del embedding
intent_clf.classes_     # [0, 1] indices de clases

# Dimension del embedder
router.model["embedder"].get_sentence_embedding_dimension()  # 384

# Thresholds configurados (si existen)
router.model.get("thresholds")  # None o dict con umbrales por clase
```

Util para: debugging, interpretabilidad, validar carga correcta.

## Dependencias (modelo ML)
```
sentence-transformers
scikit-learn
joblib
```

## Arquitectura (modelo ML)
- Embedder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2)
- Clasificadores: 2 x LogisticRegression (intent_cls, context_cls)
- Flujo: texto -> embedding -> intent_logreg + context_logreg -> labels + confidence
