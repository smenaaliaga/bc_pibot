# Modelos de Clasificación

Pipeline de 2 clasificadores secuenciales para consultas de series económicas del Banco Central de Chile.

## Uso Rápido

```python
from models.pibot_intent_router import IntentRouter
from models.pibot_series_interpreter import SeriesInterpreter

# Sin modelos descargados (modo heurístico)
router = IntentRouter()
interpreter = SeriesInterpreter()

# Con modelos ML descargados (mayor precisión)
router = IntentRouter(model_path="pibot-intent-router")
interpreter = SeriesInterpreter(model_path="pibot-jointbert")

# Uso
intent = router.predict("¿Cuál es el último Imacec?")
# → intent.label="value", context_mode.label="standalone"

series = interpreter.predict("Imacec mensual desestacionalizado")
# → indicator.label="imacec", frequency.label="m", seasonality.label="sa"

# Con modelos ML: también retorna confidence scores
intent = router.predict("¿Cuál es el último Imacec?")
print(f"Intent: {intent.intent.label} (confidence: {intent.intent.confidence:.2f})")
# → Intent: value (confidence: 0.95)

series = interpreter.predict("Imacec mensual")
if series.frequency.confidence and series.frequency.confidence < 0.7:
  print("⚠️ Baja confianza en frequency, verificar query")
```

## Estructura

```
models/
├── README.md                          # Este archivo
├── base.py                            # Tipos y clases base (contratos de los modelos)
├── pibot_intent_router/               # Modelo 1: Clasificador de intención
│   ├── __init__.py
│   ├── model.py                       # IntentRouter (wrapper)
│   ├── README.md                      # Documentación detallada
│   └── pibot-intent-router/           # ← Pesos del modelo (descargar de HuggingFace)
│       ├── context_clf.joblib         # Clasificador de contexto
│       ├── intent_clf.joblib          # Clasificador de intención
│       └── label_maps.json            # Mapeo de etiquetas
└── pibot_series_interpreter/          # Modelo 2: Clasificador de características
    ├── __init__.py
    ├── model.py                       # SeriesInterpreter (wrapper)
    ├── README.md                      # Documentación detallada
    └── pibot-jointbert/               # ← Pesos del modelo (descargar de HuggingFace)
        ├── modeling_jointbert.py      # Arquitectura JointBERT personalizada
        ├── module.py                  # 7 clasificadores multi-head
        ├── model.safetensors          # Pesos del modelo
        ├── config.json                # Configuración BERT
        ├── tokenizer.json             # Tokenizador
        ├── vocab.txt                  # Vocabulario
        └── *_label.txt                # 7 archivos de etiquetas por cabeza
```

## Descargar Modelos

Por defecto usa **modo heurístico** (reglas). Para mayor precisión, descarga los modelos:

```bash
# Autenticación
huggingface-cli login

# IntentRouter (SentenceTransformer + LogisticRegression)
huggingface-cli download smenaaliaga/pibot-intent-router \
  --local-dir ./src/models/pibot_intent_router/pibot-intent-router

# SeriesInterpreter (JointBERT multi-head)
huggingface-cli download smenaaliaga/pibot-jointbert \
  --local-dir ./src/models/pibot_series_interpreter/pibot-jointbert
```

## base.py - Contratos de Tipos

Define las estructuras de salida y valores válidos:

**LabeledScore (común):**
```python
@dataclass
class LabeledScore:
  label: str
  confidence: Optional[float] = None
```

* Confidence Scores: 0.0-1.0 (probabilidad _softmax_)

**IntentRouterOutput:**
```python
@dataclass
class IntentRouterOutput:
  intent: LabeledScore      # label: "value"|"methodology"
  context_mode: LabeledScore  # label: "standalone"|"followup"
```

**SeriesInterpreterOutput:**
```python
@dataclass
class SeriesInterpreterOutput:
  indicator: LabeledScore     # label: "imacec"|"pib"
  metric_type: LabeledScore   # label: "index"|"contribution"
  seasonality: LabeledScore   # label: "sa"|"nsa"
  activity: LabeledScore      # label: "total"|"imc_*"|"pib_*"
  frequency: LabeledScore     # label: "m"|"q"|"a"
  calc_mode: LabeledScore     # label: "none"|"yoy"|"prev_period"
  req_form: LabeledScore      # label: "latest"|"point"|"range"
```

## Modelos

### 1. IntentRouter

**Clasifica:** Intención del usuario
- `intent`: `"value"` (quiere datos) | `"methodology"` (quiere explicación)
- `context_mode`: `"standalone"` (pregunta nueva) | `"followup"` (continuación)

**Modos:**
- **Heurístico:** Palabras clave ("metodología", "cómo se calcula" → methodology)
- **ML:** SentenceTransformer + 2 LogisticRegression

**Archivos del modelo:**
- `intent_clf.joblib`, `context_clf.joblib`, `label_maps.json`

### 2. SeriesInterpreter

**Clasifica:** Características de la serie (7 dimensiones)
- `indicator`, `metric_type`, `seasonality`, `activity`, `frequency`, `calc_mode`, `req_form`

**Modos:**
- **Heurístico:** Regex y palabras clave
- **DP:** JointBERT con 7 cabezas de clasificación

**Archivos del modelo:**
- `model.safetensors`, `config.json`, `tokenizer.json`, `modeling_jointbert.py`, `module.py`
- 7 archivos `*_label.txt` (uno por cabeza)

## Documentación Detallada

- [pibot_intent_router/README.md](pibot_intent_router/README.md) - Detalles del IntentRouter
- [pibot_series_interpreter/README.md](pibot_series_interpreter/README.md) - Detalles del SeriesInterpreter

