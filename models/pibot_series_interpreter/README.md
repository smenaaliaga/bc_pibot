# pibot_series_interpreter

Modelo JointBERT para clasificación de características de series económicas del Banco Central de Chile.

## Descripción

Este modelo clasifica 7 dimensiones de la consulta:

| Cabeza | Valores posibles | Descripción |
|--------|------------------|-------------|
| `indicator` | `imacec`, `pib` | Indicador económico |
| `metric_type` | `index`, `contribution` | Tipo de metrica |
| `seasonality` | `sa`, `nsa` | Ajuste estacional |
| `activity` | `total`, `imc_*`, `pib_*` | Sector/actividad |
| `frequency` | `m`, `q`, `a` | Frecuencia temporal |
| `req_form` | `latest`, `point`, `range` | Forma de la solicitud |
| `calc_mode` | `none`, `yoy`, `prev_period` | Tipo de variación |

## Descargar Modelo (Opcional)

El modelo `pibot-jointbert` (7-head JointBERT) está disponible en HuggingFace. **Sin descargarlo, el sistema funciona en modo heurístico**.

```bash
hf download smenaaliaga/pibot-jointbert \
  --local-dir ./src/models/pibot_series_interpreter/pibot-jointbert
```

Esto descarga los **pesos del modelo** a la carpeta `pibot-jointbert/`:

```
pibot_series_interpreter/
├── model.py                      ← Wrapper del modelo
├── README.md                     ← Este archivo
└── pibot-jointbert/              ← Pesos descargados de HuggingFace
    ├── modeling_jointbert.py     # Arquitectura JointBERT personalizada
    ├── module.py                 # 7 clasificadores multi-head
    ├── model.safetensors         # Pesos del modelo
    ├── config.json               # Configuración BERT
    ├── tokenizer.json            # Tokenizador
    ├── vocab.txt                 # Vocabulario
    └── *_label.txt               # 7 archivos de etiquetas (indicator, metric_type, etc.)
```

## Uso desde el pipeline

```python
from models.pibot_series_interpreter import SeriesInterpreter

# Con clasificador heurístico
interpreter = SeriesInterpreter()

# Con modelo entrenado
interpreter = SeriesInterpreter(model_path="pibot-jointbert")

# Predecir
result = interpreter.predict("Cuanto creció la economia en diciembre 2025?")
print(result.indicator.label)       # "imacec"
print(result.frequency.label)       # "m"
print(result.seasonality.label)     # "sa"
print(result.req_form.label)        # "point"
print(result.indicator.confidence)  # 0.95 (ej., None si heuristico)
```

## Inspeccion del modelo (modo DP)

Cuando el modelo JointBERT esta cargado, puedes acceder a:

```python
# Path del modelo
interpreter.model_path  # "pibot-jointbert"

# Label maps de las 7 cabezas
interpreter.label_maps
# → {"indicator": ["imacec", "pib"],
#    "metric_type": ["index", "contribution"],
#    "seasonality": ["sa", "nsa"],
#    ...}

# Modelo PyTorch (arquitectura completa)
model = interpreter.model["model"]
model  # JointBERT con encoder + 7 clasificadores

# Tokenizer (vocabulario, tokens especiales)
tokenizer = interpreter.model["tokenizer"]
tokenizer.vocab_size           # ~30,000
tokenizer.all_special_tokens   # ['[CLS]', '[SEP]', '[PAD]', ...]
tokenizer.tokenize("Imacec")   # ['Ima', '##ce', '##c']

# Config del BERT base
model.bert.config.hidden_size        # 768
model.bert.config.num_hidden_layers  # 12
model.bert.config.num_attention_heads # 12

# Clasificadores por cabeza (cada uno es un Linear layer)
model.indicator_classifier    # Linear(768 -> num_indicator_labels)
model.metric_type_classifier  # Linear(768 -> num_metric_type_labels)
# ... y asi para las 7 cabezas

# Numero total de parametros
sum(p.numel() for p in model.parameters())  # ~110M parametros
```

**Casos de uso:**
- **Fine-tuning**: acceder a capas especificas para congelar/entrenar
- **Debugging**: verificar dimensiones, vocabulario, estructura
- **Interpretabilidad**: analizar attention weights del BERT
- **Optimizacion**: cuantizacion, pruning de cabezas poco usadas

**Ejemplo - analizar predicciones por cabeza:**
```python
result = interpreter.predict("Imacec mensual de diciembre")
for field in ["indicator", "frequency", "seasonality", "req_form"]:
    attr = getattr(result, field)
    print(f"{field}: {attr.label} (conf: {attr.confidence:.3f})")
```

## Dependencias (modelo DP - Transformer)
```
torch
transformers
sentencepiece
```

## Arquitectura (modelo DP - Transformer)
- Encoder: BERT base (embeddings + self-attention) compartido.
- Cabezas: 7 clasificadores softmax independientes (indicator, metric_type, seasonality, activity, frequency, req_form, calc_mode).
- Flujo: texto -> tokenizer -> encoder -> cabezas -> etiquetas + confidencia por cabeza.
