# pibot_series_interpreter

Modelo JointBERT para clasificación de características de series económicas del Banco Central de Chile.

## Descripción

Este modelo expone la **nueva taxonomía** y mantiene compatibilidad legacy:

| Cabeza (nueva) | Valores posibles | Descripción |
|--------|------------------|-------------|
| `calc_mode_cls` | `original`, `prev_period`, `yoy`, `contribution` | Tipo de variación |
| `frequency_cls` | `m`, `q`, `a` | Frecuencia temporal |
| `activity_cls` | `general`, `specific`, `none` | Sector/actividad |
| `region_cls` | `general`, `specific`, `none` | Región/territorio |
| `req_form_cls` | `general`, `specific`, `none` | Forma de la solicitud |

Legacy disponible (si el modelo lo entrega): `indicator`, `metric_type`, `seasonality`, `activity`, `frequency`, `calc_mode`, `req_form`.

## Descargar Modelo (Opcional)

El modelo `pibot-jointbert` (7-head JointBERT) está disponible en HuggingFace. **Sin descargarlo, el sistema funciona en modo heurístico**.

```bash
hf download smenaaliaga/pibot-jointbert \
  --local-dir ./models/pibot_series_interpreter/pibot-jointbert
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
print(result.frequency_cls.label)   # "m"
print(result.calc_mode_cls.label)   # "yoy"
print(result.req_form_cls.label)    # "specific"
print(result.frequency_cls.confidence)  # 0.95 (ej., None si heuristico)
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
# ... y asi para las 7 cabezas legacy

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
for field in ["frequency_cls", "calc_mode_cls", "req_form_cls"]:
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
- Cabezas: JointBERT legacy con 7 clasificadores (indicator, metric_type, seasonality, activity, frequency, req_form, calc_mode).
- El wrapper mapea esas salidas a la taxonomía nueva (`*_cls`).
