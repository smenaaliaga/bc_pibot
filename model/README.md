# Clonado de modelos Hugging Face

Para descargar cualquier modelo público de Hugging Face directamente a la carpeta `model/out/{modelo}`, usa el script incluido:

```bash
pip install huggingface_hub
python model/clone_hf_model.py <nombre-del-modelo> [ruta_destino]
```

Ejemplo para pibot-jointbert:

```bash
python model/clone_hf_model.py smenaaliaga/pibot-jointbert model/out/beto
```

Esto descargará todos los archivos del modelo a la carpeta indicada.

> Si necesitas autenticación (por ejemplo, para modelos privados), configura tu token de Hugging Face con:
> ```bash
> huggingface-cli login
> ```
# Modelo JointBERT

Estructura de directorios para el clasificador de intenciones y entidades usando Joint BERT.

## Estructura

```
model/
├── in/              # Código del modelo (arquitectura)
│   ├── __init__.py
│   ├── modeling_jointbert.py  # Clase JointBERT
│   └── module.py              # Clasificadores Intent/Slot
└── out/             # Modelos entrenados
    └── pibot_model_beto/      # Modelo en producción
        ├── config.json
        ├── intent_label.txt
        ├── model.safetensors
        ├── slot_label.txt
        └── training_args.bin
```

## Modelo (in/)

Contiene la **arquitectura** del modelo Joint BERT:

- **`modeling_jointbert.py`**: Implementación principal de `JointBERT`
  - Basado en `BertPreTrainedModel` de HuggingFace
  - Clasificación dual: intenciones + slots (etiquetado de entidades)
  - Soporta CRF opcional para mejor secuenciado

- **`module.py`**: Clasificadores auxiliares
  - `IntentClassifier`: Clasificador de intenciones
  - `SlotClassifier`: Etiquetador de slots (entidades)

### Dependencias

```bash
pip install transformers pytorch-crf torch
```

## Modelos Entrenados (out/)

Contiene los **pesos** del modelo ya entrenado:

### `pibot_model_beto/`

Modelo basado en BETO (BERT español) entrenado para:

**Intenciones:**
- `value`: Usuario pide valores/datos numéricos
- `methodology`: Usuario pregunta sobre metodología/definición

**Slots (Entidades):**
- `indicator`: Nombre del indicador (PIB, IMACEC, etc.)
- `period`: Período temporal (agosto 2024, trimestre, etc.)
- `frequency`: Frecuencia (mensual, trimestral, anual)
- `component`: Componente del indicador
- `seasonality`: Ajuste estacional

### Archivos del modelo:

- `config.json`: Configuración de BERT
- `model.safetensors`: Pesos del modelo
- `training_args.bin`: Argumentos de entrenamiento
- `intent_label.txt`: Lista de intenciones
- `slot_label.txt`: Lista de etiquetas BIO

## Uso

### Importar desde el proyecto:

```python
from orchestrator import get_predictor

# Cargar modelo (singleton)
predictor = get_predictor()

# Predecir
result = predictor.predict("cual fue el imacec de agosto 2024")
print(result)
# {
#     'intent': 'value',
#     'confidence': 0.98,
#     'entities': {'indicator': 'imacec', 'period': 'agosto 2024'}
# }
```

Ver [docs/JOINT_BERT_USAGE.md](../docs/JOINT_BERT_USAGE.md) para más detalles.

## Entrenamiento

Si necesitas reentrenar el modelo:

1. Prepara datos de entrenamiento en formato:
   ```
   texto\tintención\tslots_BIO
   ```

2. Entrena usando el script correspondiente (no incluido en este repo)

3. Guarda los pesos en `model/out/pibot_model_beto/`

## Notas Técnicas

- **Formato de slots**: Etiquetado BIO (Begin-Inside-Outside)
  - `B-indicator`: Inicio de indicador
  - `I-indicator`: Continuación de indicador
  - `O`: Fuera de entidad

- **Tokenización**: Usa BertTokenizer de HuggingFace
  - Subtokens se mapean a palabras originales
  - Padding a `max_seq_len` (default: 50)

- **CRF**: Conditional Random Field opcional
  - Mejora la coherencia de secuencias de etiquetas
  - Configurado en `training_args.bin`
