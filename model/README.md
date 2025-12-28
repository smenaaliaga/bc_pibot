# Model — Arquitectura, Pesos y Scripts (local-first)

Estructura recomendada para mantener código del modelo, artefactos y utilidades de manera clara:

```
model/
├── src_model/           # Código importable del modelo (arquitectura)
│   ├── __init__.py
│   └── modeling_jointbert.py   # Clase `JointBERT`
├── weights/             # Pesos entrenados y snapshots de modelos
│   └── pibot_model_beto/
│       ├── config.json
│       ├── intent_label.txt
│       ├── model.safetensors
│       ├── slot_label.txt
│       └── training_args.bin
├── tokenizers/          # (Opcional) Tokenizadores/base models clonados localmente
│   └── bert-base-spanish-wwm-cased/
└── scripts/             # Scripts CLI (no importables como librería)
  └── clone_hf_model.py
```

La aplicación intenta primero cargar el tokenizer desde una ruta local (`BERT_MODEL_NAME`) y, si no existe, cae a carga remota.

## Clonado de modelos/tokenizers (local-first)

Usa el script en `model/scripts/clone_hf_model.py` para clonar repos de Hugging Face dentro del proyecto.

```bash
# Instalar dependencia si hace falta
uv run pip install huggingface_hub

# 1) Clonar BETO (tokenizer/base) → model/tokenizers/
uv run python model/scripts/clone_hf_model.py dccuchile/bert-base-spanish-wwm-cased tokenizers

# 2) Clonar un modelo entrenado (si aplica) → model/weights/
uv run python model/scripts/clone_hf_model.py smenaaliaga/pibot-jointbert weights
```

Configura variables de entorno para uso local y fallback:

```powershell
# PowerShell (Windows)
$env:BERT_MODEL_NAME = "model/tokenizers/bert-base-spanish-wwm-cased"
$env:BERT_REMOTE_REPO = "dccuchile/bert-base-spanish-wwm-cased"   # fallback si falta local
$env:JOINT_BERT_MODEL_DIR = "model/weights/pibot_model_beto"
uv run streamlit run main.py
```

> Para modelos privados, inicia sesión:
> ```bash
> huggingface-cli login
> ```
## Arquitectura (src_model/)

Contiene la **arquitectura** del modelo Joint BERT:

- **`modeling_jointbert.py`**: Implementación principal de `JointBERT`
  - Basado en `BertPreTrainedModel` de Hugging Face
  - Cabezas de clasificación: intenciones + slots (entidades BIO)
  - Soporte opcional de CRF para coherencia de secuencias

### Dependencias

```bash
pip install transformers pytorch-crf torch
```

### Uso local-first en la app

1. Descarga el modelo/tokenizer con el script (ver arriba).
2. Indica la ruta local en la app:
  - Desde la barra lateral: fija "Modelo BERT" a `model/tokenizers/bert-base-spanish-wwm-cased` (o la carpeta descargada) y haz clic en "Nueva sesión".
  - O usando variable de entorno antes de ejecutar:

```powershell
# PowerShell (Windows)
$env:BERT_MODEL_NAME = "model/tokenizers/bert-base-spanish-wwm-cased"
uv run streamlit run main.py
```

La app intentará cargar el tokenizer localmente (`local_files_only=True`). Si no está disponible:
- Si `HF_LOCAL_ONLY=false` y `BERT_REMOTE_REPO` está definido, clonará el tokenizador en `model/tokenizers/` y lo cargará localmente.
- En último recurso, hará carga remota directa (cache de Hugging Face) si el clon falla o está deshabilitado.

## Pesos Entrenados (weights/)

Contiene los **pesos** del modelo entrenado y también puede contener **snapshots** descargados desde Hugging Face para uso offline.

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

Si clonaste un modelo HF, también verás archivos de tokenizer como `tokenizer.json`, `vocab.txt`, `tokenizer_config.json` para operar en modo offline.

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

3. Guarda los pesos en `model/weights/pibot_model_beto/`

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
