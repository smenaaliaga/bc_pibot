# PIBot Joint BERT

Modelo JointBERT multi-cabeza para consultas económicas en español.

## Configuración del checkpoint

- **Task**: `pibimacecv5`
- **Base model**: `dccuchile/bert-base-spanish-wwm-cased`
- **Model type**: `beto`

## Cabezas de intención incluidas

- `calc_mode`
- `activity`
- `region`
- `investment`
- `req_form`

## Slot filling

- `slot_label.txt` para etiquetado BIO por token.

## Archivos importantes

- `model.safetensors` o `pytorch_model.bin`
- `config.json`
- `training_args.bin`
- tokenizer (`tokenizer_config.json`, `tokenizer.json` y/o vocabulario)
- `modeling_jointbert.py`, `module.py`, `__init__.py`
- `*_label.txt`

## Carga recomendada

```python
from transformers import AutoConfig, AutoTokenizer
from model_in import JointBERT

model_dir = "<ruta_local_o_repo_hf>"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)

# Cargar training_args.bin y labels antes de instanciar JointBERT
```
