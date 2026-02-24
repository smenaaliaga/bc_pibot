# PIBot Joint BERT - 7 Heads

Modelo Joint BERT para clasificación multi-cabeza de consultas sobre indicadores económicos.

## Cabezas de Clasificación

El modelo predice simultáneamente 7 atributos (legacy):
- **indicator**: Indicador económico (ej: imacec, pib)
- **metric_type**: Tipo de métrica (ej: index, level)
- **calc_mode**: Modo de cálculo (ej: yoy, mom)
- **seasonality**: Ajuste estacional (ej: sa, nsa)
- **req_form**: Forma de solicitud (ej: latest, historical)
- **frequency**: Frecuencia (ej: m, q, a)
- **activity**: Actividad/Sector (ej: total, agriculture)

El wrapper del proyecto mapea estas salidas a la taxonomía nueva (`*_cls`).

## Uso

### Opción 1: Local (Recomendado para máxima compatibilidad)

```python
from load_local_model import PIBotPredictor

predictor = PIBotPredictor("path/to/model")
result = predictor.predict("cual fue el pib del último trimestre")
print(result)
```

### Opción 2: Desde Hugging Face Hub

```python
from load_local_model import PIBotPredictor

# Descargar y usar
predictor = PIBotPredictor("username/pibot-jointbert")
result = predictor.predict("cual fue el imacec")
print(result)
```

### Línea de comandos

```bash
python load_local_model.py --model_dir path/to/model --text "tu consulta"
```

## Estructura del Checkpoint

```
model_dir/
├── model.safetensors              # Pesos del modelo
├── config.json                    # Configuración de BERT
├── training_args.bin              # Argumentos de entrenamiento
├── tokenizer.json                 # Tokenizer rápido
├── tokenizer_config.json
├── vocab.txt
├── modeling_jointbert.py          # Arquitectura custom
├── module.py                      # Clasificadores custom
├── __init__.py
├── *_label.txt                    # Labels para cada cabeza (7 archivos)
└── README.md
```

## Detalles Técnicos

- **Base Model**: dccuchile/bert-base-spanish-wwm-cased (BETO)
- **Framework**: PyTorch + Transformers
- **Formato de pesos**: SafeTensors
- **Tokenizer**: AutoTokenizer con use_fast=True

## Licencia

[Especificar licencia]

## Autor

[Tu información]
