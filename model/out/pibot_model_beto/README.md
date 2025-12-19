---
language: es
tags:
- intent-classification
- slot-filling
- joint-bert
- spanish
- economics
- chile
license: mit
---

# PIBot Joint BERT - BETO

Modelo Joint BERT entrenado para clasificación de intención y extracción de entidades (slot filling) en consultas sobre indicadores económicos del Banco Central de Chile.

## Descripción del Modelo

Este modelo está basado en la arquitectura Joint BERT que realiza simultáneamente:
1. **Clasificación de Intención**: Determina si la consulta busca valores (`value`) o información metodológica (`methodology`)
2. **Slot Filling**: Identifica y extrae entidades como indicadores, períodos, tipos de medida, sectores, etc.

### Modelo Base

- **Arquitectura**: BERT (dccuchile/bert-base-spanish-wwm-cased)
- **Idioma**: Español
- **Task**: pibimacec
- **Épocas de entrenamiento**: 20.0

## Uso

### Instalación

```bash
pip install torch transformers pytorch-crf
```

### Ejemplo de Uso

```python
from transformers import BertTokenizer
from modeling_jointbert import JointBERT
import torch

# Cargar modelo y tokenizer
model_dir = "smenaaliaga/pibot-jointbert-beto"  # Cambiar por tu repo
tokenizer = BertTokenizer.from_pretrained(model_dir)

# Cargar labels
intent_labels = ["methodology", "value"]
slot_labels = ["O", "B-indicator", "I-indicator", "B-period", "I-period", ...]

# Inicializar modelo (requiere código personalizado de JointBERT)
model = JointBERT.from_pretrained(
    model_dir,
    intent_label_lst=intent_labels,
    slot_label_lst=slot_labels
)

# Predecir
text = "cual fue el imacec de agosto 2024"
# ... (código de predicción)
```

## Datos de Entrenamiento

El modelo fue entrenado en un dataset especializado de consultas sobre:
- **IMACEC**: Indicador Mensual de Actividad Económica
- **PIB**: Producto Interno Bruto
- Sectores económicos (minería, comercio, industria, etc.)
- Períodos temporales (meses, trimestres, años)

### Etiquetas

**Intenciones:**
- `value`: Consultas sobre valores/datos específicos
- `methodology`: Consultas sobre metodología/definiciones

**Slots (entidades):**
- `indicator`: Indicador económico (IMACEC, PIB)
- `period`: Período temporal
- `measure_type`: Tipo de medida (variación, índice, etc.)
- `sector`: Sector económico
- `series_type`: Tipo de serie (original, desestacionalizada, tendencia-ciclo)

## Rendimiento

- **Intent Accuracy**: ~95%+
- **Slot F1-Score**: ~90%+

(Valores aproximados, ver logs de entrenamiento para métricas exactas)

## Limitaciones

- Entrenado específicamente para consultas sobre indicadores económicos chilenos
- Mejor rendimiento en consultas cortas-medianas (< 50 tokens)
- Puede tener dificultades con consultas muy ambiguas o fuera de dominio

## Cita

Si usas este modelo, por favor cita:

```bibtex
@misc{pibot-jointbert,
  author = {Banco Central de Chile},
  title = {PIBot Joint BERT - Modelo de Clasificación de Intención y Slot Filling},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/smenaaliaga/pibot-jointbert-beto}}
}
```

## Licencia

MIT License

## Más Información

- Paper original: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)
- Implementación base: [JointBERT](https://github.com/monologg/JointBERT)
