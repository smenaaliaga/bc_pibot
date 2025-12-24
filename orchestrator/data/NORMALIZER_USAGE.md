# Uso del Normalizer en flow_data.py

Este documento mapea qué funciones en `flow_data.py` dependen de las salidas del `entity_normalizer`.

## Estructura de salida del normalizer

Ver documentación completa en `orchestrator/classifier/entity_normalizer.py::EntityNormalizer.normalize()`

```python
normalized = {
    'period': {
        'start_date': date,
        'end_date': date,
        'granularity': str,  # 'month', 'quarter', 'year'
        'period_key': str,    # '2024-08' o '2024-Q3'
        'label': str,
    },
    'indicator': {
        'standard_name': str,      # 'imacec', 'pib', 'ipc'
        'normalized': str,
        'text_normalized': str,
        'original': str,           # (opcional)
        'label': str,              # (opcional)
    },
    'component': {
        'standard_name': str,      # 'mineria', 'comercio'
        'normalized': str,
        'original': str,
    },
    'sector': { ... },             # Alias de 'component'
    'seasonality': str,            # 'desestacionalizado' o 'original'
}
```

## Funciones que consumen normalized

### 1. `_build_period_context(normalized)`
**Accede a:**
- `normalized.get("period")` → dict completo

**Campos usados del period dict:**
- `start_date` / `startDate`
- `end_date` / `endDate`
- `granularity` / `period_type`
- `period_key`
- `label`

**Propósito:** Construye contexto temporal con firstdate/lastdate/candidates

---

### 2. `_has_indicator_info(normalized, entities)`
**Accede a:**
- `normalized.get("indicator")` → puede ser dict o None

**Campos usados:**
- `standard_name`, `normalized`, `original`, `text_normalized`, `label` (todos via `_coerce_meta_value`)

**Propósito:** Verifica si hay información de indicador disponible

---

### 3. `_has_sector_info(normalized)`
**Accede a:**
- `normalized.get("sector")` → puede ser dict o None
- `normalized.get("component")` → puede ser dict o None

**Campos usados:**
- `standard_name`, `normalized`, `original` (via `_coerce_meta_value`)

**Propósito:** Verifica si hay información de sector/componente

---

### 4. `_resolve_indicator_label(normalized, entities, indicator_context)`
**Accede a:**
- `normalized.get('indicator')` → dict esperado

**Campos usados (en orden de prioridad):**
1. `standard_name`
2. `normalized`
3. `original`
4. `text_normalized`
5. `label`

**Propósito:** Resuelve etiqueta legible del indicador para mensajes

---

### 5. `_extract_seasonality(normalized)`
**Accede a:**
- `normalized['seasonality']` → puede ser dict o str

**Campos usados (si es dict):**
- `standard_name`, `normalized`, `original`, `text_normalized`, `label`

**Propósito:** Extrae valor de estacionalidad normalizado

---

## Flujo de uso típico

```python
# 1. Clasificación produce entities y normalized
classification = classify_question_with_history(...)
entities = classification.entities          # raw JointBERT output
normalized = classification.normalized      # normalized via entity_normalizer

# 2. get_full_context extrae normalized
ctx = get_full_context(classification=classification, ...)
ctx["normalized"] = normalized

# 3. Funciones helper acceden a campos específicos
indicator_label = _resolve_indicator_label(normalized, entities)
period_ctx = _build_period_context(normalized)
seasonality = _extract_seasonality(normalized)
has_indicator = _has_indicator_info(normalized, entities)
has_sector = _has_sector_info(normalized)
```

## Campos opcionales vs requeridos

**Siempre presentes (si la entidad fue detectada):**
- `indicator['standard_name']` ✓
- `period['start_date']`, `period['end_date']` ✓
- `component['standard_name']` ✓
- `seasonality` (str) ✓

**Opcionales (pueden faltar):**
- `indicator['original']`, `indicator['label']`
- `period['label']` (fallback construido si falta)

**Estrategia de fallback:**
Las funciones como `_resolve_indicator_label` y `_coerce_meta_value` implementan cascadas de fallbacks para manejar campos opcionales faltantes.
