# README de reglas de negocio y gatillantes de preguntas

Este documento resume **todas las lógicas de negocio** implementadas en `rules/business_rule.py`, cómo se gatillan desde preguntas del usuario y cómo impactan el flujo de respuesta (`GENERAL_RESPONSE` vs `SPECIFIC_RESPONSE`) en el nodo de datos.

Objetivo: que cualquier persona que intervenga el código tenga claridad de **qué modificar**, **dónde** y **qué efecto funcional produce**.

---

## 1) Dónde vive cada parte

- Reglas principales: `rules/business_rule.py`
- Aplicación de reglas en pipeline: `orchestrator/graph/nodes/data.py`
- Render del tipo de respuesta:
  - `orchestrator/data/flow_data.py`
  - `orchestrator/data/response.py`

---

## 2) Flujo general (input → reglas → output)

1. Llega pregunta + clasificación (`predict_raw`) al nodo `data`.
2. Se construye `data_params` (indicator, frequency, period, req_form_cls, etc.).
3. Se aplican reglas de `business_rule.py` en este orden:
   - `resolve_calc_mode_cls`
   - `classify_headers`
   - `build_metadata_response`
   - `apply_latest_update_period`
   - `resolve_pib_annual_validity`
4. Con metadatos + parámetros resueltos, se consulta serie y se arma payload de `flow_data`.
5. `flow_data` determina si va a:
   - `general_response` (sin serie específica)
   - `specific_response` / `specific_point_response` (con serie)

---

## 3) Reglas de negocio en `business_rule.py`

## 3.1 `resolve_region_value(value)`

**Campo afectado:** `data_params.region_value`

- **Input:** entidad de región en string/lista (ej. `"region de los rios"`, `["region", "rios"]`).
- **Condición:** siempre que exista valor de región.
- **Regla:**
  - Si es lista/string tokenizado, prioriza segundo token cuando existe.
  - Normaliza a minúsculas.
- **Output:** región normalizada (ej. `rios`) o `None`.
- **Impacto en respuesta:** indirecto (influye en metadata key y serie seleccionada).

---

## 3.2 `classify_seasonality(question)`

**Campo afectado:** `seasonality`

- **Input:** texto pregunta.
- **Condición de gatillo:** regex de estacionalidad.
- **Reglas:**
  - `sa` si detecta `desestacionalizado` / `ajustado estacional`.
  - `nsa` si detecta `sin desestacionalizar` / `original`.
- **Output:** `sa`, `nsa` o `None`.
- **Impacto en respuesta:** cambia metadata key, serie y potencial tipo de dato mostrado.

---

## 3.3 `classify_price(question)`

**Campo afectado:** `price`

- **Input:** texto pregunta.
- **Condición de gatillo:** regex de precio.
- **Reglas:**
  - `co` si detecta `precios corrientes`.
  - `en` si detecta `encadenados`.
  - Default: `co`.
- **Output:** `co` o `en`.
- **Impacto en respuesta:** cambia metadata key y serie final (corriente vs encadenada).

---

## 3.4 `classify_history(question, indicator)`

**Campo afectado:** `history`

- **Input:** pregunta + indicador.
- **Condición de gatillo:** aplica fuerte para `pib`.
- **Reglas (PIB):**
  - `inf_historic` si detecta `histórica` o años < 1996.
  - si no, `2018`.
- **Output:** `inf_historic` o `2018`.
- **Impacto en respuesta:** cambia metadata key (familia histórica vs base 2018).

---

## 3.5 `resolve_calc_mode_cls(...)`

**Campo afectado:** `data_params.calc_mode_cls`

- **Input:**
  - `question`
  - `calc_mode_cls` (clasificador)
  - `intent_cls`
  - `req_form_cls`
  - `frequency`
- **Condiciones y prioridad:**
  1. Normaliza alias explícitos (`interanual`, `annual`, `anual` → `yoy`; `mom/qoq` → `prev_period`; `original/value` → `original`).
  2. Si pregunta contiene patrón interanual → `yoy`.
  3. Si contiene patrón período anterior → `prev_period`.
  4. Defaults de negocio para `intent=value` + `req_form` (`latest|point|specific_point|range`) → `yoy`.
- **Output:** `yoy`, `prev_period`, `original` (normalizado).
- **Impacto en respuesta:**
  - Cambia columna/variación que se muestra (`yoy` vs `prev_period`).
  - Afecta metadata key y serie elegida.

---

## 3.6 `classify_headers(question, predict_raw, enabled)`

**Campos afectados:** `seasonality`, `price`, `history`

- **Input:** pregunta + `predict_raw` + flags.
- **Condición:** se ejecuta en `data.py` con flags habilitados para los 3 campos.
- **Regla:** combina extracción de indicador + clasificadores regex.
- **Output:** dict con headers clasificados.
- **Impacto en respuesta:** rellena vacíos en `data_params` antes de construir metadata key.

---

## 3.7 `build_metadata_key(data_params)`

**Campo afectado:** `metadata_key`

- **Input:** `data_params` completo.
- **Regla:** concatena 15 campos en orden fijo con `::`.
- **Output:** key única de búsqueda en `metadata_q.json`.
- **Impacto en respuesta:** determina si hay serie específica o respuesta general.

---

## 3.8 `build_metadata_response(data_params)`

**Campos afectados:** `metadata_response` (`label`, `serie_default`, `title_serie_default`, `sources_url`, `latest_update`)

- **Input:** `data_params` + catálogo `metadata_q.json`.
- **Condiciones:**
  - Busca match exacto por key.
  - Si falla y `req_form in {point, range}`, aplica fallback de req_form (`point→latest/range`, `range→range/latest`).
  - Si `latest_update` vacío y req_form=`latest`, setea default técnico por indicador (`imacec`, `pib`).
- **Output:** metadata resuelta o `{match: None}`.
- **Impacto en respuesta:**
  - Si `serie_default = none` → `GENERAL_RESPONSE`.
  - Si `serie_default` válida → `SPECIFIC_RESPONSE`.

---

## 3.9 `apply_latest_update_period(data_params, metadata_response)`

**Campo afectado:** `data_params.period`

- **Input:** frecuencia + req_form + latest_update.
- **Condición:** solo si `req_form_cls == latest`.
- **Regla:** deriva período desde `latest_update`:
  - mensual: mes actual
  - trimestral: trimestre móvil
  - anual: último año completo (si no cerró diciembre, usa año anterior)
- **Output:** override de período `[start, end]` o `None`.
- **Impacto en respuesta:** define ventana temporal efectiva de consulta para `latest`.

---

## 3.10 `resolve_pib_annual_validity(data_params, metadata_response)`

**Campos afectados:** validación anual PIB (`annual_validation`) y eventualmente `period`.

- **Input:** indicador/frecuencia/req_form/period/latest_update.
- **Condición de aplicación:**
  - `indicator == pib`
  - `frequency == a`
  - `req_form in {latest, point}`
- **Reglas:**
  - Calcula `max_valid_year` según completitud de `latest_update`.
  - `latest`: fuerza período al último año anual válido.
  - `point`:
    - si año pedido > `max_valid_year` → inválido.
    - si año pedido <= `max_valid_year` → válido y fija período del año pedido.
- **Output:** dict con:
  - `applies`, `is_valid`, `requested_year`, `max_valid_year`, `resolved_period`, `message`.
- **Impacto en respuesta:**
  - Si inválido: corta flujo y retorna mensaje específico de no-publicación anual.
  - Si válido: mantiene flujo normal con período ajustado.

---

## 4) Reglas complementarias en `data.py` (ligadas a negocio)

Aunque no están en `business_rule.py`, son parte del comportamiento final:

1. **Normalización de período anual para fetch** (`_normalize_period_for_frequency`):
   - si `target_frequency = a` y `req_form != range`, colapsa al año de referencia.
   - si `req_form = range`, **mantiene todo el rango**.

2. **Selección de frecuencia efectiva de serie**:
   - se infiere frecuencia nativa desde `series_id`.
   - evita sobreescribir cuando el usuario pide anual (`target_frequency == a`).

3. **Mensaje de anual no publicado** (PIB):
   - texto final y URL de fuente salen del bloque de validación anual.

---

## 5) Tipos de preguntas y cuándo se gatillan

Esta tabla conecta intención/pregunta con reglas y tipo de respuesta.

| Tipo de pregunta | Señales típicas | Reglas principales | Campos más impactados | Tipo de respuesta esperado |
|---|---|---|---|---|
| Valor general | `cual es el valor del ...` | `resolve_calc_mode_cls`, `classify_headers`, `build_metadata_response` | `calc_mode_cls`, `price`, `history`, `metadata_key` | General o específica según `serie_default` |
| Valor punto (año/fecha) | `... 2024`, `... del 2023` | + `resolve_pib_annual_validity` (si PIB anual) | `period`, `annual_validation` | `SPECIFIC_RESPONSE` (1 fila) o mensaje de no disponibilidad |
| Valor rango | `entre 2020 y 2023` | `resolve_calc_mode_cls`, metadata fallback, normalización de período anual sin colapso | `req_form_cls`, `period`, `series_fetch_args` | `SPECIFIC_RESPONSE` (N filas) |
| Latest | sin fecha o con `último` implícito | `apply_latest_update_period` | `period` override | General o específica |
| Regional | `region ...` / `regional` | `resolve_region_value` + metadata key | `region_cls`, `region_value` | General si no hay serie específica |
| Crecimiento | `cuanto crecio` | `resolve_calc_mode_cls` (yoy) | `calc_mode_cls=yoy` | Específica con columna variación consistente |

---

## 6) Mapa input/output por regla (resumen rápido para intervenir código)

| Regla | Input mínimo | Campo(s) tocado(s) | Output | Efecto visible |
|---|---|---|---|---|
| `classify_seasonality` | `question` | `seasonality` | `sa/nsa/None` | Cambia serie/tabla |
| `classify_price` | `question` | `price` | `co/en` | Cambia serie |
| `classify_history` | `question`,`indicator` | `history` | `2018/inf_historic` | Cambia metadata key |
| `resolve_calc_mode_cls` | `question`,`calc_mode`,`intent`,`req_form`,`freq` | `calc_mode_cls` | `yoy/prev_period/original` | Cambia narrativa + variación |
| `classify_headers` | `question`,`predict_raw` | `seasonality/price/history` | dict headers | Completa faltantes |
| `build_metadata_response` | `data_params` | `metadata_response` | serie/label/source/latest | Define general vs específica |
| `apply_latest_update_period` | `req_form`,`freq`,`latest_update` | `period` | `[start,end]` | Ajusta ventana latest |
| `resolve_pib_annual_validity` | `indicator`,`freq`,`req_form`,`period`,`latest_update` | `annual_validation`, `period` | válido/ inválido + período | Mensaje anual no publicado o período anual correcto |

---

## 7) ¿Qué modificar según objetivo?

### A) Cambiar redacción / tono
- Punto/rango/específica: `orchestrator/data/response.py`
- Mensaje PIB anual no disponible: `orchestrator/graph/nodes/data.py` (bloque `annual_validation`)

### B) Cambiar cuándo una regla aplica
- Condiciones de negocio: `rules/business_rule.py`

### C) Cambiar mapeo de serie/cobertura
- Claves y metadatos: `orchestrator/catalog/metadata_q.json`

### D) Cambiar general vs específica
- Si hay o no `serie_default` desde metadata.
- Lógica de derivación en `build_metadata_response`.

---

## 8) Casos de uso solicitados

A continuación, cómo funciona actualmente en términos de reglas/resultado esperado.

### 8.1 `cual es el valor del pib`
- **Input detectado:** `intent=value`, típicamente `req_form=latest`, `indicator=pib`.
- **Reglas clave:**
  - `resolve_calc_mode_cls` → `yoy` (default de negocio para value/latest)
  - `classify_headers` completa `price/history` si faltan
  - `build_metadata_response`
  - `apply_latest_update_period` (si latest)
- **Salida esperada:** depende de metadata (`general` o `specific`) con variación coherente.

### 8.2 `cual es el valor del imacec`
- Similar al caso PIB, pero con indicador `imacec`.
- `build_metadata_response` determina serie/fuente IMACEC.
- Si latest sin fecha, aplica ventana por `latest_update` correspondiente.

### 8.3 `cual es el valor del pib 2025` (mejorar redacción)
- **Input detectado:** `point` + `frequency=a` + `indicator=pib`.
- **Regla clave:** `resolve_pib_annual_validity`.
- **Condición:** si 2025 > último año anual completo publicado.
- **Salida actual:** mensaje de no disponibilidad anual con enlace:
  - “El PIB anual correspondiente al año 2025 aún no se encuentra publicado...”.

### 8.4 `cual es el valor del pib 2023`
- **Input:** point anual válido.
- **Reglas:** validación anual PIB retorna válido y fija período 2023.
- **Salida:** `SPECIFIC_RESPONSE` con una fila del año 2023.

### 8.5 `cual es el valor del pib entre 2020 y 2023`
- **Input:** `req_form=range` (normalmente frecuencia trimestral si no se pide anual explícito).
- **Reglas:**
  - calc_mode default `yoy`
  - no colapso de rango en fetch
- **Salida:** `SPECIFIC_RESPONSE` con múltiples filas del rango (trimestral).
- **Intro vigente:** “A continuación te muestro los valores del PIB entre el 1er trimestre del 2020 y el 4to trimestre del 2023.”

### 8.6 `cual es el valor del pib anual entre 2020 y 2023`
- **Input:** `req_form=range`, `frequency=a`.
- **Reglas clave:**
  - no colapso anual por ser rango
  - fetch anual completo del intervalo
- **Salida:** `SPECIFIC_RESPONSE` con 4 filas anuales (2020, 2021, 2022, 2023).

### 8.7 `cual es el pib de la region de los rios`
- **Input:** señal regional.
- **Reglas:** `resolve_region_value` + metadata key regional.
- **Salida:** según metadata regional; puede resultar `GENERAL_RESPONSE` si no hay serie específica.

### 8.8 `cuanto crecio el pib el 2024`
- **Input:** intención de crecimiento + punto 2024.
- **Regla clave:** `resolve_calc_mode_cls` favorece `yoy`.
- **Salida:** `SPECIFIC_RESPONSE` con último valor de 2024 y variación anual coherente.

---

## 9) Checklist de intervención segura

Antes de modificar reglas:

1. Verifica si el cambio es de negocio (`business_rule.py`) o de redacción (`response.py`).
2. Revisa impacto en `metadata_key` (puede cambiar serie/fuente/response type).
3. Ejecuta QA mínimo de regresión:
   - PIB point anual válido
   - PIB point anual inválido
   - PIB range trimestral
   - PIB range anual
   - Caso general regional
4. Confirma en `qa_trace.log`:
   - `data_params`
   - `metadata_key`
   - `series_fetch_args`
   - `TYPE_RESPONSE`

---

## 10) Notas

- Las reglas de negocio se aplican **antes** del render de respuesta.
- El tipo de respuesta final se decide principalmente por la metadata (`serie_default`) y por el `req_form` resultante.
- Para cambios de cobertura (qué consultas tienen serie específica), intervenir `metadata_q.json` además de reglas.
