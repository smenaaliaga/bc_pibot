# QA Automator (Q220526)

Automatizador de QA por **tipo de consulta** que ejecuta el **mismo grafo
LangGraph que Streamlit** (no usa `qa_batch.py`) y evalúa cada respuesta
con **OpenAI como juez**, usando exactamente el modelo definido en el
`.env` del proyecto (`OPENAI_MODEL`).

## Ejecución única

Desde la raíz del proyecto (`bc_pibot/`):

```bash
python -m qa.automator.cli
```

Eso corre **todo de una sola vez**:

1. Carga `qa/input/tipos_consulta.yaml` (12 tipos de consulta).
2. Por cada tipo, genera ~50 variantes (ejemplos → catálogo → LLM).
3. Ejecuta cada variante sobre el grafo (paridad Streamlit).
4. Evalúa cada respuesta contra la rúbrica del tipo con el juez OpenAI.
5. Escribe reportes a `qa/output/runs/<timestamp>/`:
   - `variantes/variantes_tipo_XX_*.json`
   - `resultados_tipo_XX_*.json`
   - `reporte_tipo_XX_*.txt` y `.csv`
   - `master_report.txt`

## Requisitos

- `.env` del proyecto con:
  - `OPENAI_MODEL=...` (mismo modelo del chatbot, ej. `gpt-5.4-mini`)
  - `OPENAI_API_KEY=sk-...`
- Servicios locales que el chatbot necesita ya corriendo (ej. clasificador
  JointBERT en `PREDICT_URL`).
- Paquetes Python: `openai`, `pyyaml` (opcional pero recomendado).

Si `OPENAI_MODEL` no está definido, el automator **falla de inmediato**:
no usa defaults hardcodeados. Esto garantiza que el juez y el generador
de variantes corran con el mismo modelo del chatbot.

## Opciones (todas opcionales)

```bash
python -m qa.automator.cli --types 3 --n 5 --no-judge
```

| Opción | Default | Descripción |
|---|---|---|
| `--types-file PATH` | `qa/input/tipos_consulta.yaml` | YAML/JSON de tipos. |
| `--types N` | `0` (todos) | Limita a los primeros N tipos. |
| `--n N` | `50` | Variantes por tipo. |
| `--no-judge` | off | Salta el juez (sólo trazas). |
| `--out PATH` | `qa/output/runs/<ts>` | Carpeta de salida. |

## Sanity check rápido

Antes de gastar API en una corrida completa:

```bash
python -m qa.automator.cli --types 2 --n 3 --no-judge
```

Esto valida la **paridad con Streamlit** (sin invocar al juez): 6
preguntas, sólo se imprimen trazas y respuestas.

## Estructura

```
qa/automator/
  __init__.py
  _model.py              # carga .env + resuelve OPENAI_MODEL (sin default)
  schemas.py             # dataclasses
  graph_runner.py        # replica main.stream_fn (paridad Streamlit)
  catalog_expander.py    # lee orchestrator/catalog/catalog.json
  variant_generator.py   # examples → catálogo → LLM
  grader.py              # juez OpenAI
  loader.py              # persistencia
  runner.py              # orquesta tipo → variantes
  reporter.py            # txt/csv + master report
  cli.py                 # entry point único
qa/input/
  tipos_consulta.yaml    # 12 tipos con rúbrica
qa/output/runs/<ts>/     # resultados timestamped
```

## Notas

- **No** modifica el grafo ni los nodos. Sólo lee.
- **No** comparte estado entre variantes: cada pregunta usa un
  `thread_id` único (igual que Streamlit en sesiones nuevas).
- El juez devuelve JSON estricto (`response_format=json_object`) con
  `veredicto ∈ {ok,warn,fail}`, `score ∈ [0,1]`, criterios y brecha.
