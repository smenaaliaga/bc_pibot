# QA Trace Scripts

Este directorio incluye scripts para ejecutar el grafo y registrar la traza.

## Requisitos

- Activar el entorno virtual de `pibot`.

```bash
source "/Users/hernanfernandez/Documents/09 - workspace/pibot/.venv/bin/activate"
```

## Ejecutar `qa.py` (una pregunta)

```bash
cd "/Users/hernanfernandez/Documents/09 - workspace/pibot/qa"
"/Users/hernanfernandez/Documents/09 - workspace/pibot/.venv/bin/python" qa.py "cual es el valor del imacec"
```

## Ejecutar `qa_interactive.py` (modo interactivo)

```bash
cd "/Users/hernanfernandez/Documents/09 - workspace/pibot/qa"
"/Users/hernanfernandez/Documents/09 - workspace/pibot/.venv/bin/python" qa_interactive.py
```

### Salir del modo interactivo

- Escribe `salir`, `exit` o `quit`.
- También puedes usar `Ctrl+C`.

## Salida de traza

El archivo de traza se escribe en:

- `qa/qa_trace.log`
