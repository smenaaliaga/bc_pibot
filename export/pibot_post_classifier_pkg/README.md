# pibot_post_classifier_pkg

Paquete exportable con los archivos que centralizan el post-procesamiento
del clasificador (reglas que mutan entidades / ruta) y el contrato de
instrucciones inyectadas al LLM.

## Contenido

```
pibot_post_classifier_pkg/
├── rules/
│   ├── __init__.py
│   └── post_classifier.py          # API consolidada (predicates + reglas)
│                                   # Incluye campo intent_cls en ResolvedEntities
├── orchestrator/
│   ├── data/
│   │   ├── _business_rules.py      # shim re-export desde rules.post_classifier
│   │   └── response.py             # _is_level_only_query con jerarquía
│   │                               # clasificador-driven (intent_cls, calc_mode)
│   └── graph/
│       └── nodes/
│           ├── ingest.py           # importa predicates desde rules.post_classifier
│           └── data.py             # propaga classification.intent → intent_cls
│                                   # al construir ResolvedEntities
└── tools/
    └── check_files_rules.py        # validador de contrato
```

## Uso

1. Descomprimir sobre la raíz del repo destino (`bc_pibot/`).
2. Verificar contrato:

   ```bash
   python tools/check_files_rules.py
   # o JSON
   python tools/check_files_rules.py --json
   ```

   Exit 0 = OK; 1 = símbolo/ancla faltante; 2 = error de import.

## Notas

- `orchestrator/data/response.py` y `orchestrator/graph/nodes/data.py`
  ahora están incluidos en el paquete: contienen el fix de detección de
  intent de NIVEL basado en clasificador (campo `intent_cls`).
- Tests existentes que importen `from orchestrator.data._business_rules
  import ResolvedEntities, apply_business_rules` siguen funcionando vía
  shim sin cambios.
- El campo `intent_cls` agregado a `ResolvedEntities` tiene default `None`,
  por lo que es backwards-compatible: código que no lo setea sigue
  funcionando idéntico al comportamiento previo.

## Cambios clave del drop

1. **`rules/post_classifier.py`** — `ResolvedEntities` ahora tiene un
   campo `intent_cls: Any = None` que captura el `intent` plano del
   clasificador (`value`, `variation`, `share`, etc.).
2. **`orchestrator/graph/nodes/data.py`** — extrae
   `classification.intent` y lo propaga a `ResolvedEntities(intent_cls=...)`.
3. **`orchestrator/data/response.py`** — `_is_level_only_query` ahora
   usa una jerarquía de 5 pasos (override léxico variación → override
   semántico variación → señales estructurales → `intent_cls='value'` con
   `calc_mode` neutro → fallback léxico "nivel de/del X"), eliminando
   falsos negativos en paráfrasis como "valor del IMACEC", "monto del
   PIB", "cuánto fue el IMACEC", "dame el IMACEC", "cifra del IMACEC".
