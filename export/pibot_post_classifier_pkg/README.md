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
├── orchestrator/
│   ├── data/
│   │   └── _business_rules.py      # shim re-export desde rules.post_classifier
│   └── graph/
│       └── nodes/
│           └── ingest.py           # importa predicates desde rules.post_classifier
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

- `orchestrator/data/response.py` NO se incluye: se asume idéntico en el
  repo destino. El validador comprueba sus anclas críticas.
- Tests existentes que importen `from orchestrator.data._business_rules
  import ResolvedEntities, apply_business_rules` siguen funcionando vía
  shim sin cambios.
