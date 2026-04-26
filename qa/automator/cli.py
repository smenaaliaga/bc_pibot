"""CLI: una sola ejecución corre TODO (variantes + grafo + juez + reportes).

Uso típico:
    python -m qa.automator.cli

Opciones (todas opcionales):
    --types-file PATH   Archivo YAML/JSON de tipos (default: qa/input/tipos_consulta.yaml)
    --types N           Limita a los primeros N tipos (debug)
    --n N               Variantes por tipo (default: 50)
    --no-judge          No invoca el juez OpenAI (sólo trazas)
    --out PATH          Carpeta de salida (default: qa/output/runs/<timestamp>/)
"""
from __future__ import annotations
import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, List

from ._model import get_openai_model

# Silenciar warnings ruidosos que ya conocemos (clasificador siempre va por API).
logging.getLogger("orchestrator.classifier.classifier_agent").setLevel(logging.ERROR)
from .loader import OUTPUT_DIR, save_results, save_variants
from .reporter import write_master_report, write_tipo_report
from .runner import run_tipo
from .schemas import ResultadoVariante, TipoConsulta
from .variant_generator import DEFAULT_TYPES_FILE, generate_variants, load_tipos


def _make_run_dir(custom: Path | None) -> Path:
    if custom is not None:
        custom.mkdir(parents=True, exist_ok=True)
        return custom
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QA Automator (Q220526)")
    parser.add_argument("--types-file", type=Path, default=DEFAULT_TYPES_FILE)
    parser.add_argument("--types", type=int, default=0, help="Limita a los primeros N tipos (0 = todos)")
    parser.add_argument("--n", type=int, default=50, help="Variantes por tipo")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    print(f"[QA Automator] Modelo OpenAI: {get_openai_model()}")
    tipos = load_tipos(args.types_file)
    if args.types > 0:
        tipos = tipos[: args.types]
    print(f"[QA Automator] Tipos a evaluar: {len(tipos)}")

    run_dir = _make_run_dir(args.out)
    variants_dir = run_dir / "variantes"
    print(f"[QA Automator] Run dir: {run_dir}")

    per_tipo: Dict[int, List[ResultadoVariante]] = {}

    def _progress(tipo: TipoConsulta, i: int, n: int, res):
        if res.veredicto is not None:
            estado = res.veredicto.veredicto.upper()
        elif res.trace.error:
            estado = "ERROR"
        else:
            estado = "SIN-JUEZ"
        print(
            f"  - tipo {tipo.id:02d} [{i:02d}/{n:02d}] estado: {estado} :: "
            f"{res.variante.question[:90]}"
        )

    for tipo in tipos:
        print(f"\n[Tipo {tipo.id:02d}] {tipo.label} (scope={tipo.scope})")
        variantes = generate_variants(tipo, n_total=args.n)
        save_variants(tipo, variantes, variants_dir)
        results = run_tipo(tipo, variantes, use_judge=not args.no_judge, progress=_progress)
        per_tipo[tipo.id] = results
        save_results(tipo, results, run_dir)
        write_tipo_report(tipo, results, run_dir)

    master = write_master_report(per_tipo, tipos, run_dir)
    print(f"\n[QA Automator] Listo. Master report: {master}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
