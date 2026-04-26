"""Persistencia: variantes generadas y resultados."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from .schemas import ResultadoVariante, TipoConsulta, Variante

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "qa" / "input"
OUTPUT_DIR = ROOT / "qa" / "output"


def save_variants(tipo: TipoConsulta, variantes: List[Variante], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"variantes_tipo_{tipo.id:02d}_{tipo.name}.json"
    payload = {
        "tipo_id": tipo.id,
        "tipo_name": tipo.name,
        "label": tipo.label,
        "scope": tipo.scope,
        "n": len(variantes),
        "variantes": [asdict(v) for v in variantes],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_variants(path: Path) -> List[Variante]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Variante(**v) for v in data.get("variantes", [])]


def save_results(tipo: TipoConsulta, results: Iterable[ResultadoVariante], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"resultados_tipo_{tipo.id:02d}_{tipo.name}.json"
    items = []
    for r in results:
        items.append(
            {
                "variante": asdict(r.variante),
                "trace": asdict(r.trace),
                "veredicto": asdict(r.veredicto) if r.veredicto else None,
            }
        )
    path.write_text(
        json.dumps(
            {"tipo_id": tipo.id, "tipo_name": tipo.name, "results": items},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
