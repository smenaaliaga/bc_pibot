"""Genera reportes legibles (txt + csv) por tipo y master."""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .schemas import ResultadoVariante, TipoConsulta


def _veredicto_counts(results: List[ResultadoVariante]) -> Counter:
    c: Counter = Counter()
    for r in results:
        v = r.veredicto.veredicto if r.veredicto else "n/a"
        c[v] += 1
    return c


def write_tipo_report(tipo: TipoConsulta, results: List[ResultadoVariante], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    txt = run_dir / f"reporte_tipo_{tipo.id:02d}_{tipo.name}.txt"
    counts = _veredicto_counts(results)
    n = len(results)
    lines: List[str] = []
    lines.append(f"=== Tipo {tipo.id:02d} — {tipo.label} ===")
    lines.append(f"Scope: {tipo.scope} | N variantes: {n}")
    lines.append(
        "Veredictos: "
        + ", ".join(f"{k}={v}" for k, v in counts.most_common())
    )
    lines.append("")
    for i, r in enumerate(results, 1):
        ver = r.veredicto.veredicto if r.veredicto else "n/a"
        score = f"{r.veredicto.score:.2f}" if r.veredicto else "-"
        lines.append(f"[{i:02d}] ({ver}, score={score}) {r.variante.question}")
        if r.trace.error:
            lines.append(f"     ERROR: {r.trace.error}")
        elif r.veredicto and r.veredicto.brecha:
            lines.append(f"     BRECHA: {r.veredicto.brecha}")
        lines.append(
            f"     route={r.trace.route_decision} intent={r.trace.intent_label} "
            f"calc={r.trace.calc_mode} key={r.trace.metadata_key}"
        )
        snippet = (r.trace.response or "").strip().replace("\n", " ")
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        lines.append(f"     resp: {snippet}")
        lines.append("")
    txt.write_text("\n".join(lines), encoding="utf-8")

    csv_path = run_dir / f"reporte_tipo_{tipo.id:02d}_{tipo.name}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "idx", "question", "veredicto", "score", "brecha",
            "route", "intent", "calc_mode", "metadata_key", "error",
        ])
        for i, r in enumerate(results, 1):
            w.writerow([
                i,
                r.variante.question,
                r.veredicto.veredicto if r.veredicto else "",
                f"{r.veredicto.score:.3f}" if r.veredicto else "",
                r.veredicto.brecha if r.veredicto else "",
                r.trace.route_decision or "",
                r.trace.intent_label or "",
                r.trace.calc_mode or "",
                r.trace.metadata_key or "",
                r.trace.error or "",
            ])
    return txt


def write_master_report(
    per_tipo: Dict[int, List[ResultadoVariante]],
    tipos: List[TipoConsulta],
    run_dir: Path,
) -> Path:
    path = run_dir / "master_report.txt"
    lines: List[str] = []
    lines.append("=== QA Automator — Master Report ===")
    total = 0
    grand: Counter = Counter()
    for tipo in tipos:
        results = per_tipo.get(tipo.id, [])
        n = len(results)
        total += n
        c = _veredicto_counts(results)
        grand.update(c)
        head = f"Tipo {tipo.id:02d} {tipo.name:30s} N={n:3d}  "
        head += "  ".join(f"{k}={v}" for k, v in c.most_common())
        lines.append(head)
    lines.append("")
    lines.append(f"Totales: N={total}  " + "  ".join(f"{k}={v}" for k, v in grand.most_common()))
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
