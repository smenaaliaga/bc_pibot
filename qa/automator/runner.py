"""Orquestador: para cada tipo, ejecuta variantes en el grafo y aplica el juez."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from .grader import judge
from .graph_runner import run_question
from .schemas import ResultadoVariante, TipoConsulta, TraceResult, Variante


def run_variant(variante: Variante, tipo: TipoConsulta, use_judge: bool = True) -> ResultadoVariante:
    raw = run_question(variante.question)
    trace = TraceResult(
        question=variante.question,
        response=raw.get("response", ""),
        route_decision=raw.get("route_decision"),
        metadata_key=raw.get("metadata_key"),
        calc_mode=raw.get("calc_mode"),
        intent_label=raw.get("intent_label"),
        macro_label=raw.get("macro_label"),
        error=raw.get("error"),
    )
    veredicto = None
    if use_judge:
        veredicto = judge(variante.question, trace, tipo.rubric, tipo.scope)
    return ResultadoVariante(variante=variante, trace=trace, veredicto=veredicto)


def run_tipo(
    tipo: TipoConsulta,
    variantes: List[Variante],
    use_judge: bool = True,
    progress: Optional[callable] = None,
) -> List[ResultadoVariante]:
    out: List[ResultadoVariante] = []
    for i, v in enumerate(variantes, 1):
        res = run_variant(v, tipo, use_judge=use_judge)
        out.append(res)
        if progress:
            progress(tipo, i, len(variantes), res)
    return out
