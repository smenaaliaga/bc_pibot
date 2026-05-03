"""Juez OpenAI: aplica la rúbrica de cada tipo a la respuesta del grafo."""
from __future__ import annotations
import json
from typing import List

from ._model import get_openai_api_key, get_openai_model
from .schemas import CriterioEval, RubricItem, TraceResult, Veredicto


def _fallback_veredicto(reason: str) -> Veredicto:
    return Veredicto(
        veredicto="warn",
        score=0.0,
        criterios=[],
        brecha=reason,
        raw_judge="",
    )


def judge(question: str, trace: TraceResult, rubric: List[RubricItem], scope: str) -> Veredicto:
    api_key = get_openai_api_key()
    if not api_key:
        return _fallback_veredicto("OPENAI_API_KEY no disponible — juez desactivado.")
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return _fallback_veredicto("paquete openai no instalado")
    client = OpenAI(api_key=api_key)
    rubric_text = "\n".join(f"- {r.id} (peso {r.weight}): {r.check}" for r in rubric)
    sys_prompt = (
        "Eres un evaluador estricto de respuestas de un chatbot del Banco "
        "Central de Chile. Devuelves SOLO JSON con esta forma exacta:\n"
        "{\n"
        '  "veredicto": "ok" | "warn" | "fail",\n'
        '  "score": 0..1,\n'
        '  "criterios": [{"id": "...", "ok": true|false, "evidence": "..."}],\n'
        '  "brecha": "explicación corta si no es ok"\n'
        "}\n"
        "Sé conservador: si falta cualquier criterio crítico, marca fail."
    )
    user_prompt = (
        f"Scope esperado: {scope} (in = dentro de alcance, out = fuera de alcance)\n"
        f"Pregunta: {question}\n"
        f"Respuesta del bot:\n---\n{trace.response}\n---\n"
        f"Trazas internas: route_decision={trace.route_decision}, "
        f"intent={trace.intent_label}, calc_mode={trace.calc_mode}, "
        f"metadata_key={trace.metadata_key}\n\n"
        f"Rúbrica:\n{rubric_text}\n\n"
        "Evalúa cada criterio y devuelve el JSON."
    )
    try:
        resp = client.chat.completions.create(
            model=get_openai_model(),
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        criterios = [
            CriterioEval(id=c.get("id", ""), ok=bool(c.get("ok")), evidence=str(c.get("evidence", "")))
            for c in (data.get("criterios") or [])
        ]
        return Veredicto(
            veredicto=str(data.get("veredicto", "warn")),
            score=float(data.get("score", 0.0)),
            criterios=criterios,
            brecha=str(data.get("brecha", "")),
            raw_judge=content,
        )
    except Exception as exc:
        return _fallback_veredicto(f"juez falló: {type(exc).__name__}: {exc}")
