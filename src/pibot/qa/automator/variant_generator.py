"""Carga tipos de consulta desde YAML y genera variantes vía LLM + catálogo."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from ._model import get_openai_api_key, get_openai_model
from .catalog_expander import get_catalog_facts
from .schemas import RubricItem, TipoConsulta, Variante

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "qa" / "input"
DEFAULT_TYPES_FILE = INPUT_DIR / "tipos_consulta.yaml"


def _try_yaml_load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        return _minimal_yaml(text)


def _minimal_yaml(text: str) -> Any:
    """Parser tolerante para el formato simple de tipos_consulta.yaml.

    Si yaml no está instalado, intenta leer JSON como fallback.
    """
    try:
        return json.loads(text)
    except Exception as exc:
        raise RuntimeError(
            "No se pudo parsear tipos_consulta.yaml. Instala pyyaml "
            "(pip install pyyaml) o entrega el archivo como JSON."
        ) from exc


def load_tipos(path: Path = DEFAULT_TYPES_FILE) -> List[TipoConsulta]:
    raw = _try_yaml_load(path)
    items = raw.get("tipos") if isinstance(raw, dict) else raw
    out: List[TipoConsulta] = []
    for it in items or []:
        rubric = [
            RubricItem(id=r["id"], weight=int(r.get("weight", 1)), check=r["check"])
            for r in (it.get("rubric") or [])
        ]
        out.append(
            TipoConsulta(
                id=int(it["id"]),
                name=str(it["name"]),
                label=str(it.get("label") or it["name"]),
                scope=str(it.get("scope", "in")),
                requirements=list(it.get("requirements") or []),
                examples=list(it.get("examples") or []),
                catalog_seed=dict(it.get("catalog_seed") or {}),
                rubric=rubric,
            )
        )
    return out


def _seed_from_catalog(tipo: TipoConsulta) -> List[str]:
    facts = get_catalog_facts()
    seed_cfg = tipo.catalog_seed or {}
    out: List[str] = []
    use_indicators = seed_cfg.get("indicators")
    use_activities = seed_cfg.get("activities")
    use_regions = seed_cfg.get("regions")
    template = seed_cfg.get("template")
    if not template:
        return out
    indicators = facts["indicators"] if use_indicators else [None]
    activities = facts["activities"] if use_activities else [None]
    regions = facts["regions"] if use_regions else [None]
    for ind in indicators:
        for act in activities:
            for reg in regions:
                try:
                    q = template.format(indicator=ind or "", activity=act or "", region=reg or "")
                    q = re.sub(r"\s+", " ", q).strip(" ?¿.,;:")
                    if q:
                        out.append(q + "?")
                except Exception:
                    continue
    return out


def _llm_generate(tipo: TipoConsulta, n: int) -> List[str]:
    api_key = get_openai_api_key()
    if not api_key:
        return []
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return []
    client = OpenAI(api_key=api_key)
    sys_prompt = (
        "Eres un generador de preguntas en español para un chatbot del "
        "Banco Central de Chile sobre PIB, IMACEC, regionales e inversión. "
        "Devuelves SOLO un JSON con la forma {\"variantes\": [\"...\", ...]}."
    )
    user_prompt = (
        f"Tipo de consulta: {tipo.name}\n"
        f"Descripción/label: {tipo.label}\n"
        f"Requisitos formales: {tipo.requirements}\n"
        f"Ejemplos canónicos:\n- " + "\n- ".join(tipo.examples) + "\n\n"
        f"Genera {n} variantes nuevas (no repitas literalmente los ejemplos), "
        "diversas en período, indicador, actividad o región según aplique. "
        "Devuelve JSON puro."
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
        return [str(q).strip() for q in (data.get("variantes") or []) if str(q).strip()]
    except Exception:
        return []


def generate_variants(tipo: TipoConsulta, n_total: int = 50) -> List[Variante]:
    """Garantiza n_total preguntas: examples → catálogo → LLM."""
    out: List[Variante] = []
    seen: set[str] = set()

    def _push(q: str, source: str):
        norm = re.sub(r"\s+", " ", q).strip().lower()
        if not norm or norm in seen:
            return
        seen.add(norm)
        out.append(
            Variante(
                type_id=tipo.id,
                type_name=tipo.name,
                index=len(out) + 1,
                question=q.strip(),
                source=source,
            )
        )

    for ex in tipo.examples:
        if len(out) >= n_total:
            break
        _push(ex, "example")

    for q in _seed_from_catalog(tipo):
        if len(out) >= n_total:
            break
        _push(q, "catalog")

    if len(out) < n_total:
        missing = n_total - len(out)
        for q in _llm_generate(tipo, missing * 2):
            if len(out) >= n_total:
                break
            _push(q, "llm")

    return out
