"""Dataclasses del automator."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RubricItem:
    id: str
    weight: int
    check: str


@dataclass
class TipoConsulta:
    id: int
    name: str
    label: str
    scope: str
    requirements: List[str]
    examples: List[str]
    catalog_seed: Dict[str, Any]
    rubric: List[RubricItem]


@dataclass
class Variante:
    type_id: int
    type_name: str
    index: int
    question: str
    source: str  # "example" | "catalog" | "llm" | "file"


@dataclass
class CriterioEval:
    id: str
    ok: bool
    evidence: str


@dataclass
class Veredicto:
    veredicto: str  # ok | warn | fail
    score: float
    criterios: List[CriterioEval]
    brecha: str = ""
    raw_judge: str = ""


@dataclass
class TraceResult:
    question: str
    response: str
    route_decision: Optional[str] = None
    metadata_key: Optional[str] = None
    calc_mode: Optional[str] = None
    intent_label: Optional[str] = None
    macro_label: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ResultadoVariante:
    variante: Variante
    trace: TraceResult
    veredicto: Optional[Veredicto] = None
