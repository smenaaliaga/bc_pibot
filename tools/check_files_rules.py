#!/usr/bin/env python3
"""Validador de archivos clave para el post-procesamiento del clasificador.

Verifica que los siguientes módulos existan y cumplan el contrato esperado
antes de promover el código a otra rama / repositorio:

  1. rules.post_classifier        → API consolidada (predicates + reglas).
  2. orchestrator.data._business_rules
                                  → shim re-export de rules.post_classifier.
  3. orchestrator.data.response   → SYSTEM_PROMPT + builders críticos
                                    (incluye apertura literal de
                                    CONTRIBUCIÓN GRUPAL).

Exit codes:
    0 → todo OK
    1 → falta uno o más símbolos / anclas
    2 → error inesperado al importar
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List

# Permitir ejecución desde cualquier cwd: agregamos la raíz del repo
# (parent de tools/) a sys.path si no está.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Contratos
# ---------------------------------------------------------------------------

POST_CLASSIFIER_SYMBOLS = [
    "ResolvedEntities",
    "apply_business_rules",
    "_is_calendar_intent",
    "_is_value_desestacionalizado",
    "_is_greeting",
    "_is_out_of_scope",
    "_is_empty_cls",
    "_CALENDAR_INTENT_RE",
    "_VALUE_DESEST_RE",
    "_GREETING_RE",
    "_OUT_OF_SCOPE_ALLOWLIST_RE",
    "_SHARE_INTENT_RE",
    "_YOY_KEYWORDS_RE",
    "_VAR_MENSUAL_RE",
    "_VAR_TRIMESTRAL_RE",
    "_rule_detect_share_intent",
    "_rule_natural_freq_variation_is_prev_period",
    "_rule_seasonality_sa_implies_prev_period",
    "_rule_contribution_investment_force_general",
    "_rule_contribution_demanda_interna",
    "_rule_imacec_force_monthly",
    "_rule_imacec_default_activity",
    "_rule_assign_price",
    "_rule_pib_hist_flag",
    "_rule_redirect_pib_monthly_to_quarterly",
]

BUSINESS_RULES_SHIM_SYMBOLS = [
    "ResolvedEntities",
    "apply_business_rules",
]

RESPONSE_SYMBOLS = [
    "SYSTEM_PROMPT",
    "stream_data_response",
    "handle_tool_call",
    "format_period_labels",
]

SYSTEM_PROMPT_ANCHORS = [
    "ORDEN DEL PRIMER ENUNCIADO",
    "ESTRUCTURA OBLIGATORIA DE RESPUESTA",
    "REGLA DE CONTRIBUCIONES",
    "FORMATO NUMÉRICO ESPAÑOL",
    "FORMATO NEGRITA OBLIGATORIO",
    "PRIORIZACIÓN DE MÉTRICAS",
]

POLARITY_BUILDER_NAME = "_build_contribution_ranking_polarity_instruction"
POLARITY_ANCHORS = [
    "APERTURA LITERAL OBLIGATORIA",
    "REGLA DE PUREZA DE SIGNO POR GRUPO",
    "REGLA DE TOP-",
    "polaridad cruzada",
]


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

@dataclass
class ModuleResult:
    name: str
    ok: bool = True
    missing_symbols: List[str] = field(default_factory=list)
    missing_anchors: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "missing_symbols": self.missing_symbols,
            "missing_anchors": self.missing_anchors,
            "error": self.error,
        }


def _check_symbols(mod, symbols: List[str]) -> List[str]:
    return [s for s in symbols if not hasattr(mod, s)]


def _check_regex_compilable(mod, attr: str) -> bool:
    obj = getattr(mod, attr, None)
    return isinstance(obj, re.Pattern)


def _check_post_classifier() -> ModuleResult:
    res = ModuleResult(name="rules.post_classifier")
    try:
        mod = importlib.import_module("rules.post_classifier")
    except Exception as exc:  # pragma: no cover
        res.ok = False
        res.error = f"import error: {exc}"
        return res

    res.missing_symbols = _check_symbols(mod, POST_CLASSIFIER_SYMBOLS)

    # Regex compilados
    for rx in (
        "_CALENDAR_INTENT_RE",
        "_VALUE_DESEST_RE",
        "_GREETING_RE",
        "_OUT_OF_SCOPE_ALLOWLIST_RE",
        "_SHARE_INTENT_RE",
        "_YOY_KEYWORDS_RE",
        "_VAR_MENSUAL_RE",
        "_VAR_TRIMESTRAL_RE",
    ):
        if hasattr(mod, rx) and not _check_regex_compilable(mod, rx):
            res.missing_anchors.append(f"{rx}: no es re.Pattern compilado")

    res.ok = not res.missing_symbols and not res.missing_anchors
    return res


def _check_business_rules_shim() -> ModuleResult:
    res = ModuleResult(name="orchestrator.data._business_rules")
    try:
        mod = importlib.import_module("orchestrator.data._business_rules")
    except Exception as exc:
        res.ok = False
        res.error = f"import error: {exc}"
        return res

    res.missing_symbols = _check_symbols(mod, BUSINESS_RULES_SHIM_SYMBOLS)

    # Verificar que sea efectivamente shim del nuevo módulo.
    try:
        from rules.post_classifier import ResolvedEntities as _RE_canon
        if getattr(mod, "ResolvedEntities", None) is not _RE_canon:
            res.missing_anchors.append(
                "ResolvedEntities en shim no apunta a rules.post_classifier.ResolvedEntities"
            )
    except Exception as exc:
        res.missing_anchors.append(f"comparación shim falló: {exc}")

    res.ok = not res.missing_symbols and not res.missing_anchors
    return res


def _check_response() -> ModuleResult:
    res = ModuleResult(name="orchestrator.data.response")
    try:
        mod = importlib.import_module("orchestrator.data.response")
    except Exception as exc:
        res.ok = False
        res.error = f"import error: {exc}"
        return res

    res.missing_symbols = _check_symbols(mod, RESPONSE_SYMBOLS)

    sp = getattr(mod, "SYSTEM_PROMPT", "") or ""
    for anchor in SYSTEM_PROMPT_ANCHORS:
        if anchor not in sp:
            res.missing_anchors.append(f"SYSTEM_PROMPT no contiene: {anchor!r}")

    polarity = getattr(mod, POLARITY_BUILDER_NAME, None)
    if polarity is None:
        res.missing_symbols.append(POLARITY_BUILDER_NAME)
    else:
        try:
            src = inspect.getsource(polarity)
        except Exception as exc:  # pragma: no cover
            res.missing_anchors.append(f"no se pudo leer source de {POLARITY_BUILDER_NAME}: {exc}")
            src = ""
        for anchor in POLARITY_ANCHORS:
            if anchor not in src:
                res.missing_anchors.append(
                    f"{POLARITY_BUILDER_NAME} no contiene: {anchor!r}"
                )

    res.ok = not res.missing_symbols and not res.missing_anchors
    return res


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def _print_human(results: List[ModuleResult]) -> None:
    ok_count = sum(1 for r in results if r.ok)
    print(f"\n=== Validación de archivos post-clasificador ===")
    print(f"Módulos OK: {ok_count}/{len(results)}\n")
    for r in results:
        status = "OK " if r.ok else "FAIL"
        print(f"[{status}] {r.name}")
        if r.error:
            print(f"       error: {r.error}")
        for s in r.missing_symbols:
            print(f"       · símbolo faltante: {s}")
        for a in r.missing_anchors:
            print(f"       · ancla faltante:   {a}")
    print()


def _print_json(results: List[ModuleResult]) -> None:
    payload = {
        "ok": all(r.ok for r in results),
        "modules": [r.to_dict() for r in results],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Salida JSON")
    args = parser.parse_args()

    try:
        results = [
            _check_post_classifier(),
            _check_business_rules_shim(),
            _check_response(),
        ]
    except Exception as exc:  # pragma: no cover
        print(f"ERROR INESPERADO: {exc}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(results)
    else:
        _print_human(results)

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
