"""
NER Normalizer — Normalización de entidades para indicadores económicos chilenos.

Punto de entrada principal del pipeline de normalización que transforma las
entidades crudas detectadas por el modelo NER en valores canónicos listos
para consultar el catálogo de series (IMACEC / PIB).

Arquitectura del módulo (tras refactorización):
    _vocab.py   → Diccionarios de vocabulario (única fuente de verdad)
    _text.py    → Normalización de texto y matching fuzzy
    _period.py  → Parseo y resolución de períodos temporales
    normalizer.py (este archivo) → Orquestación y reglas de negocio

API pública (consumida por classifier_agent y tests):
    normalize_entities()                          → normalización completa para /predict
    normalize_region()                            → normalización aislada de región
    normalize_ner_entities()                      → normalización interna legacy
    normalize_from_json()                         → wrapper CLI con entrada/salida JSON
    coerce_req_form_from_period_and_frequency()   → ajuste de req_form si period cubre varios períodos

Reglas de negocio (orden de ejecución):
    1. Fuzzy matching  — Cada entidad se normaliza contra su vocabulario con
       tolerancia a faltas de ortografía. Se priorizan variantes negativas
       cuando el input contiene "no" (ej: "no minero" → "no_mineria").

    2. Separación compuesta — En activity/region/investment, expresiones con "y"
       se dividen en subvalores salvo que la frase completa ya tenga buen match.

    3. Inferencia indicator/frequency:
       a) Si hay frequency explícita → indicator se deriva directamente.
       b) Si hay region/investment → indicator=pib.
       c) Si hay period → se infiere frequency desde granularidad del período.
       d) Fallback: imacec/m.

    4. Inferencia seasonality:
       - prev_period → sa; yoy/original/contribution/vacío → nsa.
       - PIB regional → siempre nsa.

    5. Resolución de period:
       - Siempre rango [inicio, fin] en formato YYYY-MM-DD.
       - Soporta: referencias relativas, trimestres, meses, décadas, latest.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# ─── Submódulos internos ───────────────────────────────────────────────────────
from orchestrator.normalizer._vocab import (
    ACTIVITY_TERMS_IMACEC,
    ACTIVITY_TERMS_PIB,
    ACTIVITY_TERMS_PIB_REGIONAL,
    FREQUENCY_TERMS,
    INVESTMENT_TERMS,
    MONTHS,
    REGION_TERMS,
    SEASONALITY_TERMS,
)
from orchestrator.normalizer._text import (
    best_vocab_key,
    fuzzy_match,
    is_generic_indicator,
    normalize_text,
)
from orchestrator.normalizer._period import (
    contains_latest_ref,
    fmt_month_start,
    has_quarter_ref,
    infer_frequency_from_period,
    is_year_only_ref,
    normalize_single_period,
    parse_yyyymmdd,
    reference_now,
    resolve_period,
)

# Aliases internos para acceso legado (compatibilidad con imports directos)
_normalize_text = normalize_text
_fuzzy_match = fuzzy_match
_best_vocab_key = best_vocab_key
_is_generic_indicator_value = is_generic_indicator
_contains_latest_reference = contains_latest_ref
_has_quarter_reference = has_quarter_ref
_is_year_only_period_reference = is_year_only_ref
_infer_frequency_from_period_for_point = infer_frequency_from_period
_reference_now = reference_now
_parse_yyyymmdd = parse_yyyymmdd
_format_month_start = fmt_month_start

# Constantes re-exportadas para compatibilidad
from orchestrator.normalizer._vocab import (  # noqa: E402, F811
    ACTIVITY_TERMS_IMACEC as ACTIVITY_TERMS_IMACEC,
    ACTIVITY_TERMS_PIB as ACTIVITY_TERMS_PIB,
    ACTIVITY_TERMS_PIB_REGIONAL as ACTIVITY_TERMS_PIB_REGIONAL,
    FREQUENCY_TERMS as FREQUENCY_TERMS,
    INDICATOR_TERMS as INDICATOR_TERMS,
    INVESTMENT_TERMS as INVESTMENT_TERMS,
    MONTHS as MONTHS,
    MONTH_ALIASES as MONTH_ALIASES,
    PERIOD_LATEST_TERMS as PERIOD_LATEST_TERMS,
    QUARTERS_START_MONTH as QUARTERS_START_MONTH,
    REGION_TERMS as REGION_TERMS,
    SEASONALITY_TERMS as SEASONALITY_TERMS,
    SPANISH_NUMBER_WORDS as SPANISH_NUMBER_WORDS,
    DECADE_WORDS as DECADE_WORDS,
    ROMAN_QUARTERS as ROMAN_QUARTERS,
    REFERENCE_TIMEZONE as REFERENCE_TIMEZONE,
)
from orchestrator.normalizer._period import (  # noqa: E402, F811
    PERIOD_LATEST_REGEX as PERIOD_LATEST_REGEX_PATTERNS,
    PERIOD_PREVIOUS_REGEX as PERIOD_PREVIOUS_REGEX_PATTERNS,
)
# Re-export las funciones de _period que se importaban vía PERIOD_*_REGEX
from orchestrator.normalizer._vocab import PERIOD_LATEST_REGEX, PERIOD_PREVIOUS_REGEX  # noqa: F811

_ENTITY_KEYS = ("indicator", "seasonality", "frequency", "activity", "region", "investment", "period")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers de contexto
# ═══════════════════════════════════════════════════════════════════════════════

def _activity_vocab(indicator: Optional[str], regional: bool = False) -> Dict[str, List[str]]:
    """Selecciona el vocabulario de actividades según indicador y contexto regional."""
    if indicator == "pib":
        return ACTIVITY_TERMS_PIB_REGIONAL if regional else ACTIVITY_TERMS_PIB
    return ACTIVITY_TERMS_IMACEC


def _is_regional_pib(
    indicator: Optional[str],
    region_value: Optional[str] = None,
    region_intent_label: Optional[str] = None,
) -> bool:
    """``True`` si la consulta es PIB con contexto regional."""
    if indicator != "pib":
        return False
    return bool(region_value) or region_intent_label in {"specific", "general"}


def _activity_match_count(raw_values: List[str], vocab: Dict[str, List[str]]) -> int:
    return sum(
        1 for r in raw_values if r and best_vocab_key(
            r, vocab, threshold=0.75, prefer_negative_if_no=True, negative_threshold=0.72,
        )
    )


def _split_conjoined(
    entity_key: str,
    raw_values: List[str],
    indicator: Optional[str],
) -> List[str]:
    """Divide valores con conjunción "y" salvo que la frase completa ya matchee."""
    if entity_key not in {"activity", "region", "investment"}:
        return raw_values

    vocab = {"activity": _activity_vocab(indicator), "region": REGION_TERMS, "investment": INVESTMENT_TERMS}.get(entity_key)
    if not vocab:
        return raw_values

    expanded: List[str] = []
    for raw in raw_values:
        if not raw:
            continue
        if best_vocab_key(raw, vocab=vocab, threshold=0.9):
            if raw not in expanded:
                expanded.append(raw)
            continue
        parts = [p.strip() for p in re.split(r"\s+y\s+", raw, flags=re.IGNORECASE) if p.strip()]
        for p in (parts if len(parts) > 1 else [raw]):
            if p not in expanded:
                expanded.append(p)
    return expanded


def _as_list(value: Optional[Union[str, List[str]]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v]
    return [value] if value else []


def _intent_label(intent_value: Any) -> Optional[str]:
    if intent_value is None:
        return None
    if isinstance(intent_value, dict):
        lbl = intent_value.get("label")
        return str(lbl).lower() if lbl is not None else None
    return str(intent_value).lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Normalización individual por entidad
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_indicator(value: Optional[str], frequency: Optional[str]) -> Optional[str]:
    """Normaliza ``indicator`` o lo infiere desde ``frequency``."""
    if is_generic_indicator(value):
        return {"m": "imacec", "q": "pib", "a": "pib"}.get(frequency, "imacec")

    n = normalize_text(value)
    if re.search(r"\bimacec\b", n):
        return "imacec"
    if re.search(r"\bpib\b", n) or "producto interno bruto" in n:
        return "pib"

    match = best_vocab_key(
        n,
        {"imacec": ["imacec"], "pib": ["pib", "producto interno bruto", "producto bruto", "interno bruto"]},
        threshold=0.6,
    )
    if match:
        return match

    return {"m": "imacec", "q": "pib", "a": "pib"}.get(frequency)


def normalize_seasonality(value: Optional[str], calc_mode: Optional[str]) -> Optional[str]:
    """Normaliza ``seasonality`` o la infiere desde ``calc_mode``."""
    if value:
        match = best_vocab_key(
            value,
            {"sa": SEASONALITY_TERMS.get("sa", []), "nsa": SEASONALITY_TERMS.get("nsa", [])},
            threshold=0.7,
            prefer_negative_if_no=True,
            negative_threshold=0.68,
        )
        if match:
            return match
    # Inferencia: prev_period→sa, resto→nsa
    return "sa" if calc_mode == "prev_period" else "nsa"


def normalize_frequency(value: Optional[str]) -> Optional[str]:
    """Normaliza ``frequency`` a código ``m``/``q``/``a``."""
    if not value:
        return None
    return best_vocab_key(value, FREQUENCY_TERMS, threshold=0.75)


def normalize_activity(
    value: Optional[str],
    indicator: Optional[str],
    calc_mode: Optional[str] = None,
    region_value: Optional[str] = None,
    regional_pib_context: Optional[bool] = None,
) -> Tuple[Optional[str], List[str]]:
    """Normaliza ``activity`` según indicador y contexto regional."""
    if not value:
        return None, []

    n = normalize_text(value)

    # PIB + contribución: "no minero" no existe en ese contexto.
    if indicator == "pib" and calc_mode == "contribution" and "no" in set(n.split()):
        if re.search(r"\bminer(?:o|a|ia)\b", n):
            return None, [value]

    is_regional = (
        _is_regional_pib(indicator=indicator, region_value=region_value)
        if regional_pib_context is None
        else bool(regional_pib_context)
    )
    vocab = _activity_vocab(indicator, regional=is_regional)

    match = best_vocab_key(n, vocab, threshold=0.75, prefer_negative_if_no=True, negative_threshold=0.72)
    return (match, []) if match else (None, [value])


def normalize_region(value: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """Normaliza ``region`` a clave canónica de región de Chile."""
    if not value:
        return None, []
    match = best_vocab_key(value, REGION_TERMS, threshold=0.75)
    return (match, []) if match else (None, [value])


def normalize_investment(value: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """Normaliza ``investment`` a componente de gasto conocido."""
    if not value:
        return None, []
    match = best_vocab_key(value, INVESTMENT_TERMS, threshold=0.75)
    return (match, []) if match else (None, [value])


def normalize_period(value: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """Parsea un solo valor de período a ``YYYY-MM-DD`` (fallback simple)."""
    return normalize_single_period(value)


# ═══════════════════════════════════════════════════════════════════════════════
# Normalización interna legacy (normalize_ner_entities)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_ner_entities(
    ner_output: Dict[str, Any],
    calc_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Normaliza entidades NER con inferencia de valores faltantes.

    Usado internamente como paso base antes de aplicar reglas de negocio
    en ``normalize_entities``.
    """
    entities = ner_output.get("interpretation", {}).get("entities", {})

    def _first(key: str) -> Optional[str]:
        vals = entities.get(key)
        return vals[0] if vals else None

    indicator_raw = _first("indicator")
    freq_raw = _first("frequency")
    seasonality_raw = _first("seasonality")
    region_raw = _first("region")
    investment_raw = _first("investment")
    activity_raw = _first("activity")
    period_raw = _first("period")

    norm_freq = normalize_frequency(freq_raw)
    norm_ind = normalize_indicator(indicator_raw, norm_freq)
    norm_seas = normalize_seasonality(seasonality_raw, calc_mode)
    norm_region, fail_region = normalize_region(region_raw)
    norm_inv, fail_inv = normalize_investment(investment_raw)

    # Indicador faltante + contexto regional/inversión → PIB.
    if not indicator_raw and (norm_region is not None or norm_inv is not None):
        norm_ind = "pib"
        if not norm_freq:
            norm_freq = "q"

    # PIB regional fuerza nsa.
    if _is_regional_pib(indicator=norm_ind, region_value=norm_region):
        norm_seas = "nsa"

    norm_act, fail_act = normalize_activity(
        activity_raw, norm_ind, calc_mode=calc_mode, region_value=norm_region,
    )
    norm_period, fail_period = normalize_period(period_raw)

    # Construir failed_matches.
    failed: Dict[str, List[str]] = {}
    if fail_act:
        failed["ACTIVITY"] = fail_act
    if fail_region:
        failed["REGION"] = fail_region
    if fail_inv:
        failed["INVESTMENT"] = fail_inv
    if fail_period:
        failed["PERIOD"] = fail_period

    # Registrar reglas de inferencia aplicadas.
    rules: List[str] = []
    if not indicator_raw:
        if norm_freq == "m":
            rules.append("FREQUENCY='m' + empty INDICATOR → INDICATOR='imacec'")
        elif norm_freq in ("q", "a"):
            rules.append(f"FREQUENCY='{norm_freq}' + empty INDICATOR → INDICATOR='pib'")
        else:
            rules.append("empty INDICATOR + empty FREQUENCY → INDICATOR='imacec' (default)")
    if not seasonality_raw:
        if calc_mode == "prev_period":
            rules.append("calc_mode='prev_period' + empty SEASONALITY → SEASONALITY='sa'")
        else:
            rules.append("calc_mode + empty SEASONALITY → SEASONALITY='nsa'")

    return {
        "normalized_entities": {
            "indicator": norm_ind,
            "seasonality": norm_seas,
            "frequency": norm_freq,
            "activity": norm_act,
            "region": norm_region,
            "investment": norm_inv,
            "period": norm_period,
        },
        "failed_matches": failed or None,
        "inference_rules_applied": rules or None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API principal: normalize_entities (usada por /predict)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_multi(
    key: str,
    raw_values: List[str],
    calc_mode: Optional[str],
    base: Dict[str, Optional[str]],
) -> List[str]:
    """Normaliza N valores para una misma entidad, deduplica y preserva orden."""
    out: List[str] = []
    for raw in raw_values:
        if not raw:
            continue
        val: Optional[str] = None
        if key == "indicator":
            val = normalize_indicator(raw, base.get("frequency"))
        elif key == "seasonality":
            val = normalize_seasonality(raw, calc_mode)
        elif key == "frequency":
            val = normalize_frequency(raw)
        elif key == "activity":
            val = normalize_activity(raw, base.get("indicator"), calc_mode=calc_mode, region_value=base.get("region"))[0]
        elif key == "region":
            val = normalize_region(raw)[0]
        elif key == "investment":
            val = normalize_investment(raw)[0]
        if val and val not in out:
            out.append(val)
    return out


def normalize_entities(
    entities: Dict[str, List[str]],
    calc_mode: Optional[str] = None,
    req_form: Optional[str] = None,
    intents: Optional[Dict[str, Any]] = None,
) -> Dict[str, Union[List[str], None]]:
    """Normalizador principal consumido por ``/predict``.

    Retorna un diccionario donde cada entidad (excepto ``period``) es una lista
    y ``period`` es un rango ``[inicio, fin]`` o ``None``.
    """
    # ── Paso 1: normalización base ──────────────────────────────────────────
    base_result = normalize_ner_entities(
        {"interpretation": {"entities": entities}}, calc_mode=calc_mode,
    )
    base = base_result.get("normalized_entities", {})

    # ── Paso 2: normalizar cada entidad (soporta múltiples valores) ─────────
    response: Dict[str, Union[List[str], str, None]] = {}
    for key in _ENTITY_KEYS:
        if key == "period":
            continue
        raw = _split_conjoined(key, entities.get(key) or [], base.get("indicator"))
        response[key] = (
            _normalize_multi(key, raw, calc_mode, base)
            if len(raw) > 1
            else _as_list(base.get(key))
        )

    # ── Paso 3: reglas de negocio para indicator / frequency ────────────────
    raw_indicator = (entities.get("indicator") or [None])[0]
    raw_frequency = (entities.get("frequency") or [None])[0]
    has_explicit_freq = bool(str(raw_frequency or "").strip())
    ind_is_generic = is_generic_indicator(raw_indicator)
    req_norm = (req_form or "").strip().lower()

    # Contexto regional / inversión.
    region_lbl = _intent_label((intents or {}).get("region"))
    inv_lbl = _intent_label((intents or {}).get("investment"))
    has_region_ctx = (
        region_lbl not in {None, "none"}
        or bool(response.get("region"))
        or bool(entities.get("region"))
    )
    has_inv_ctx = (
        inv_lbl not in {None, "none"}
        or bool(response.get("investment"))
        or bool(entities.get("investment"))
    )
    has_r_or_i = has_region_ctx or has_inv_ctx

    # Cobertura de actividades.
    period_raw = entities.get("period") or []
    split_acts = _split_conjoined("activity", entities.get("activity") or [], None)
    imacec_cnt = _activity_match_count(split_acts, ACTIVITY_TERMS_IMACEC) if split_acts else 0
    pib_cnt = _activity_match_count(split_acts, ACTIVITY_TERMS_PIB) if split_acts else 0
    all_imacec = bool(split_acts) and imacec_cnt == len(split_acts)
    all_pib = bool(split_acts) and pib_cnt == len(split_acts)

    # Inferencia de frequency desde period (solo si no es explícita).
    is_imacec = "imacec" in response.get("indicator", [])
    skip_period_inf = req_norm == "range" and is_imacec and not ind_is_generic
    inferred_freq = (
        infer_frequency_from_period(period_raw)
        if req_norm in {"point", "range", "latest"} and not raw_frequency and not skip_period_inf
        else None
    )

    # Aplicar reglas de indicator + frequency.
    if ind_is_generic and has_r_or_i:
        response["indicator"] = ["pib"]
        if not raw_frequency:
            response["frequency"] = [inferred_freq or "q"]
    elif ind_is_generic and not raw_frequency:
        if inferred_freq == "m":
            response["indicator"], response["frequency"] = ["imacec"], ["m"]
        elif inferred_freq in {"q", "a"}:
            response["indicator"], response["frequency"] = ["pib"], [inferred_freq]
        elif all_imacec and not has_r_or_i:
            response["indicator"], response["frequency"] = ["imacec"], ["m"]
        elif has_r_or_i or all_pib:
            response["indicator"], response["frequency"] = ["pib"], ["q"]
        else:
            response["indicator"], response["frequency"] = ["imacec"], ["m"]
    elif not raw_frequency and "pib" in response.get("indicator", []):
        response["frequency"] = [inferred_freq or "q"]
    elif not raw_frequency and "imacec" in response.get("indicator", []):
        response["frequency"] = [inferred_freq or "m"]
    elif not raw_frequency and inferred_freq and not response.get("frequency"):
        response["frequency"] = [inferred_freq]

    # ── Paso 4: re-normalizar activity con el indicador final ───────────────
    final_ind = (response.get("indicator") or [None])[0]
    final_reg = (response.get("region") or [None])[0]
    final_regional = _is_regional_pib(final_ind, final_reg, region_lbl)

    if final_regional:
        response["seasonality"] = ["nsa"]

    if entities.get("activity"):
        act_raw = _split_conjoined("activity", entities["activity"], final_ind)
        normed_acts: List[str] = []
        for raw in act_raw:
            a, _ = normalize_activity(
                raw, final_ind, calc_mode=calc_mode,
                region_value=final_reg, regional_pib_context=final_regional,
            )
            if a and a not in normed_acts:
                normed_acts.append(a)
        response["activity"] = normed_acts

    # ── Paso 5: ajuste final de frequency ───────────────────────────────────
    has_year_only_point = any(is_year_only_ref(r) for r in period_raw if r)
    if "imacec" in response.get("indicator", []) and not has_explicit_freq:
        response["frequency"] = ["m"]
    elif (
        req_norm in {"point", "range", "latest"}
        and "pib" in response.get("indicator", [])
        and has_year_only_point
        and not has_explicit_freq
    ):
        response["frequency"] = ["a"]

    # ── Paso 6: resolver period ─────────────────────────────────────────────
    eff_freq = (response.get("frequency") or [None])[0]
    response["period"] = resolve_period(
        raw_values=period_raw,
        calc_mode=calc_mode,
        base_normalized=base,
        req_form=req_form,
        frequency=eff_freq,
    )

    return response


# ═══════════════════════════════════════════════════════════════════════════════
# Ajuste de req_form según período/frequency
# ═══════════════════════════════════════════════════════════════════════════════

def coerce_req_form_from_period_and_frequency(
    req_form: Optional[str],
    normalized_entities: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Si el period resuelto abarca más de un período para la frequency dada,
    convierte ``point`` / ``specific_point`` en ``range``."""
    rf = str(req_form or "").strip().lower()
    if rf not in {"point", "specific_point"}:
        return req_form
    if not isinstance(normalized_entities, dict):
        return req_form

    pv = normalized_entities.get("period")
    if not isinstance(pv, list) or len(pv) < 2:
        return req_form

    start = parse_yyyymmdd(str(pv[0] or "").strip())
    end = parse_yyyymmdd(str(pv[-1] or "").strip())
    if start is None or end is None:
        return req_form

    fv = normalized_entities.get("frequency")
    fc = str(fv[0] or "").strip().lower() if isinstance(fv, list) and fv else None

    if fc in {"a", "annual", "anual"}:
        same = start.year == end.year
    elif fc in {"q", "t", "quarterly", "trimestral"}:
        sq = ((start.month - 1) // 3) + 1
        eq = ((end.month - 1) // 3) + 1
        same = start.year == end.year and sq == eq
    else:
        same = start.year == end.year and start.month == end.month

    return req_form if same else "range"


# ═══════════════════════════════════════════════════════════════════════════════
# Wrapper JSON (CLI / testing)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_from_json(json_input: str, calc_mode: Optional[str] = None) -> str:
    """Procesa JSON del modelo NER y retorna JSON normalizado."""
    try:
        ner_output = json.loads(json_input)
        result = normalize_ner_entities(ner_output, calc_mode)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps(
            {"error": f"Invalid JSON input: {e}", "failed_matches": None, "inference_rules_applied": None},
            ensure_ascii=False,
            indent=2,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Ejemplos (ejecución directa)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _examples = [
        (
            "EJEMPLO 1: IMACEC más reciente",
            '{"text":"cual fue la ultima cifra del imacec","interpretation":{"entities":{"indicator":["imacec"],"period":["ultima"]}}}',
            None,
        ),
        (
            "EJEMPLO 2: Inferir INDICATOR desde FREQUENCY",
            '{"text":"dame los datos trimestrales de actividad","interpretation":{"entities":{"frequency":["trimestral"]}}}',
            "original",
        ),
        (
            "EJEMPLO 3: Múltiples entidades",
            '{"text":"actividad minera en metropolitana feb 2024 desestacionalizado","interpretation":{"entities":{"indicator":["imacec"],"activity":["mineria"],"region":["metropolitana"],"period":["febrero 2024"],"seasonality":["desestacionalizado"]}}}',
            "original",
        ),
    ]
    for title, payload, cm in _examples:
        print(title)
        print("Salida:", normalize_from_json(payload, cm))
        print("=" * 70)
