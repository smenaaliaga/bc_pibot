"""
Normalizer — Normalización de entidades y ajuste de clasificaciones.

Transforma las entidades crudas del NER en valores canónicos y aplica reglas
de negocio que también modifican las clasificaciones del modelo (calc_mode,
activity, region, investment, req_form). El resultado final es un
ResolvedEntities listo para buscar en data_store.

Arquitectura:
    _vocab.py       → Diccionarios de vocabulario (fuente de verdad)
    _text.py        → Normalización de texto y matching fuzzy
    _period.py      → Parseo y resolución de períodos temporales
    normalizer.py   → Orquestación, reglas de negocio y resolución final

API pública:
    normalize_entities()                          → normalización completa para /predict
    normalize_region()                            → normalización aislada de región
    normalize_ner_entities()                      → normalización interna legacy
    normalize_from_json()                         → wrapper CLI con entrada/salida JSON
    resolve_entities_for_data_query()             → resolución final para el nodo DATA
    coerce_req_form_from_period_and_frequency()   → ajuste de req_form según período
    coerce_specific_class_labels()                → preserva cls original (specific/general/none) sin degradar

Pipeline de normalización NER (normalize_entities):
    1. Fuzzy matching — Cada entidad se normaliza contra su vocabulario con
       tolerancia a errores. Prioriza variantes negativas si hay "no"
       (ej: "no minero" → "no_mineria").
    2. Separación compuesta — activity/region/investment con "y" se dividen
       en subvalores salvo que la frase completa ya tenga match.
    3. Inferencia indicator/frequency:
       a) frequency explícita → indicator derivado.
       b) region/investment presentes → indicator=pib.
       c) period → frequency desde granularidad.
       d) Fallback: imacec/m.
    4. Inferencia seasonality:
       - prev_period → sa; yoy/original/contribution/vacío → nsa.
       - PIB regional → siempre nsa.
    5. Resolución de period:
       - Rango [inicio, fin] en YYYY-MM-DD.
       - Soporta: relativas, trimestres, meses, décadas, latest.

Reglas de negocio sobre clasificaciones (apply_business_rules):
    Estas reglas mutan campos *_cls del ResolvedEntities después de la
    normalización NER. Se ejecutan en resolve_entities_for_data_query().

    1. contribution + investment_specific sin entidad ni región
       → activity_cls_resolved = "general"
       (Fuerza búsqueda por actividad general como desglose de contribución)

    2. IMACEC fuerza frecuencia mensual
       → frequency_ent = "m"
       (IMACEC solo tiene frecuencia mensual)

    3. IMACEC sin actividad explícita
       → activity_ent = "imacec"
       (Default del indicador como actividad raíz)

    4. Asignación de precio
       → price = price_ent o "enc" por defecto
       (Encadenado es el precio estándar)

    5. Flag histórico (hist)
       → hist = 1 si period <= 1996, 0 en caso contrario

    6. contribution + investment=demanda_interna
       → investment_cls = "general"
       (Demanda interna se trata como componente general de gasto)

    7. PIB con frecuencia mensual → redirige a trimestral
       → frequency_ent = "q", req_form_cls = "latest", period_ent = []
       (PIB mensual no existe; se redirige al último trimestre)

Ajustes adicionales a clasificaciones (fuera de apply_business_rules):
    - coerce_specific_class_labels(): preserva activity_cls/region_cls/
      investment_cls tal como vienen del clasificador, sin degradar a "none"
      aunque la entidad normalizada quede vacía.
    - coerce_req_form_from_period_and_frequency(): cambia req_form de
      "point" → "range" si el período resuelto abarca más de un período
      para la frecuencia dada.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# ─── Submódulos internos ───────────────────────────────────────────────────────
from orchestrator.normalizer._vocab import (
    ACTIVITY_TERMS_IMACEC,
    ACTIVITY_TERMS_PIB,
    ACTIVITY_TERMS_PIB_REGIONAL,
    FREQUENCY_TERMS,
    INVESTMENT_TERMS,
    PRICE_TERMS,
    REGION_TERMS,
    SEASONALITY_TERMS,
)
from orchestrator.normalizer._text import (
    best_vocab_key,
    is_generic_indicator,
    normalize_text,
)
from orchestrator.normalizer._period import (
    infer_frequency_from_period,
    is_year_only_ref,
    normalize_single_period,
    parse_yyyymmdd,
    reference_now,
    resolve_period,
)
from orchestrator.normalizer.routing_utils import (
    INTENT_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_NONE_KEYS,
)

# Alias legado usado en tests (monkeypatch)
_reference_now = reference_now

_ENTITY_KEYS = ("indicator", "seasonality", "frequency", "activity", "region", "investment", "price", "period")

logger = logging.getLogger(__name__)


def _first_non_empty_value(value: Any) -> Any:
    if isinstance(value, list):
        for item in value:
            if item not in (None, "", [], {}, ()):
                return item
        return None
    if value in (None, "", [], {}, ()):
        return None
    return value


def _all_non_empty_values(value: Any) -> List[str]:
    """Retorna una lista con todos los valores no vacíos."""
    if isinstance(value, list):
        return [v for v in value if v not in (None, "", [], {}, ())]
    if value in (None, "", [], {}, ()):
        return []
    return [value]


def _coerce_period_value(period_value: Any) -> List[Any]:
    if period_value in (None, "", [], {}, ()):
        return []
    if isinstance(period_value, list):
        return period_value
    return [period_value]


def _extract_year_value(value: Any) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", str(value or "").strip())
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


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


def _intent_label(intent_value: Any, *, key: Optional[str] = None) -> Optional[str]:
    if intent_value is None:
        return None
    if isinstance(intent_value, dict):
        if key in LOW_CONFIDENCE_NONE_KEYS:
            conf_raw = intent_value.get("confidence")
            if conf_raw is not None:
                try:
                    if float(conf_raw) < INTENT_CONFIDENCE_THRESHOLD:
                        return "none"
                except (TypeError, ValueError):
                    pass
        lbl = intent_value.get("label")
        return str(lbl).lower() if lbl is not None else None
    return str(intent_value).lower()


def coerce_class_label(value: Any, *, apply_threshold: bool = True) -> Optional[str]:
    """Extrae label de un payload de clasificador, aplicando threshold de confianza.

    Si ``value`` es un dict con ``label`` y ``confidence``, devuelve el label
    como string en lowercase.  Cuando ``apply_threshold=True`` y la confianza
    es menor a ``INTENT_CONFIDENCE_THRESHOLD``, devuelve ``"none"``.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        if apply_threshold:
            conf_raw = value.get("confidence")
            if conf_raw is not None:
                try:
                    if float(conf_raw) < INTENT_CONFIDENCE_THRESHOLD:
                        return "none"
                except (TypeError, ValueError):
                    pass
        lbl = value.get("label")
        return str(lbl).strip().lower() if lbl is not None else None
    text = str(value).strip().lower()
    return text or None


def coerce_specific_class_labels(
    *,
    activity_label: Optional[str],
    region_label: Optional[str],
    investment_label: Optional[str],
    normalized_entities: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Preserva la clasificación original de cls sin degradar a 'none'.

    Aunque la entidad normalizada quede vacía (sin match en vocabulario),
    se mantiene el cls original (specific/general/none) para que el
    downstream pueda manejar el caso correctamente.
    """
    def _coerce(label: Optional[str]) -> Optional[str]:
        return str(label or "").strip().lower() or None

    return (
        _coerce(activity_label),
        _coerce(region_label),
        _coerce(investment_label),
    )


@dataclass
class ResolvedEntities:
    """Entidades finales listas para consultar catálogo/data_store."""

    indicator_ent: Optional[str] = None
    seasonality_ent: Optional[str] = None
    frequency_ent: Optional[str] = None
    activity_ent: List[str] = field(default_factory=list)
    region_ent: List[str] = field(default_factory=list)
    investment_ent: List[str] = field(default_factory=list)
    price_ent: Optional[str] = None
    period_ent: List[Any] = field(default_factory=list)

    calc_mode_cls: Any = None
    activity_cls: Any = None
    activity_cls_resolved: Any = None
    region_cls: Any = None
    investment_cls: Any = None
    req_form_cls: Any = None

    price: Optional[str] = None
    hist: Optional[int] = None


def apply_business_rules(ent: ResolvedEntities) -> ResolvedEntities:
    """Aplica reglas de dominio y retorna el mismo objeto mutado."""
    _rule_contribution_investment_force_general(ent)
    _rule_imacec_force_monthly(ent)
    _rule_imacec_default_activity(ent)
    _rule_assign_price(ent)
    _rule_pib_hist_flag(ent)
    _rule_contribution_demanda_interna(ent)
    _rule_redirect_pib_monthly_to_quarterly(ent)
    return ent


def resolve_entities_for_data_query(
    *,
    normalized_entities: Optional[Dict[str, Any]],
    calc_mode_cls: Any = None,
    activity_cls: Any = None,
    region_cls: Any = None,
    investment_cls: Any = None,
    req_form_cls: Any = None,
) -> ResolvedEntities:
    """Construye y resuelve las entidades finales usadas por el nodo DATA."""
    normalized_entities = normalized_entities if isinstance(normalized_entities, dict) else {}

    ent = ResolvedEntities(
        indicator_ent=_first_non_empty_value(normalized_entities.get("indicator")),
        seasonality_ent=_first_non_empty_value(normalized_entities.get("seasonality")),
        frequency_ent=_first_non_empty_value(normalized_entities.get("frequency")),
        activity_ent=_all_non_empty_values(normalized_entities.get("activity")),
        region_ent=_all_non_empty_values(normalized_entities.get("region")),
        investment_ent=_all_non_empty_values(normalized_entities.get("investment")),
        price_ent=_first_non_empty_value(normalized_entities.get("price")),
        period_ent=_coerce_period_value(normalized_entities.get("period")),
        calc_mode_cls=calc_mode_cls,
        activity_cls=activity_cls,
        activity_cls_resolved=activity_cls,
        region_cls=region_cls,
        investment_cls=investment_cls,
        req_form_cls=req_form_cls,
    )
    return apply_business_rules(ent)


def _rule_contribution_investment_force_general(ent: ResolvedEntities) -> None:
    if (
        ent.calc_mode_cls == "contribution"
        and ent.investment_cls == "specific"
        and (not ent.investment_ent or ent.investment_ent == ["none"])
        and ent.region_cls in (None, "none")
    ):
        ent.activity_cls_resolved = "general"


def _rule_imacec_force_monthly(ent: ResolvedEntities) -> None:
    if (
        str(ent.indicator_ent or "").strip().lower() == "imacec"
        and str(ent.frequency_ent or "").strip().lower() != "m"
    ):
        ent.frequency_ent = "m"
        logger.info("[DATA_NODE] IMACEC: forzando frecuencia mensual (m)")


def _rule_imacec_default_activity(ent: ResolvedEntities) -> None:
    if ent.indicator_ent == "imacec" and not ent.activity_ent:
        ent.activity_ent = ["imacec"]


def _rule_assign_price(ent: ResolvedEntities) -> None:
    if ent.price_ent:
        ent.price = ent.price_ent
    else:
        ent.price = "enc"


def _rule_pib_hist_flag(ent: ResolvedEntities) -> None:
    ref_year = _extract_year_value(ent.period_ent[0]) if ent.period_ent else None
    ent.hist = 1 if (ref_year is not None and ref_year <= 1996) else 0


def _rule_contribution_demanda_interna(ent: ResolvedEntities) -> None:
    if (
        ent.calc_mode_cls == "contribution"
        and ent.investment_cls == "specific"
        and "demanda_interna" in ent.investment_ent
    ):
        ent.investment_cls = "general"


def _rule_redirect_pib_monthly_to_quarterly(ent: ResolvedEntities) -> None:
    if ent.indicator_ent != "pib":
        return
    if str(ent.frequency_ent or "").strip().lower() != "m":
        return

    if str(ent.calc_mode_cls or "").strip().lower() == "contribution":
        logger.warning("[DATA_NODE] PIB contribución mensual no existe; redirigiendo a trimestral")
    else:
        logger.warning("[DATA_NODE] PIB mensual no existe; redirigiendo a trimestral")

    ent.frequency_ent = "q"
    ent.req_form_cls = "latest"
    ent.period_ent = []


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


def normalize_price(value: Optional[str]) -> Optional[str]:
    """Normaliza ``price`` a código ``enc`` o ``co``."""
    if not value:
        return None
    return best_vocab_key(value, PRICE_TERMS, threshold=0.70)


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
    price_raw = _first("price")
    period_raw = _first("period")

    norm_freq = normalize_frequency(freq_raw)
    norm_ind = normalize_indicator(indicator_raw, norm_freq)
    norm_seas = normalize_seasonality(seasonality_raw, calc_mode)
    norm_region, fail_region = normalize_region(region_raw)
    norm_inv, fail_inv = normalize_investment(investment_raw)
    norm_price = normalize_price(price_raw)

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

    # Actividad per_capita → indicador pib_per_capita.
    if norm_act == "per_capita":
        norm_ind = "pib_per_capita"

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
            "price": norm_price,
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
    _NORMALIZERS = {
        "indicator": lambda raw, cm, b: normalize_indicator(raw, b.get("frequency")),
        "seasonality": lambda raw, cm, b: normalize_seasonality(raw, cm),
        "frequency": lambda raw, cm, b: normalize_frequency(raw),
        "activity": lambda raw, cm, b: normalize_activity(raw, b.get("indicator"), calc_mode=cm, region_value=b.get("region"))[0],
        "region": lambda raw, cm, b: normalize_region(raw)[0],
        "investment": lambda raw, cm, b: normalize_investment(raw)[0],
        "price": lambda raw, cm, b: normalize_price(raw),
    }
    fn = _NORMALIZERS.get(key)
    if fn is None:
        return []
    out: List[str] = []
    for raw in raw_values:
        if not raw:
            continue
        val = fn(raw, calc_mode, base)
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
    region_lbl = _intent_label((intents or {}).get("region"), key="region")
    inv_lbl = _intent_label((intents or {}).get("investment"), key="investment")
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

    # ── Paso 4b: per_capita → indicador pib_per_capita ──────────────────────
    if "per_capita" in (response.get("activity") or []):
        response["indicator"] = ["pib_per_capita"]

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
