# -*- coding: utf-8 -*-
"""
series_default.py
-----------------
Lógica modular y determinista para resolver SERIES POR DEFECTO (las más consultadas)
independiente del resto del código. Pensado para que varias personas colaboren
sin tocar el orquestador ni los módulos de búsqueda vectorial.

Conceptos clave:
- "key" (dominio lógico): IMACEC, PIB, etc.
- "variant" (subtipo determinista): GENERAL, MINERO, NO_MINERO, DESESTACIONALIZADA, etc.
- "series_id": identificador del BCCh (string opaco).
- "frequency" y "agg": parámetros por defecto para obtención.

Uso típico en orquestador:
    from series_default import resolve_series_from_text, DEFAULT_SERIES
    sd = resolve_series_from_text(question, default_key="IMACEC")
    series_id = sd.series_id
    freq = sd.frequency
    agg = sd.agg

Reglas de texto (deterministas):
- Contiene "no minero" → variante NO_MINERO
- Contiene "minero"     → variante MINERO
- Si no se encuentra calificador → GENERAL

Extensión colaborativa:
- Añadir nuevas variantes en DEFAULT_SERIES.
- Ajustar reglas en `resolve_variant_from_text`.
- (Opcional) Cargar desde JSON externo con `load_catalog_json(path)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import json
import logging

@dataclass(frozen=True)
class SeriesDef:
    key: str
    variant: str
    series_id: str
    frequency: str = "M"
    agg: str = "avg"
    synonyms: List[str] = None  # sinónimos para matching textual (opcional)

# Catalogo mínimo de series por defecto (las más consultadas)
# Puedes ampliarlo libremente manteniendo la estructura.
DEFAULT_SERIES: List[SeriesDef] = [
    # IMACEC general
    SeriesDef(
        key="IMACEC",
        variant="GENERAL",
        series_id="F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
        frequency="M",
        agg="avg",
        synonyms=["imacec", "indice mensual de actividad economica", "actividad economica"],
    ),
    # IMACEC minero
    SeriesDef(
        key="IMACEC",
        variant="MINERO",
        series_id="F032.IMC.IND.Z.Z.EP18.03.Z.0.M",
        frequency="M",
        agg="avg",
        synonyms=["imacec minero", "minero", "minería"],
    ),
    # IMACEC no minero
    SeriesDef(
        key="IMACEC",
        variant="NO_MINERO",
        series_id="F032.IMC.IND.Z.Z.EP18.N03.Z.0.M",
        frequency="M",
        agg="avg",
        synonyms=["imacec no minero", "no minero", "no-minero"],
    ),
    # PIB total (ejemplo general)
    SeriesDef(
        key="PIB",
        variant="GENERAL",
        series_id="F032.PIB.FLU.N.CLP.EP18.Z.Z.0.T",
        frequency="Q",
        agg="avg",
        synonyms=["pib", "producto interno bruto"],
    ),
    # Imacec servicios
    SeriesDef(
        key="IMACEC",
        variant="SERVICIOS",
        series_id="F032.IMC.IND.Z.Z.EP18.SERV.Z.0.M",
        frequency="M",
        agg="avg",
        synonyms=["imacec servicios", "servicios"],
    ),
]

# Índices auxiliares en memoria para resolución rápida
_INDEX_DEFAULT_BY_KEY: Dict[str, SeriesDef] = {}
_INDEX_BY_VARIANT: Dict[str, Dict[str, SeriesDef]] = {}
_INDEX_BY_SYNONYM: Dict[str, SeriesDef] = {}

# Inicialización de índices
def _build_indexes() -> None:
    _INDEX_DEFAULT_BY_KEY.clear()
    _INDEX_BY_VARIANT.clear()
    _INDEX_BY_SYNONYM.clear()
    for sd in DEFAULT_SERIES:
        key_u = sd.key.upper()
        var_u = sd.variant.upper()
        # default por key: GENERAL si existe, sino el primero encontrado
        if var_u == "GENERAL" and key_u not in _INDEX_DEFAULT_BY_KEY:
            _INDEX_DEFAULT_BY_KEY[key_u] = sd
        # por variante
        _INDEX_BY_VARIANT.setdefault(key_u, {})[var_u] = sd
        # por sinónimo
        for s in (sd.synonyms or []):
            _INDEX_BY_SYNONYM[s.lower()] = sd
    # fallback: si alguna key no obtuvo GENERAL, usar el primero por esa key
    for sd in DEFAULT_SERIES:
        key_u = sd.key.upper()
        if key_u not in _INDEX_DEFAULT_BY_KEY:
            _INDEX_DEFAULT_BY_KEY[key_u] = sd

_build_indexes()

# --- Reglas deterministas de texto -> variante ---
def resolve_variant_from_text(text: str, default_key: Optional[str] = None) -> Optional[str]:
    t = (text or "").lower()
    if not t:
        return None
    # Buscar por sinónimos definidos en el catálogo (JSON/DEFAULT_SERIES)
    for sd in DEFAULT_SERIES:
        if default_key and sd.key.upper() != default_key.upper():
            continue
        for syn in (sd.synonyms or []):
            if syn.lower() in t:
                logging.getLogger("orchestrator").info(
                    f"[SERIES_DEFAULT] variant_match_by_syn key={sd.key} variant={sd.variant} syn='{syn}'"
                )
                return sd.variant.upper()
    return None  # No se detectó calificador por catálogo

# --- Resolución principal ---
def resolve_series_from_text(text: str, default_key: str) -> Optional[SeriesDef]:
    """
    Determinista: para una pregunta y un default_key (ej. "IMACEC"),
    devuelve la SeriesDef apropiada (GENERAL / MINERO / NO_MINERO),
    o el default si no hay calificador en el texto.
    """
    # 1) Matching por sinónimos directos (catálogo) con prioridad:
    #    - Preferir sinónimos más largos (más específicos)
    #    - Preferir variantes distintas de GENERAL cuando haya varias coincidencias
    t = (text or "").lower()
    matches: List[tuple] = []  # (syn, sd)
    for syn, sd in _INDEX_BY_SYNONYM.items():
        if sd.key.upper() != default_key.upper():
            continue
        syn_l = syn.lower()
        if syn_l in t:
            matches.append((syn, sd))
    if matches:
        # Ordenar por longitud de sinónimo desc y preferir NO_GENERAL
        def _rank(item):
            syn, sd = item
            return (len(syn), 1 if sd.variant.upper() != "GENERAL" else 0)
        best_syn, best_sd = sorted(matches, key=_rank, reverse=True)[0]
        logging.getLogger("orchestrator").info(
            f"[SERIES_DEFAULT] synonym_match key={default_key.upper()} variant={best_sd.variant} sid={best_sd.series_id} syn='{best_syn}'"
        )
        return best_sd
    # 2) Regla de calificadores
    var = resolve_variant_from_text(text, default_key=default_key)
    if var:
        _sd = _INDEX_BY_VARIANT.get(default_key.upper(), {}).get(var.upper())
        logging.getLogger("orchestrator").info(
            f"[SERIES_DEFAULT] qualifier_match key={default_key.upper()} variant={var} sid={getattr(_sd,'series_id',None)}"
        )
        return _sd
    # 3) Fallback: GENERAL por key
    _fallback = _INDEX_DEFAULT_BY_KEY.get(default_key.upper())
    logging.getLogger("orchestrator").info(
        f"[SERIES_DEFAULT] fallback_default key={default_key.upper()} variant={getattr(_fallback,'variant',None)} sid={getattr(_fallback,'series_id',None)}"
    )
    return _fallback

# --- Extensión opcional: cargar catálogo desde JSON ---
def load_catalog_json(items: List[dict]) -> None:
    """
    Reemplaza el catálogo en memoria con entries provenientes de JSON ya cargado.
    Cada item debe tener: key, variant, series_id, y opcional frequency, agg, synonyms.
    """
    global DEFAULT_SERIES
    DEFAULT_SERIES = []
    for it in items:
        DEFAULT_SERIES.append(
            SeriesDef(
                key=it["key"],
                variant=it.get("variant", "GENERAL"),
                series_id=it["series_id"],
                frequency=it.get("frequency", "M"),
                agg=it.get("agg", "avg"),
                synonyms=it.get("synonyms", []),
            )
        )
    _build_indexes()

def load_catalog_json_from_path(path: Optional[str] = None) -> bool:
    """
    Carga el catálogo desde un archivo JSON si existe.
    Formato esperado:
        { "series": [ {key, variant, series_id, frequency?, agg?, synonyms?}, ... ] }
    Devuelve True si cargó correctamente, False si no encontró archivo.
    """
    try:
        if path:
            p = Path(path)
        else:
            # Por defecto: series/default_series.json (junto al proyecto)
            p = Path(__file__).resolve().parent / "series" / "default_series.json"
        if not p.is_file():
            return False
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data.get("series", [])
        if not isinstance(items, list):
            raise ValueError("Formato JSON inválido: falta 'series' lista")
        load_catalog_json(items)
        logging.getLogger("orchestrator").info(f"[SERIES_DEFAULT] catálogo cargado desde {p}")
        return True
    except Exception as e:
        logging.getLogger("orchestrator").error(f"[SERIES_DEFAULT] error cargando catálogo JSON: {e}")
        return False

# Intentar cargar catálogo externo automáticamente si existe
try:
    _loaded = load_catalog_json_from_path()
    if _loaded:
        logging.getLogger("orchestrator").info("[SERIES_DEFAULT] usando catálogo externo JSON")
except Exception:
    pass

# --- Utilidades ---
def get_default_series_for_key(default_key: str) -> Optional[SeriesDef]:
    return _INDEX_DEFAULT_BY_KEY.get(default_key.upper())

def list_variants_for_key(default_key: str) -> List[str]:
    return sorted(list((_INDEX_BY_VARIANT.get(default_key.upper()) or {}).keys()))
