"""
NER Normalizer for Economic Indicators Dataset

Este módulo normaliza entidades detectadas por un modelo NER (Named Entity Recognition)
para consultas sobre indicadores económicos chilenos (IMACEC y PIB).

Características:
- Normalización mediante fuzzy matching (tolerancia a faltas de ortografía)
- Inducción de valores faltantes según reglas semánticas
- Conversión de períodos a formato YYYY-MM-DD
- Separación de menciones compuestas con conjunción "y" (en entidades soportadas)
- Registro de entidades no reconocidas (failed matches)
- Salida en formato JSON estructurado

Reglas aplicadas (orden de ejecución conceptual):
1) Normalización base y fuzzy matching
        - Todo texto se normaliza a minúsculas y sin acentos antes de comparar.
        - Se utiliza matching fuzzy global para mapear al término más cercano del
            vocabulario por entidad (`indicator`, `seasonality`, `frequency`,
            `activity`, `region`, `investment`).
        - Para términos con negación, se priorizan variantes negativas cuando el
            texto contiene `no` (ej: `"no mineri"` -> `"no_mineria"`).

2) Separación de entidades compuestas
        - En `activity`, `region` e `investment`, expresiones con conjunción `y`
            pueden separarse en subvalores antes de normalizar.
        - Protección anti-sobre-splitting: si la frase completa ya matchea fuerte
            contra el vocabulario, no se divide.

3) Inferencia de `indicator` y `frequency`
        - `indicator` se infiere desde `frequency` cuando viene vacío/genérico.
        - Si `indicator` queda en `pib` sin frecuencia explícita -> `frequency=q`.
        - Si `indicator` queda en `imacec` sin frecuencia explícita -> `frequency=m`.
        - Regla crítica para indicador genérico (ej. `economia`) sin `frequency`:
          a) Primero: si hay `frequency` explícita, se resuelve desde ahí
              (`m` -> `imacec`, `q/a` -> `pib`).
          b) Luego: si no hay `frequency` y `req_form=point`, se evalúa `period`
              para inferir frecuencia:
            - mes -> `m` (y `indicator=imacec`)
            - trimestre (incluyendo typos) -> `q` (y `indicator=pib`)
            - año solo -> `a` (y `indicator=pib`)
          c) Si no se pudo inferir por `period` (o `req_form` no es `point`),
              solo se asigna `imacec/m` cuando `activity` cae en cobertura IMACEC
              y `intents.region` + `intents.investment` son `none`.
          d) Si hay señales de PIB (contexto de `region`/`investment` o
              cobertura de actividades PIB), se asigna `pib/q`.
          e) Si no hay señales suficientes, fallback final `imacec/m`.
        - Regla adicional: si `req_form=point`, `indicator=pib` y `period` es solo
            año (sin mes/trimestre), la frecuencia se ajusta a `a`.

4) Inferencia de `seasonality`
        - Si existe `seasonality` explícita y matchea (`sa`/`nsa`), prevalece.
        - Si no existe o no matchea:
                - `calc_mode=prev_period` -> `sa`
                - `calc_mode=yoy` -> `nsa`
                - resto (`original`, `contribution`, vacío, etc.) -> `nsa`

5) Resolución y formato de `period`
        - Salida siempre en formato `YYYY-MM-DD`.
        - `period` retorna siempre rango de 2 elementos `[inicio, fin]` para
            `latest`, `point` y `range`.
        - Contexto trimestral si `frequency=q` o el texto menciona trimestre
            (incluyendo variantes con errores ortográficos).
        - En contexto trimestral:
                - Se prioriza extracción de trimestres explícitos.
                - `point` devuelve inicio y fin de trimestre.
                - `range` devuelve [inicio menor, fin mayor] a nivel trimestral.
                - `latest` devuelve el trimestre inmediatamente anterior.
        - Si el período es solo año, se expande al año completo
            (`YYYY-01-01` a `YYYY-12-31`).
        - Rangos invertidos se ordenan automáticamente menor -> mayor.
        - Si faltan años explícitos en meses/trimestres, se usa contexto cercano o
            año actual como fallback.

6) Contrato de salida de `normalize_entities`
        - `indicator`, `seasonality`, `frequency`, `activity`, `region`,
            `investment`: listas (vacías si no hay valor).
        - `period`: lista de 2 fechas `[fecha_inicio, fecha_fin]`.

Entidades soportadas:
  - INDICATOR: imacec, pib
  - SEASONALITY: sa (ajustado estacionalmente), nsa (sin ajuste)
  - FREQUENCY: m (mensual), q (trimestral), a (anual)
  - ACTIVITY: actividades específicas por indicador
  - REGION: regiones de Chile
  - INVESTMENT: componentes de inversión/gasto
    - PERIOD: fechas/períodos en formato YYYY-MM-DD

Entrada esperada (modelo NER):
{
  "text": "cual fue la ultima cifra del imacec",
  "entities": {
    "period": ["ultima"],
    "indicator": ["imacec"]
  }
}

Salida normalizada (JSON):
{
  "normalized_entities": {
    "indicator": "imacec",
    "seasonality": "nsa",
    "frequency": "m",
    "activity": null,
    "region": null,
    "investment": null,
    "period": "2026-01-01"
  },
  "failed_matches": {
    "PERIOD": ["palabra_desconocida"]
  },
  "inference_rules_applied": [
    "FREQUENCY=m + empty INDICATOR → INDICATOR=imacec",
    "original + empty SEASONALITY → SEASONALITY=nsa"
  ]
}
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from difflib import SequenceMatcher
from datetime import datetime, timedelta


# ============================================================================
# CONSTANTES DE DICCIONARIOS (Reutilizadas del dataset_generator.py)
# ============================================================================

INDICATOR_TERMS = {
    "imacec": ["imacec"],
    "pib": ["pib", "producto interno bruto"],
}

SEASONALITY_TERMS = {
    "none": [""],
    "sa": [
        "desestacionalizado", "ajustado estacionalmente", "serie ajustada",
        "sin estacionalidad", "ajuste estacional", "con tratamiento estacional",
        "ajuste de estacionalidad", "libre de estacionalidad",
        "corregido por estacionalidad", "con ajuste estacional",
        "serie desestacionalizada", "serie con ajuste estacional",
        "eliminando el componente estacional", "quitando el ajuste estacional",
        "descontando el ajuste estacional", "ajustado por estacionalidad",
    ],
    "nsa": [
        "sin desestacionalizar", "no ajustado", "datos brutos", "serie original",
        "sin ajuste estacional", "sin efecto estacional",
        "no ajustado por estacionalidad", "sin correccion de estacionalidad",
        "sin estacionalidad aplicada", "sin ajustar por estacionalidad",
        "datos sin ajuste estacional", "serie sin ajuste estacional",
        "estacional", "con estacionalidad", "serie con estacionalidad",
        "sin eliminar estacionalidad",
    ],
}

FREQUENCY_TERMS = {
    "m": [
        "mensual", "frecuencia mensual", "periodicidad mensual", "mensualmente",
    ],
    "q": [
        "trimestral", "frecuencia trimestral", "periodicidad trimestral",
        "por trimestre", "trim",
    ],
    "a": [
        "anual", "frecuencia anual", "periodicidad anual", "anualmente",
    ],
}

ACTIVITY_TERMS_IMACEC = {
    "bienes": ["bienes", "producciones de bienes"],
    "mineria": ["mineria", "minería", "minero", "minera"],
    "industria": ["industria", "industrial"],
    "resto_bienes": ["resto de bienes", "otros bienes"],
    "comercio": ["comercio", "comercial"],
    "servicios": ["servicios"],
    "no_mineria": ["no minero", "no minería", "no mineria"],
    "impuestos": [
        "impuestos", "impuestos netos sobre productos", "impuestos netos",
        "impuestos sobre productos", "impuestos sobre productos netos"
    ],
}

ACTIVITY_TERMS_PIB = {
    "agropecuario": ["agro", "agropecuario", "agropecuaria"],
    "pesca": ["pesca", "pesquero"],
    "mineria": ["mineria", "minería", "minero", "minera"],
    "industria": [
        "industria", "industrial", "manufacturera", "industria manufacturera",
        "manufactura"
    ],
    "electricidad": [
        "electricidad", "energía", "energético", "electricidad gas y agua",
        "gas", "gas y agua", "gestión de desechos"
    ],
    "construccion": ["construcción", "constructora", "construccion"],
    "comercio": ["comercio", "comercial"],
    "restaurantes": [
        "restaurantes", "restaurant", "restoran", "hotel", "hoteles",
        "hoteles y restaurantes", "hotelería y restaurantes"
    ],
    "transporte": ["transporte", "transportes"],
    "comunicaciones": [
        "comunicaciones", "servicios de información y comunicaciones",
        "servicios de información", "comunicaciones y servicios de información"
    ],
    "servicio_financieros": ["servicios financieros", "financieros", "finanzas"],
    "servicios_empresariales": [
        "servicios empresariales", "servicios de empresas",
        "servicios profesionales", "empresariales", "empresas", 
    ],
    "servicio_viviendas": [
        "viviendas", "servicios de vivienda", "servicios inmobiliarios",
        "inmobiliarios", "servicios de vivienda y inmobiliarios"
    ],
    "servicio_personales": [
        "servicios personales", "servicios de personas", "personales"
    ],
    "admin_publica": [
        "administración pública", "servicios públicos",
        "servicios de administración pública", "admin_publica"
    ],
    "impuestos": [
        "impuestos", "impuestos netos sobre productos", "impuestos netos",
        "impuestos sobre productos", "impuestos sobre productos netos"
    ],
}

REGION_TERMS = {
    "arica_parinacota": [
        "arica y parinacota", "arica", "parinacota", "región de arica y parinacota",
        "región xv", "xv región", "región 15", "región n°15", "región nro 15",
        "decimoquinta región", "15va región", "15m región",
    ],
    "tarapaca": [
        "tarapacá", "tarapaca", "región de tarapacá", "región i", "i región",
        "región 1", "región n°1", "región nro 1", "primera región",
        "1ra región", "1a región", "1m región",
    ],
    "antofagasta": [
        "antofagasta", "región de antofagasta", "región ii", "ii región",
        "región 2", "región n°2", "región nro 2", "segunda región",
        "2da región", "2a región", "2m región",
    ],
    "atacama": [
        "atacama", "región de atacama", "región iii", "iii región",
        "región 3", "región n°3", "región nro 3", "tercera región",
        "3ra región", "3a región", "3m región",
    ],
    "coquimbo": [
        "coquimbo", "región de coquimbo", "región iv", "iv región",
        "región 4", "región n°4", "región nro 4", "cuarta región",
        "4ta región", "4a región", "4m región",
    ],
    "valparaiso": [
        "valparaíso", "valparaiso", "región de valparaíso",
        "región del valparaíso", "región v", "v región", "región 5",
        "región n°5", "región nro 5", "quinta región", "5ta región",
        "5a región", "5m región",
    ],
    "metropolitana": [
        "región metropolitana", "metropolitana", "rm", "r.m.", "santiago",
        "región de santiago", "región metropolitana de santiago",
        "región central", "región xiii", "xiii región", "región 13",
        "región n°13", "región nro 13", "decimotercera región",
        "13va región", "13m región",
    ],
    "ohiggins": [
        "región del libertador general bernardo o'higgins",
        "libertador general bernardo o'higgins", "región del libertador",
        "región de o'higgins", "o'higgins", "o higgins", "región vi",
        "vi región", "región 6", "región n°6", "región nro 6",
        "sexta región", "6ta región", "6a región", "6m región",
    ],
    "maule": [
        "maule", "región del maule", "región de maule", "región vii",
        "vii región", "región 7", "región n°7", "región nro 7",
        "séptima región", "septima región", "7ma región", "7a región",
        "7m región",
    ],
    "nuble": [
        "ñuble", "nuble", "región de ñuble", "región xvi", "xvi región",
        "región 16", "región n°16", "región nro 16", "decimosexta región",
        "16va región", "16a región", "16m región",
    ],
    "biobio": [
        "biobío", "biobio", "región del biobío", "región de biobío",
        "región viii", "viii región", "región 8", "región n°8",
        "región nro 8", "octava región", "8va región", "8a región",
        "8m región",
    ],
    "araucania": [
        "araucanía", "araucania", "región de la araucanía",
        "región de araucanía", "región ix", "ix región", "región 9",
        "región n°9", "región nro 9", "novena región", "9na región",
        "9a región", "9m región",
    ],
    "los_rios": [
        "los ríos", "los rios", "región de los ríos", "región los ríos",
        "región xiv", "xiv región", "región 14", "región n°14",
        "región nro 14", "decimocuarta región", "14va región",
        "14a región", "14m región",
    ],
    "los_lagos": [
        "los lagos", "región de los lagos", "región los lagos",
        "región x", "x región", "región 10", "región n°10",
        "región nro 10", "décima región", "decima región", "10ma región",
        "10a región", "10m región",
    ],
    "aysen": [
        "aysén", "aysen", "región de aysén",
        "región de aysén del general carlos ibáñez del campo",
        "aysén del general carlos ibáñez del campo", "región xi",
        "xi región", "región 11", "región n°11", "región nro 11",
        "undécima región", "undecima región", "11va región", "11a región",
        "11m región",
    ],
    "magallanes": [
        "magallanes", "región de magallanes",
        "región de magallanes y de la antártica chilena",
        "magallanes y de la antártica chilena", "punta arenas",
        "región xii", "xii región", "región 12", "región n°12",
        "región nro 12", "duodécima región", "duodecima región",
        "12va región", "12a región", "12m región",
    ],
}

INVESTMENT_TERMS = {
    "demanda_interna": ["demanda interna", "consumo interno", "gasto interno"],
    "consumo": [
        "consumo", "consumo final", "gasto de consumo", "consumo de los hogares",
        "consumo de las familias", "consumo de hogares", "IPSFL",
        "consumo de hogares e IPSFL"
    ],
    "consumo_gobierno": [
        "consumo del gobierno", "gasto del gobierno", "consumo público",
        "gasto público"
    ],
    "inversion": [
        "inversión", "formación de capital", "formacion bruta de capital", "capex"
    ],
    "inversion_fijo": [
        "inversión fija", "formación bruta de capital fijo", "capex fijo"
    ],
    "existencia": ["existencia", "inventarios", "variación de existencias"],
    "exportacion": [
        "exportación", "exportaciones",
        "exportacion de bienes y servicios",
        "exportaciones de bienes y servicios"
    ],
    "importacion": [
        "importación", "importaciones",
        "importacion de bienes y servicios",
        "importaciones de bienes y servicios"
    ],
    "ahorro_externo": [
        "ahorro externo", "financiamiento externo", "inversión extranjera"
    ],
    "ahorro_interno": [
        "ahorro interno", "financiamiento interno", "inversión nacional"
    ],
}

# Mapa de períodos implícitos a convertir
PERIOD_LATEST_TERMS = [
    "última", "ultima", "último", "ultimo", "reciente", "recientemente",
    "último dato", "ultima cifra", "última cifra", "último dato disponible",
    "cifra más reciente", "última publicación", "data más reciente",
]

# Números de meses y trimestres para conversión
MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
    "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

QUARTERS_START_MONTH = {
    1: 1, 2: 4, 3: 7, 4: 10,
}


# ============================================================================
# FUNCIONES AUXILIARES DE FUZZY MATCHING
# ============================================================================

def _fuzzy_match(input_text: str, target_terms: List[str],
                 threshold: float = 0.75) -> Optional[str]:
    """
    Busca la mejor coincidencia fuzzy de input_text en target_terms.
    
    Parámetros:
        input_text: Texto a normalizar
        target_terms: Lista de términos válidos para buscar coincidencia
        threshold: Mínimo ratio de similitud (0.0-1.0)
    
    Retorna:
        Término exacto de target_terms si encuentra coincidencia > threshold,
        None si no encuentra coincidencia suficiente.
    
    Algoritmo:
        - Normaliza ambos textos (minúsculas, sin acentos)
        - Usa SequenceMatcher para calcular ratio de similitud
        - Retorna el término con mayor similitud si supera threshold
    """
    if not input_text or not target_terms:
        return None

    input_normalized = _normalize_text(input_text)
    best_match = None
    best_ratio = 0.0

    for target in target_terms:
        target_normalized = _normalize_text(target)
        ratio = SequenceMatcher(None, input_normalized, target_normalized).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_match = target

    return best_match if best_ratio >= threshold else None


def _normalize_text(text: str) -> str:
    """
    Normaliza texto: minúsculas y remueve acentos.
    
    Ejemplo:
        "Minería" → "mineria"
        "Región Xv" → "region xv"
    """
    if not text:
        return ""
    
    # Convertir a minúsculas
    text = text.lower().strip()
    
    # Remover acentos
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ü': 'u', 'ñ': 'n',
    }
    for accented, unaccented in replacements.items():
        text = text.replace(accented, unaccented)
    
    return text


def _similarity_ratio(a: str, b: str) -> float:
    """Calcula similitud fuzzy entre dos strings normalizados."""
    return SequenceMatcher(None, a, b).ratio()


def _best_vocab_key(
    input_text: str,
    vocab: Dict[str, List[str]],
    threshold: float = 0.75,
    prefer_negative_if_no: bool = False,
    negative_threshold: Optional[float] = None,
) -> Optional[str]:
    """
    Retorna la clave con mejor score fuzzy global en todo el vocabulario.

    Si `prefer_negative_if_no=True` y el input contiene token "no",
    primero evalúa términos negativos (que empiezan con "no " o contienen " no ").
    """
    input_normalized = _normalize_text(input_text)
    tokens = set(input_normalized.split())

    def _find_best(candidates: Dict[str, List[str]], min_threshold: float) -> Optional[str]:
        best_key: Optional[str] = None
        best_ratio = 0.0

        for key, terms in candidates.items():
            for term in terms:
                term_norm = _normalize_text(term)
                ratio = _similarity_ratio(input_normalized, term_norm)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = key

        if best_key is not None and best_ratio >= min_threshold:
            return best_key
        return None

    if prefer_negative_if_no and "no" in tokens:
        negative_vocab: Dict[str, List[str]] = {}
        for key, terms in vocab.items():
            neg_terms = [
                t for t in terms if _normalize_text(t).startswith("no ") or " no " in _normalize_text(t)
            ]
            if neg_terms:
                negative_vocab[key] = neg_terms

        if negative_vocab:
            neg_threshold = threshold if negative_threshold is None else negative_threshold
            negative_match = _find_best(negative_vocab, neg_threshold)
            if negative_match is not None:
                return negative_match

    return _find_best(vocab, threshold)


def _is_generic_indicator_value(indicator_value: Optional[str]) -> bool:
    """Detecta menciones genéricas de indicador (ej: economía) tratándolas como indicador faltante."""
    if not indicator_value:
        return True

    generic_terms = [
        "economia",
        "actividad economica",
        "economico",
        "economica",
    ]
    indicator_normalized = _normalize_text(indicator_value)
    return any(_fuzzy_match(indicator_normalized, [term], threshold=0.72) for term in generic_terms)


def _entity_vocab_for_key(entity_key: str, indicator: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
    if entity_key == "activity":
        return ACTIVITY_TERMS_PIB if indicator == "pib" else ACTIVITY_TERMS_IMACEC
    if entity_key == "region":
        return REGION_TERMS
    if entity_key == "investment":
        return INVESTMENT_TERMS
    return None


def _split_conjoined_values(
    entity_key: str,
    raw_values: List[str],
    indicator: Optional[str],
) -> List[str]:
    """
    Divide valores con conjunción "y" para entidades compuestas.

    Aplica para activity/region/investment, excepto cuando la frase completa
    ya coincide fuertemente con un término válido del vocabulario.
    """
    if entity_key not in {"activity", "region", "investment"}:
        return raw_values

    vocab = _entity_vocab_for_key(entity_key, indicator)
    if not vocab:
        return raw_values

    expanded: List[str] = []
    for raw in raw_values:
        if not raw:
            continue

        # Si la frase completa ya es una buena coincidencia, no dividir.
        full_match = _best_vocab_key(raw, vocab=vocab, threshold=0.9)
        if full_match:
            if raw not in expanded:
                expanded.append(raw)
            continue

        parts = [part.strip() for part in re.split(r"\s+y\s+", raw, flags=re.IGNORECASE) if part.strip()]
        if len(parts) <= 1:
            if raw not in expanded:
                expanded.append(raw)
            continue

        for part in parts:
            if part not in expanded:
                expanded.append(part)

    return expanded


def _activity_match_count(raw_values: List[str], vocab: Dict[str, List[str]]) -> int:
    """Cuenta cuántas actividades raw coinciden con el vocabulario dado."""
    match_count = 0
    for raw in raw_values:
        if not raw:
            continue

        if _best_vocab_key(
            input_text=raw,
            vocab=vocab,
            threshold=0.75,
            prefer_negative_if_no=True,
            negative_threshold=0.72,
        ):
            match_count += 1

    return match_count


# ============================================================================
# FUNCIONES DE NORMALIZACIÓN POR ENTIDAD
# ============================================================================

def normalize_indicator(indicator_value: Optional[str],
                        frequency: Optional[str],
                        intent_label: Optional[str] = None,
                        context_label: Optional[str] = None) -> Optional[str]:
    """
    Normaliza INDICATOR detectado o infiere valor basado en FREQUENCY.
    
    Reglas de inferencia:
        1. Si INDICATOR existe y es cercano a términos conocidos → normalizar
        2. Si INDICATOR está vacío o no coincide:
           a. Si FREQUENCY = "m" → INDICATOR = "imacec" (mensual solo para IMACEC)
           b. Si FREQUENCY in ("q", "a") → INDICATOR = "pib" (trimestral/anual)
           c. Si FREQUENCY está vacío → asumir "imacec" (más general)
    
    Parámetros:
        indicator_value: Valor detectado por NER (ej: "imacec", "pib")
        frequency: Código de frecuencia ("m", "q", "a") o None
    
    Retorna:
        "imacec", "pib", o None si no se puede determinar
    """
    _ = context_label
    intent_normalized = _normalize_text(intent_label or "")
    is_value_intent = intent_normalized == "value"

    if _is_generic_indicator_value(indicator_value):
        if is_value_intent and frequency == "m":
            return "imacec"
        if is_value_intent and frequency in ("q", "a"):
            return "pib"
        if is_value_intent and not frequency:
            return "imacec"
        if not frequency and not is_value_intent:
            return None
        return None

    indicator_normalized = _normalize_text(indicator_value)

    # Detección explícita por palabras clave para frases compuestas
    # (ej: "pib regional" -> "pib").
    if re.search(r"\bimacec\b", indicator_normalized):
        return "imacec"
    if re.search(r"\bpib\b", indicator_normalized) or "producto interno bruto" in indicator_normalized:
        return "pib"

    match = _best_vocab_key(
        input_text=indicator_normalized,
        vocab={
            "imacec": ["imacec"],
            "pib": ["pib", "producto interno bruto", "producto bruto", "interno bruto"],
        },
        threshold=0.6,
    )
    if match:
        return match

    # Si no coincide bien, aplicar reglas de fallback condicionadas por intent.
    if is_value_intent and frequency == "m":
        return "imacec"
    elif is_value_intent and frequency in ("q", "a"):
        return "pib"
    elif is_value_intent and not frequency:
        return "imacec"
    elif not is_value_intent and not frequency:
        return None

    return None


def normalize_seasonality(seasonality_value: Optional[str],
                          calc_mode: Optional[str]) -> Optional[str]:
    """
    Normaliza SEASONALITY o infiere por defecto según calc_mode.
    
    Reglas de inferencia:
        1. Si SEASONALITY existe y coincide → usar ese valor (prioridad alta)
        2. Si SEASONALITY está vacío o no coincide:
           a. Si calc_mode = "prev_period" → asumir "sa"
           b. Si calc_mode = "yoy" → asumir "nsa"
           c. Si calc_mode in ("original", "contribution") o está vacío → asumir "nsa"
    
    Parámetros:
        seasonality_value: Valor detectado por NER (ej: "desestacionalizado")
        calc_mode: Modo de cálculo ("original", "prev_period", "yoy", "contribution")
    
    Retorna:
        "sa", "nsa", o None si no se puede determinar
    """
    if seasonality_value:
        match = _best_vocab_key(
            input_text=seasonality_value,
            vocab={
                "sa": SEASONALITY_TERMS.get("sa", []),
                "nsa": SEASONALITY_TERMS.get("nsa", []),
            },
            threshold=0.7,
            prefer_negative_if_no=True,
            negative_threshold=0.68,
        )
        if match:
            return match

    # Inferir por defecto según calc_mode (solo cuando seasonality no vino o no matcheó)
    if calc_mode == "prev_period":
        return "sa"
    if calc_mode == "yoy":
        return "nsa"
    else:
        # original, contribution, vacío, etc. → nsa
        return "nsa"


def normalize_frequency(frequency_value: Optional[str]) -> Optional[str]:
    """
    Normaliza FREQUENCY a código estándar.
    
    Parámetros:
        frequency_value: Valor detectado (ej: "mensual", "trimestral", "anual")
    
    Retorna:
        "m" (mensual), "q" (trimestral), "a" (anual), o None
    """
    if not frequency_value:
        return None

    return _best_vocab_key(
        input_text=frequency_value,
        vocab=FREQUENCY_TERMS,
        threshold=0.75,
    )


def normalize_activity(activity_value: Optional[str],
                       indicator: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """
    Normaliza ACTIVITY según indicador (IMACEC vs PIB).
    
    Parámetros:
        activity_value: Valor detectado por NER (ej: "minería", "agricultura")
        indicator: Indicador contexto ("imacec", "pib")
    
    Retorna:
        Tupla (clave_normalizada, lista_fallidos):
            - clave_normalizada: p.ej. "mineria", "agropecuario"
            - lista_fallidos: entidades que no se pudieron normalizar
    """
    if not activity_value:
        return None, []

    # Seleccionar vocabulario según indicador
    if indicator == "pib":
        activity_vocab = ACTIVITY_TERMS_PIB
    else:  # imacec (default)
        activity_vocab = ACTIVITY_TERMS_IMACEC

    activity_normalized = _normalize_text(activity_value)

    match = _best_vocab_key(
        input_text=activity_normalized,
        vocab=activity_vocab,
        threshold=0.75,
        prefer_negative_if_no=True,
        negative_threshold=0.72,
    )
    if match:
        return match, []

    # No encontrada coincidencia
    return None, [activity_value]


def normalize_region(region_value: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """
    Normaliza REGION a código de región de Chile.
    
    Parámetros:
        region_value: Valor detectado (ej: "región metropolitana", "maule")
    
    Retorna:
        Tupla (clave_normalizada, lista_fallidos):
            - clave_normalizada: p.ej. "metropolitana", "maule"
            - lista_fallidos: valores no encontrados
    """
    if not region_value:
        return None, []

    match = _best_vocab_key(
        input_text=region_value,
        vocab=REGION_TERMS,
        threshold=0.75,
    )
    if match:
        return match, []

    # No encontrada coincidencia
    return None, [region_value]


def normalize_investment(investment_value: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """
    Normaliza INVESTMENT a componente de gasto conocido.
    
    Parámetros:
        investment_value: Valor detectado (ej: "inversión", "consumo")
    
    Retorna:
        Tupla (clave_normalizada, lista_fallidos)
    """
    if not investment_value:
        return None, []

    match = _best_vocab_key(
        input_text=investment_value,
        vocab=INVESTMENT_TERMS,
        threshold=0.75,
    )
    if match:
        return match, []

    # No encontrada coincidencia
    return None, [investment_value]


def normalize_period(period_value: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """
    Normaliza PERIOD a formato YYYY-MM-DD.
    
    Formato de entrada soportado:
        - Términos implícitos: "última", "último dato", "cifra más reciente"
        - Fechas explícitas: "enero 2024", "2024-01", "2024"
        - Trimestres: "1er trimestre 2024", "T1 2024", "2024-T1"
        - Rangos: se ignoran (devuelven None)
    
    Parámetros:
        period_value: Texto de período detectado
    
    Retorna:
        Tupla (fecha_normalizada_YYYYMMDD, lista_fallidos):
            - fecha_normalizada_YYYYMMDD: p.ej. "2026-01-15" para "última"
            - lista_fallidos: períodos que no se pudieron parsear
    """
    if not period_value:
        return None, []

    period_normalized = _normalize_text(period_value)

    # Caso 1: Términos implícitos (última, reciente, etc.)
    for latest_term in PERIOD_LATEST_TERMS:
        if _fuzzy_match(period_normalized, [latest_term], threshold=0.8):
            # Usar fecha actual como fecha de último dato
            today = datetime.now()
            return _format_month_start(today), []

    # Caso 2: Año explícito (ej: "2024") o implícito (año actual)
    year_match = re.search(r'\b(20\d{2})\b', period_normalized)
    year = int(year_match.group(1)) if year_match else datetime.now().year

    # Buscar mes explícito (ej: "enero 2024" o "enero")
    for month_name, month_num in MONTHS.items():
        if month_name in period_normalized:
            return f"{year:04d}-{month_num:02d}-01", []

    # Buscar trimestre (ej: "1er trimestre 2024", "T1 2024", "T1")
    quarter_match = re.search(r'(?:t|trimestre)\s*([1-4])', period_normalized)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        start_month = QUARTERS_START_MONTH[quarter]
        return f"{year:04d}-{start_month:02d}-01", []

    # Si solo hay año, usar 01-01-YYYY
    if year_match:
        return f"{year:04d}-01-01", []

    # Caso 3: Formato ISO (ej: "2024-01", "2024-T1")
    iso_match = re.search(r'(20\d{2})-(\d{2})', period_normalized)
    if iso_match:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        return f"{year:04d}-{month:02d}-01", []

    # No se pudo parsear
    return None, [period_value]


# ============================================================================
# FUNCIÓN PRINCIPAL: NORMALIZAR ENTIDADES NER
# ============================================================================

def normalize_ner_entities(ner_output: Dict[str, Any],
                           calc_mode: Optional[str] = None,
                           intent_label: Optional[str] = None,
                           context_label: Optional[str] = None) -> Dict[str, Any]:
    """
    Normaliza todas las entidades detectadas por el modelo NER.
    
    Proceso:
        1. Normaliza cada entidad individualmente con fuzzy matching
        2. Registra entidades que fallaron (no coincidieron con vocabulario)
        3. Aplica reglas de inferencia para valores faltantes
        4. Retorna estructura normalizada JSON
    
    Parámetros:
        ner_output: Diccionario con estructura:
            {
              "text": "...",
              "interpretation": {
                "entities": {
                  "indicator": [...],
                  "seasonality": [...],
                  "frequency": [...],
                  "activity": [...],
                  "region": [...],
                  "investment": [...],
                  "period": [...]
                }
              }
            }
        calc_mode: Modo de cálculo opcional para inferencias
    
    Retorna:
        {
          "normalized_entities": {
            "indicator": "imacec|pib|None",
            "seasonality": "sa|nsa|None",
            "frequency": "m|q|a|None",
            "activity": "clave|None",
            "region": "clave|None",
            "investment": "clave|None",
            "period": "YYYY-MM-DD|None"
          },
          "failed_matches": {
            "INDICATOR": [...],
            "SEASONALITY": [...],
            ...
          },
          "inference_rules_applied": [
            "Descripción de regla aplicada",
            ...
          ]
        }
    """
    # Extraer entidades del NER output
    entities = ner_output.get("interpretation", {}).get("entities", {})

    # Parsear valores individuales (el NER devuelve listas)
    indicator_raw = entities.get("indicator", [None])[0] if entities.get("indicator") else None
    seasonality_raw = entities.get("seasonality", [None])[0] if entities.get("seasonality") else None
    frequency_raw = entities.get("frequency", [None])[0] if entities.get("frequency") else None
    activity_raw = entities.get("activity", [None])[0] if entities.get("activity") else None
    region_raw = entities.get("region", [None])[0] if entities.get("region") else None
    investment_raw = entities.get("investment", [None])[0] if entities.get("investment") else None
    period_raw = entities.get("period", [None])[0] if entities.get("period") else None

    # Normalizar entidades
    normalized_frequency = normalize_frequency(frequency_raw)
    normalized_indicator = normalize_indicator(
        indicator_raw,
        normalized_frequency,
        intent_label=intent_label,
        context_label=context_label,
    )
    normalized_seasonality = normalize_seasonality(seasonality_raw, calc_mode)
    normalized_activity, failed_activity = normalize_activity(activity_raw, normalized_indicator)
    normalized_region, failed_region = normalize_region(region_raw)
    normalized_investment, failed_investment = normalize_investment(investment_raw)
    normalized_period, failed_period = normalize_period(period_raw)

    # Registrar entidades fallidas
    failed_matches = {}
    if failed_activity:
        failed_matches["ACTIVITY"] = failed_activity
    if failed_region:
        failed_matches["REGION"] = failed_region
    if failed_investment:
        failed_matches["INVESTMENT"] = failed_investment
    if failed_period:
        failed_matches["PERIOD"] = failed_period

    # Rastro de reglas aplicadas
    inference_rules = []

    # Registrar inferencias realizadas
    if not indicator_raw:
        if normalized_frequency == "m":
            inference_rules.append("FREQUENCY='m' + empty INDICATOR → INDICATOR='imacec'")
        elif normalized_frequency in ("q", "a"):
            inference_rules.append("FREQUENCY in ('q','a') + empty INDICATOR → INDICATOR='pib'")
        else:
            inference_rules.append("empty INDICATOR + empty FREQUENCY → INDICATOR='imacec' (default)")

    if not seasonality_raw:
        if calc_mode == "prev_period":
            inference_rules.append("calc_mode='prev_period' + empty SEASONALITY → SEASONALITY='sa'")
        else:
            inference_rules.append("calc_mode in ('original','yoy','contribution') + empty SEASONALITY → SEASONALITY='nsa'")

    return {
        "normalized_entities": {
            "indicator": normalized_indicator,
            "seasonality": normalized_seasonality,
            "frequency": normalized_frequency,
            "activity": normalized_activity,
            "region": normalized_region,
            "investment": normalized_investment,
            "period": normalized_period,
        },
        "failed_matches": failed_matches if failed_matches else None,
        "inference_rules_applied": inference_rules if inference_rules else None,
    }


# ============================================================================
# WRAPPER PARA API: SALIDA LIMPIA Y SOPORTE DE LISTAS
# ============================================================================

_ENTITY_KEYS = (
    "indicator",
    "seasonality",
    "frequency",
    "activity",
    "region",
    "investment",
    "period",
)


def _normalize_multiple_values(
    entity_key: str,
    raw_values: List[str],
    calc_mode: Optional[str],
    base_normalized: Dict[str, Optional[str]],
    intent_label: Optional[str] = None,
    context_label: Optional[str] = None,
) -> List[str]:
    """
    Normaliza múltiples valores para una misma entidad.

    Si hay más de un elemento en la entidad, devuelve lista (sin duplicados,
    preservando orden). Si solo queda uno, devuelve string.
    """
    normalized_values: List[str] = []

    for raw in raw_values:
        if not raw:
            continue

        normalized_value: Optional[str]
        if entity_key == "indicator":
            normalized_value = normalize_indicator(
                raw,
                base_normalized.get("frequency"),
                intent_label=intent_label,
                context_label=context_label,
            )
        elif entity_key == "seasonality":
            normalized_value = normalize_seasonality(raw, calc_mode)
        elif entity_key == "frequency":
            normalized_value = normalize_frequency(raw)
        elif entity_key == "activity":
            normalized_value = normalize_activity(raw, base_normalized.get("indicator"))[0]
        elif entity_key == "region":
            normalized_value = normalize_region(raw)[0]
        elif entity_key == "investment":
            normalized_value = normalize_investment(raw)[0]
        elif entity_key == "period":
            normalized_value = normalize_period(raw)[0]
        else:
            normalized_value = None

        if normalized_value and normalized_value not in normalized_values:
            normalized_values.append(normalized_value)

    return normalized_values


def _as_list(value: Optional[Union[str, List[str]]]) -> List[str]:
    """Normaliza un valor escalar/lista/null a lista homogénea."""
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item]
    return [value] if value else []


def _parse_yyyymmdd(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


def _format_month_start(date_value: datetime) -> str:
    """Formatea una fecha al primer día de su mes en YYYY-MM-DD."""
    return f"{date_value.year:04d}-{date_value.month:02d}-01"


def _format_year_start(date_value: datetime) -> str:
    """Formatea una fecha al primer día de su año en YYYY-MM-DD."""
    return f"{date_value.year:04d}-01-01"


def _previous_month_anchor(date_value: datetime) -> datetime:
    if date_value.month == 1:
        return datetime(date_value.year - 1, 12, 1)
    return datetime(date_value.year, date_value.month - 1, 1)


def _previous_quarter_anchor(date_value: datetime) -> datetime:
    quarter_start = _quarter_start_month(date_value.month)
    if quarter_start == 1:
        return datetime(date_value.year - 1, 10, 1)
    return datetime(date_value.year, quarter_start - 3, 1)


def _previous_year_anchor(date_value: datetime) -> datetime:
    return datetime(date_value.year - 1, 1, 1)


def _quarter_start_month(month: int) -> int:
    """Retorna mes inicial del trimestre para un mes dado."""
    return ((month - 1) // 3) * 3 + 1


def _format_quarter_start(date_value: datetime) -> str:
    """Formatea una fecha al inicio de su trimestre en YYYY-MM-DD."""
    quarter_month = _quarter_start_month(date_value.month)
    return f"{date_value.year:04d}-{quarter_month:02d}-01"


def _format_month_end(date_value: datetime) -> str:
    """Formatea una fecha al último día de su mes en YYYY-MM-DD."""
    if date_value.month == 12:
        next_month = datetime(date_value.year + 1, 1, 1)
    else:
        next_month = datetime(date_value.year, date_value.month + 1, 1)
    last_day = next_month - timedelta(days=1)
    return last_day.strftime("%Y-%m-%d")


def _format_quarter_end(date_value: datetime) -> str:
    """Formatea una fecha al último día de su trimestre en YYYY-MM-DD."""
    quarter_start = _quarter_start_month(date_value.month)
    quarter_end_month = quarter_start + 2
    quarter_end_anchor = datetime(date_value.year, quarter_end_month, 1)
    return _format_month_end(quarter_end_anchor)


def _extract_quarter_based_dates(text: str) -> List[str]:
    """Extrae fechas trimestrales YYYY-MM-DD desde menciones de trimestres."""
    normalized_text = _normalize_text(text)
    tokens = re.findall(r'[a-záéíóúüñ0-9]+', normalized_text)
    if not tokens:
        return []

    quarter_word_map = {
        "primer": 1,
        "primero": 1,
        "segundo": 2,
        "tercer": 3,
        "tercero": 3,
        "cuarto": 4,
    }

    quarter_tokens: List[Tuple[int, int]] = []  # (token_idx, quarter)
    year_tokens: List[Tuple[int, int]] = []     # (token_idx, year)

    for idx, token in enumerate(tokens):
        if re.fullmatch(r'20\d{2}', token):
            year_tokens.append((idx, int(token)))
            continue

        compact_quarter = re.fullmatch(r'[tq]([1-4])', token)
        if compact_quarter:
            quarter_tokens.append((idx, int(compact_quarter.group(1))))
            continue

        if token in quarter_word_map and idx + 1 < len(tokens) and _is_trimester_like_token(tokens[idx + 1]):
            quarter_tokens.append((idx, quarter_word_map[token]))
            continue

        if token in {"1", "2", "3", "4"} and idx + 1 < len(tokens) and _is_trimester_like_token(tokens[idx + 1]):
            quarter_tokens.append((idx, int(token)))
            continue

    if not quarter_tokens:
        return []

    resolved_dates: List[str] = []
    current_year = datetime.now().year

    for quarter_idx, quarter in quarter_tokens:
        right_year = next((year for idx, year in year_tokens if idx > quarter_idx), None)
        left_year = next((year for idx, year in reversed(year_tokens) if idx < quarter_idx), None)
        year = right_year if right_year is not None else left_year
        if year is None:
            year = current_year

        start_month = QUARTERS_START_MONTH[quarter]
        date_value = f"{year:04d}-{start_month:02d}-01"
        if date_value not in resolved_dates:
            resolved_dates.append(date_value)

    return resolved_dates


def _has_quarter_reference(text: str) -> bool:
    normalized_text = _normalize_text(text)
    if any(_is_trimester_like_token(token) for token in re.findall(r'[a-záéíóúüñ0-9]+', normalized_text)):
        return True
    return re.search(r'\b[tq]\s*[1-4]\b|\b[tq][1-4]\b', normalized_text) is not None


def _is_trimester_like_token(token: str) -> bool:
    """Detecta variantes de 'trimestre' con tolerancia a typos."""
    if not token:
        return False

    normalized_token = _normalize_text(token)
    if normalized_token.startswith("trimestre"):
        return True

    return _fuzzy_match(
        normalized_token,
        ["trimestre", "trimestres"],
        threshold=0.74,
    ) is not None


def _extract_year_based_dates(text: str) -> List[str]:
    """Extrae años explícitos como fechas de inicio de año YYYY-MM-DD."""
    normalized_text = _normalize_text(text)
    year_matches = re.findall(r'\b(20\d{2})\b', normalized_text)
    if not year_matches:
        return []

    resolved_dates: List[str] = []
    for year_text in year_matches:
        date_value = f"{int(year_text):04d}-01-01"
        if date_value not in resolved_dates:
            resolved_dates.append(date_value)

    return resolved_dates


def _is_year_only_period_reference(text: str) -> bool:
    """Indica si el texto refiere a un año sin mes ni trimestre explícito."""
    if not text:
        return False

    normalized_text = _normalize_text(text)
    has_year = bool(re.search(r'\b20\d{2}\b', normalized_text))
    if not has_year:
        return False

    has_month_reference = any(month_name in normalized_text for month_name in MONTHS)
    has_quarter_reference = _has_quarter_reference(text)
    return has_year and not has_month_reference and not has_quarter_reference


def _infer_frequency_from_period_for_point(raw_values: List[str]) -> Optional[str]:
    """Infiere frecuencia m/q/a desde period cuando req_form=point."""
    valid_values = [raw for raw in raw_values if raw]
    if not valid_values:
        return None

    if any(_has_quarter_reference(raw) for raw in valid_values):
        return "q"

    if any(_extract_month_based_dates(raw) for raw in valid_values):
        return "m"

    if any(_is_year_only_period_reference(raw) for raw in valid_values):
        return "a"

    return None


def _extract_month_based_dates(text: str) -> List[str]:
    """Extrae fechas YYYY-MM-DD desde menciones de meses en un texto de período."""
    normalized_text = _normalize_text(text)
    tokens = re.findall(r'[a-záéíóúüñ0-9]+', normalized_text)
    if not tokens:
        return []

    month_names = list(MONTHS.keys())
    month_tokens: List[Tuple[int, int]] = []  # (token_idx, month_num)
    year_tokens: List[Tuple[int, int]] = []   # (token_idx, year)

    for idx, token in enumerate(tokens):
        if re.fullmatch(r'20\d{2}', token):
            year_tokens.append((idx, int(token)))
            continue

        month_match = _fuzzy_match(token, month_names, threshold=0.78)
        if month_match:
            month_tokens.append((idx, MONTHS[month_match]))

    if not month_tokens:
        return []

    if not year_tokens:
        current_year = datetime.now().year
        resolved_dates: List[str] = []
        for _, month_num in month_tokens:
            date_value = f"{current_year:04d}-{month_num:02d}-01"
            if date_value not in resolved_dates:
                resolved_dates.append(date_value)
        return resolved_dates

    resolved_dates: List[str] = []
    for month_idx, month_num in month_tokens:
        right_year = next((year for idx, year in year_tokens if idx > month_idx), None)
        left_year = next((year for idx, year in reversed(year_tokens) if idx < month_idx), None)
        year = right_year if right_year is not None else left_year
        if year is None:
            continue

        date_value = f"{year:04d}-{month_num:02d}-01"
        if date_value not in resolved_dates:
            resolved_dates.append(date_value)

    return resolved_dates


def _resolve_period_value(
    raw_values: List[str],
    calc_mode: Optional[str],
    base_normalized: Dict[str, Optional[str]],
    req_form: Optional[str],
    frequency: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Resuelve period según req_form:
    - latest -> [fecha_inicio, fecha_fin]
    - point  -> [fecha_inicio, fecha_fin]
    - range  -> [fecha_inicio, fecha_fin]
    """
    candidate_dates: List[str] = []
    is_quarterly_context = (frequency == "q") or any(_has_quarter_reference(raw) for raw in raw_values if raw)
    has_year_only_reference = False

    for raw in raw_values:
        if not raw:
            continue

        # En contexto trimestral, priorizar extracción de trimestres explícitos.
        extracted_dates = _extract_quarter_based_dates(raw) if is_quarterly_context else []

        # Luego intentar extracción por meses (útil para contexto mensual o fallback).
        if not extracted_dates:
            extracted_dates = _extract_month_based_dates(raw)

        for extracted in extracted_dates:
            if extracted not in candidate_dates:
                candidate_dates.append(extracted)

        # Si no hubo extracción de pares mes-año/trimestre, intentar años explícitos.
        if not extracted_dates:
            year_dates = _extract_year_based_dates(raw)
            for year_date in year_dates:
                if year_date not in candidate_dates:
                    candidate_dates.append(year_date)

            normalized_raw = _normalize_text(raw)
            has_month_reference = any(month_name in normalized_raw for month_name in MONTHS)
            if year_dates and not has_month_reference and not _has_quarter_reference(raw):
                has_year_only_reference = True

        # Si no hubo extracción, usar parse simple como fallback
        if not extracted_dates and not _extract_year_based_dates(raw):
            normalized_single = normalize_period(raw)[0]
            if normalized_single and normalized_single not in candidate_dates:
                candidate_dates.append(normalized_single)

    if not candidate_dates:
        base_period = base_normalized.get("period")
        if base_period:
            candidate_dates.append(base_period)

    req_form_norm = (req_form or "").strip().lower()

    if not candidate_dates:
        if req_form_norm == "latest":
            today = datetime.now()
            if is_quarterly_context:
                prev_quarter = _previous_quarter_anchor(today)
                return [_format_quarter_start(prev_quarter), _format_quarter_end(prev_quarter)]
            prev_month = _previous_month_anchor(today)
            return [_format_month_start(prev_month), _format_month_end(prev_month)]
        return None

    parsed_dates = [
        parsed for parsed in (_parse_yyyymmdd(value) for value in candidate_dates) if parsed is not None
    ]
    if not parsed_dates:
        return None

    if is_quarterly_context:
        quarter_dates: List[datetime] = []
        seen_quarters = set()
        for date_value in parsed_dates:
            quarter_date = datetime(date_value.year, _quarter_start_month(date_value.month), 1)
            quarter_key = (quarter_date.year, quarter_date.month)
            if quarter_key not in seen_quarters:
                seen_quarters.add(quarter_key)
                quarter_dates.append(quarter_date)
        parsed_dates = quarter_dates

    if req_form_norm == "latest":
        if has_year_only_reference and all(date_value.month == 1 and date_value.day == 1 for date_value in parsed_dates):
            prev_year = _previous_year_anchor(datetime.now())
            return [_format_year_start(prev_year), f"{prev_year.year:04d}-12-31"]
        if is_quarterly_context:
            prev_quarter = _previous_quarter_anchor(datetime.now())
            return [_format_quarter_start(prev_quarter), _format_quarter_end(prev_quarter)]
        prev_month = _previous_month_anchor(datetime.now())
        return [_format_month_start(prev_month), _format_month_end(prev_month)]

    if req_form_norm == "range":
        sorted_dates = sorted(parsed_dates)
        first_date = _format_month_start(sorted_dates[0])
        if has_year_only_reference and all(date_value.month == 1 and date_value.day == 1 for date_value in sorted_dates):
            last_date = f"{sorted_dates[-1].year:04d}-12-31"
        elif is_quarterly_context:
            last_date = _format_quarter_end(sorted_dates[-1])
        else:
            last_date = _format_month_end(sorted_dates[-1])
        return [first_date, last_date]

    # point (o default): fecha mencionada
    first_date = parsed_dates[0]
    if has_year_only_reference:
        return [_format_year_start(first_date), f"{first_date.year:04d}-12-31"]
    if is_quarterly_context:
        return [_format_quarter_start(first_date), _format_quarter_end(first_date)]
    return [_format_month_start(first_date), _format_month_end(first_date)]


def normalize_entities(
    entities: Dict[str, List[str]],
    calc_mode: Optional[str] = None,
    req_form: Optional[str] = None,
    intents: Optional[Dict[str, Any]] = None,
    intent_label: Optional[str] = None,
    context_label: Optional[str] = None,
) -> Dict[str, Union[List[str], None]]:
    """
    API-friendly normalizer used by `/predict`.

    - No retorna `failed_matches`
    - No retorna `inference_rules_applied`
    - Todas las entidades excepto period retornan listas (vacía si no hay normalizado)
    - period retorna lista de 2 elementos en todos los casos
        - Puede usar `intents` (activity/region/investment) para inferencias de
            indicator/frequency
    """
    def _extract_label(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, dict):
            label = value.get("label")
            return str(label).lower() if label is not None else None
        return str(value).lower()

    routing_intent_label = _extract_label(intent_label)
    if routing_intent_label is None:
        routing_intent_label = _extract_label((intents or {}).get("intent"))

    routing_context_label = _extract_label(context_label)
    if routing_context_label is None:
        routing_context_label = _extract_label((intents or {}).get("context"))

    ner_output = {"interpretation": {"entities": entities}}
    base_result = normalize_ner_entities(
        ner_output,
        calc_mode=calc_mode,
        intent_label=routing_intent_label,
        context_label=routing_context_label,
    )
    base_normalized = base_result.get("normalized_entities", {})

    response: Dict[str, Union[List[str], str, None]] = {}
    for key in _ENTITY_KEYS:
        if key == "period":
            continue

        raw_values = entities.get(key) or []

        # Separar expresiones compuestas con "y" en activity/region/investment
        # antes de normalizar cada subentidad.
        raw_values = _split_conjoined_values(
            entity_key=key,
            raw_values=raw_values,
            indicator=base_normalized.get("indicator"),
        )

        if len(raw_values) > 1:
            value = _normalize_multiple_values(
                key,
                raw_values,
                calc_mode,
                base_normalized,
                intent_label=routing_intent_label,
                context_label=routing_context_label,
            )
        else:
            value = _as_list(base_normalized.get(key))

        response[key] = value

    # Regla de negocio: indicador genérico/vacío depende de frequency + intent.
    raw_indicator = (entities.get("indicator") or [None])[0]
    raw_frequency = (entities.get("frequency") or [None])[0]
    indicator_is_generic_or_missing = _is_generic_indicator_value(raw_indicator)
    intent_is_value = _normalize_text(routing_intent_label or "") == "value"
    period_raw_values = entities.get("period") or []

    if indicator_is_generic_or_missing and not raw_frequency:
        if intent_is_value:
            response["indicator"] = ["imacec"]
            response["frequency"] = ["m"]
        else:
            response["indicator"] = []
    elif not raw_frequency and "pib" in response.get("indicator", []):
        response["frequency"] = ["q"]
    elif not raw_frequency and "imacec" in response.get("indicator", []):
        response["frequency"] = ["m"]

    effective_frequency = response.get("frequency", [])
    frequency_code = effective_frequency[0] if isinstance(effective_frequency, list) and effective_frequency else None
    response["period"] = _resolve_period_value(
        raw_values=period_raw_values,
        calc_mode=calc_mode,
        base_normalized=base_normalized,
        req_form=req_form,
        frequency=frequency_code,
    )

    return response


# ============================================================================
# FUNCIÓN WRAPPER: PROCESAR JSON de entrada DIRECTA
# ============================================================================

def normalize_from_json(json_input: str, calc_mode: Optional[str] = None) -> str:
    """
    Procesa entrada JSON del modelo NER y retorna salida normalizada JSON.
    
    Entrada esperada:
    {
      "text": "cual fue la ultima cifra del imacec",
      "interpretation": {
        "entities": {
          "indicator": ["imacec"],
          "period": ["ultima"]
        }
      }
    }
    
    Salida:
    {
      "normalized_entities": {...},
      "failed_matches": {...},
      "inference_rules_applied": [...]
    }
    
    Parámetros:
        json_input: String JSON con salida del modelo NER
        calc_mode: Modo de cálculo opcional para contexto
    
    Retorna:
        String JSON formateado con entidades normalizadas
    """
    try:
        ner_output = json.loads(json_input)
        result = normalize_ner_entities(ner_output, calc_mode)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Invalid JSON input: {str(e)}",
            "failed_matches": None,
            "inference_rules_applied": None,
        }, ensure_ascii=False, indent=2)


# ============================================================================
# EJEMPLOS Y TEST
# ============================================================================

if __name__ == "__main__":
    # Ejemplo 1: Entrada simple con INDICATOR y PERIOD
    example1 = """
    {
      "text": "cual fue la ultima cifra del imacec",
      "interpretation": {
        "entities": {
          "indicator": ["imacec"],
          "period": ["ultima"]
        }
      }
    }
    """
    print("EJEMPLO 1: Búsqueda simple del IMACEC más reciente")
    print("Entrada:", example1.strip())
    result1 = normalize_from_json(example1)
    print("Salida:", result1)
    print("\n" + "="*70 + "\n")

    # Ejemplo 2: Entrada con FREQUENCY para inferir INDICATOR
    example2 = """
    {
      "text": "dame los datos trimestrales de actividad",
      "interpretation": {
        "entities": {
          "frequency": ["trimestral"]
        }
      }
    }
    """
    print("EJEMPLO 2: Inferir INDICATOR desde FREQUENCY")
    print("Entrada:", example2.strip())
    result2 = normalize_from_json(example2, calc_mode="original")
    print("Salida:", result2)
    print("\n" + "="*70 + "\n")

    # Ejemplo 3: Entrada con múltiples entidades
    example3 = """
    {
      "text": "cual fue la actividad minera en metropolitana febrero 2024 desestacionalizado",
      "interpretation": {
        "entities": {
          "indicator": ["imacec"],
          "activity": ["mineria"],
          "region": ["metropolitana"],
          "period": ["febrero 2024"],
          "seasonality": ["desestacionalizado"]
        }
      }
    }
    """
    print("EJEMPLO 3: Normalizar múltiples entidades")
    print("Entrada:", example3.strip())
    result3 = normalize_from_json(example3, calc_mode="original")
    print("Salida:", result3)
    print("\n" + "="*70 + "\n")

    # Ejemplo 4: Con entidades no reconocidas (fuzzy matching fallido)
    example4 = """
    {
      "text": "cual fue la producion agricola sin ajuste",
      "interpretation": {
        "entities": {
          "indicator": [],
          "activity": ["producion agricola"],
          "frequency": ["anual"],
          "seasonality": ["sin ajuste"]
        }
      }
    }
    """
    print("EJEMPLO 4: Con entidades no reconocidas")
    print("Entrada:", example4.strip())
    result4 = normalize_from_json(example4, calc_mode="original")
    print("Salida:", result4)