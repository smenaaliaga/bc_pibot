"""
Vocabularios y constantes del normalizador NER.

Contiene todos los diccionarios de términos válidos por entidad
(indicator, seasonality, frequency, activity, region, investment, period),
así como mapas auxiliares de meses, trimestres, números en español, etc.

Estos vocabularios son la única fuente de verdad para el matching fuzzy
y la validación de entidades. Cualquier nuevo sinónimo o alias debe
agregarse aquí.
"""

from typing import Dict, List

# ─── Indicadores ───────────────────────────────────────────────────────────────

INDICATOR_TERMS: Dict[str, List[str]] = {
    "imacec": ["imacec"],
    "pib": ["pib", "producto interno bruto"],
}

# ─── Estacionalidad ───────────────────────────────────────────────────────────

SEASONALITY_TERMS: Dict[str, List[str]] = {
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

# ─── Frecuencia ────────────────────────────────────────────────────────────────

FREQUENCY_TERMS: Dict[str, List[str]] = {
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

# ─── Actividades por indicador ─────────────────────────────────────────────────

ACTIVITY_TERMS_IMACEC: Dict[str, List[str]] = {
    "bienes": ["bienes", "producciones de bienes"],
    "mineria": ["mineria", "minería", "minero", "minera"],
    "industria": ["industria", "industrial"],
    "resto_bienes": ["resto de bienes", "otros bienes"],
    "comercio": ["comercio", "comercial"],
    "servicios": ["servicios"],
    "no_mineria": ["no minero", "no minería", "no mineria"],
    "impuestos": [
        "impuestos", "impuestos netos sobre productos", "impuestos netos",
        "impuestos sobre productos", "impuestos sobre productos netos",
    ],
    "per_capita": [
        "per capita", "per cápita", "percapita", "por habitante", "por persona",
        "por cada habitante",
    ],
}

ACTIVITY_TERMS_PIB: Dict[str, List[str]] = {
    "agropecuario": ["agro", "agropecuario", "agropecuaria"],
    "pesca": ["pesca", "pesquero"],
    "mineria": ["mineria", "minería", "minero", "minera"],
    "no_mineria": ["no minero", "no minería", "no mineria"],
    "industria": [
        "industria", "industrial", "manufacturera", "industria manufacturera",
        "manufactura",
    ],
    "electricidad": [
        "electricidad", "energía", "energético", "electricidad gas y agua",
        "gas", "gas y agua", "gestión de desechos",
    ],
    "construccion": ["construcción", "constructora", "construccion"],
    "comercio": ["comercio", "comercial"],
    "restaurantes": [
        "restaurantes", "restaurant", "restoran", "hotel", "hoteles",
        "hoteles y restaurantes", "hotelería y restaurantes",
    ],
    "transporte": ["transporte", "transportes"],
    "comunicaciones": [
        "comunicaciones", "servicios de información y comunicaciones",
        "servicios de información", "comunicaciones y servicios de información",
    ],
    "servicios_financieros": ["servicios financieros", "financieros", "finanzas"],
    "servicios_empresariales": [
        "servicios empresariales", "servicios de empresas",
        "servicios profesionales", "empresariales", "empresas",
    ],
    "servicio_viviendas": [
        "viviendas", "servicios de vivienda", "servicios inmobiliarios",
        "inmobiliarios", "servicios de vivienda y inmobiliarios",
    ],
    "servicio_personales": [
        "servicios personales", "servicios de personas", "personales",
    ],
    "admin_publica": [
        "administración pública", "servicios públicos",
        "servicios de administración pública", "admin_publica",
    ],
    "impuestos": [
        "impuestos", "impuestos netos sobre productos", "impuestos netos",
        "impuestos sobre productos", "impuestos sobre productos netos",
    ],
    "per_capita": [
        "per capita", "per cápita", "por habitante", "por persona",
        "por cada habitante",
    ],
}

# Subconjunto de actividades PIB para contexto regional.
ACTIVITY_TERMS_PIB_REGIONAL: Dict[str, List[str]] = {
    "bienes": ["bienes", "producciones de bienes"],
    "mineria": ["mineria", "minería", "minero", "minera"],
    "industria": [
        "industria", "industrial", "manufacturera", "industria manufacturera",
        "manufactura",
    ],
    "resto_bienes": ["resto de bienes", "otros bienes"],
    "comercio": ["comercio", "comercial"],
    "servicios": ["servicios"],
}

# ─── Regiones de Chile ─────────────────────────────────────────────────────────

REGION_TERMS: Dict[str, List[str]] = {
    "arica_parinacota": [
        "arica y parinacota", "arica", "parinacota", "región de arica y parinacota",
        "región xv", "xv región", "región 15", "región n°15", "región nro 15",
        "decimoquinta región", "15va región", "15m región", "xv",
    ],
    "tarapaca": [
        "tarapacá", "tarapaca", "región de tarapacá", "región i", "i región",
        "región 1", "región n°1", "región nro 1", "primera región",
        "1ra región", "1a región", "1m región", "i",
    ],
    "antofagasta": [
        "antofagasta", "región de antofagasta", "región ii", "ii región",
        "región 2", "región n°2", "región nro 2", "segunda región",
        "2da región", "2a región", "2m región", "ii",
    ],
    "atacama": [
        "atacama", "región de atacama", "región iii", "iii región",
        "región 3", "región n°3", "región nro 3", "tercera región",
        "3ra región", "3a región", "3m región", "iii",
    ],
    "coquimbo": [
        "coquimbo", "región de coquimbo", "región iv", "iv región",
        "región 4", "región n°4", "región nro 4", "cuarta región",
        "4ta región", "4a región", "4m región", "iv",
    ],
    "valparaiso": [
        "valparaíso", "valparaiso", "región de valparaíso",
        "región del valparaíso", "región v", "v región", "región 5",
        "región n°5", "región nro 5", "quinta región", "5ta región",
        "5a región", "5m región", "v",
    ],
    "metropolitana": [
        "región metropolitana", "metropolitana", "rm", "r.m.", "santiago",
        "región de santiago", "región metropolitana de santiago",
        "región central", "región xiii", "xiii región", "región 13",
        "región n°13", "región nro 13", "decimotercera región",
        "13va región", "13m región", "xiii",
    ],
    "ohiggins": [
        "región del libertador general bernardo o'higgins",
        "libertador general bernardo o'higgins", "región del libertador",
        "región de o'higgins", "o'higgins", "o higgins", "región vi",
        "bernardo ohiggins", "bernardo o higgins", "region de bernardo ohiggins",
        "vi región", "región 6", "región n°6", "región nro 6",
        "sexta región", "6ta región", "6a región", "6m región", "vi",
    ],
    "maule": [
        "maule", "región del maule", "región de maule", "región vii",
        "vii región", "región 7", "región n°7", "región nro 7",
        "séptima región", "septima región", "7ma región", "7a región",
        "7m región", "vii",
    ],
    "nuble": [
        "ñuble", "nuble", "región de ñuble", "región xvi", "xvi región",
        "región 16", "región n°16", "región nro 16", "decimosexta región",
        "16va región", "16a región", "16m región", "xvi",
    ],
    "biobio": [
        "biobío", "biobio", "región del biobío", "región de biobío",
        "región viii", "viii región", "región 8", "región n°8",
        "región nro 8", "octava región", "8va región", "8a región",
        "8m región", "viii",
    ],
    "araucania": [
        "araucanía", "araucania", "región de la araucanía",
        "región de araucanía", "región ix", "ix región", "región 9",
        "región n°9", "región nro 9", "novena región", "9na región",
        "9a región", "9m región", "ix",
    ],
    "los_rios": [
        "los ríos", "los rios", "región de los ríos", "región los ríos",
        "región xiv", "xiv región", "región 14", "región n°14",
        "región nro 14", "decimocuarta región", "14va región",
        "14a región", "14m región", "rios", "xiv",
    ],
    "los_lagos": [
        "los lagos", "región de los lagos", "región los lagos",
        "región x", "x región", "región 10", "región n°10",
        "región nro 10", "décima región", "decima región", "10ma región",
        "10a región", "10m región", "lagos", "x",
    ],
    "aysen": [
        "aysén", "aysen", "región de aysén",
        "región de aysén del general carlos ibáñez del campo",
        "aysén del general carlos ibáñez del campo", "región xi",
        "xi región", "región 11", "región n°11", "región nro 11",
        "undécima región", "undecima región", "11va región", "11a región",
        "11m región", "xi",
    ],
    "magallanes": [
        "magallanes", "región de magallanes",
        "región de magallanes y de la antártica chilena",
        "magallanes y de la antártica chilena", "punta arenas",
        "región xii", "xii región", "región 12", "región n°12",
        "región nro 12", "duodécima región", "duodecima región",
        "12va región", "12a región", "12m región", "xii ",
        "antartica", "la antartica", "antartica chilena",
        "antartica chilena y magallanes",
    ],
}

# ─── Precios ───────────────────────────────────────────────────────────────────

PRICE_TERMS: Dict[str, List[str]] = {
    "enc": [
        "volumen a precios del año anterior encadenado",
        "medidas encadenadas de volumen",
        "en términos reales",
        "serie real",
        "volumen real",
        "a precios constantes",
    ],
    "co": [
        "en valores corrientes",
        "a precios actuales",
        "en términos nominales",
        "serie nominal",
        "valor nominal",
    ],
}

# ─── Componentes de inversión / gasto ──────────────────────────────────────────

INVESTMENT_TERMS: Dict[str, List[str]] = {
    "demanda_interna": ["demanda interna", "consumo interno", "gasto interno"],
    "consumo": [
        "consumo", "consumo final", "gasto de consumo", "consumo de los hogares",
        "consumo de las familias", "consumo de hogares", "IPSFL",
        "consumo de hogares e IPSFL",
    ],
    "consumo_gobierno": [
        "consumo del gobierno", "gasto del gobierno", "consumo público",
        "gasto público",
    ],
    "inversion": [
        "inversión", "formación de capital", "formacion bruta de capital", "capex",
    ],
    "inversion_fijo": [
        "inversión fija", "formación bruta de capital fijo", "capex fijo",
    ],
    "existencia": ["existencia", "inventarios", "variación de existencias"],
    "exportacion": [
        "exportación", "exportaciones",
        "exportacion de bienes y servicios",
        "exportaciones de bienes y servicios",
    ],
    "importacion": [
        "importación", "importaciones",
        "importacion de bienes y servicios",
        "importaciones de bienes y servicios",
    ],
    "ahorro_externo": [
        "ahorro externo", "financiamiento externo", "inversión extranjera",
    ],
    "ahorro_interno": [
        "ahorro interno", "financiamiento interno", "inversión nacional",
    ],
}

# ─── Términos de período (latest / previous) ──────────────────────────────────

PERIOD_LATEST_TERMS: List[str] = [
    "última", "ultima", "último", "ultimo", "ultimos", "últimos",
    "ultimas", "últimas", "reciente", "recientemente", "más reciente",
    "mas reciente", "lo más reciente", "lo mas reciente",
    "último dato", "ultimo dato", "último dato disponible",
    "ultimo dato disponible", "ultima cifra", "última cifra",
    "ultima observacion", "última observación", "última publicación",
    "ultima publicacion", "último registro", "ultimo registro",
    "dato más reciente", "dato mas reciente", "cifra más reciente",
    "cifra mas reciente", "valor más reciente", "valor mas reciente",
    "último valor", "ultimo valor", "mas nuevo", "más nuevo",
    "al día", "al dia", "vigente", "actual", "dato vigente",
    "dato actual", "ultima lectura", "última lectura",
]

PERIOD_LATEST_REGEX: List[str] = [
    r"\bultim[oa]s?\b",
    r"\breciente(s)?\b",
    r"\bmas\s+reciente(s)?\b",
    r"\bultimo\s+dato(s)?\b",
    r"\bultimo\s+dato\s+disponible\b",
    r"\bdato\s+disponible\b",
    r"\bultima\s+cifra\b",
    r"\bcifra\s+mas\s+reciente\b",
    r"\bultimo\s+registro\b",
    r"\bultima\s+lectura\b",
    r"\bal\s+dia\b",
]

PERIOD_PREVIOUS_REGEX: List[str] = [
    r"\bmes(es)?\s+pasad[oa]s?\b",
    r"\btrimestre(s)?\s+pasad[oa]s?\b",
    r"\bano(s)?\s+pasad[oa]s?\b",
    r"\bmes(es)?\s+anterior(es)?\b",
    r"\btrimestre(s)?\s+anterior(es)?\b",
    r"\bano(s)?\s+anterior(es)?\b",
    r"\bel\s+pasado\b",
    r"\bla\s+pasada\b",
]

# ─── Mapas temporales auxiliares ───────────────────────────────────────────────

MONTHS: Dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
    "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

MONTH_ALIASES: Dict[str, int] = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11,
    "dic": 12,
}

DECADE_WORDS: Dict[str, int] = {
    "sesenta": 1960, "setenta": 1970, "ochenta": 1980, "noventa": 1990,
}

ROMAN_QUARTERS: Dict[str, int] = {"i": 1, "ii": 2, "iii": 3, "iv": 4}

QUARTERS_START_MONTH: Dict[int, int] = {1: 1, 2: 4, 3: 7, 4: 10}

SPANISH_NUMBER_WORDS: Dict[str, int] = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
    "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20,
}

# ─── Términos genéricos que NO identifican a un indicador concreto ─────────────

GENERIC_INDICATOR_TERMS: List[str] = [
    "economia", "actividad economica", "economico", "economica",
    "economia chilena", "crecimiento económico",
    # Referencias a fuentes de datos (no son indicadores específicos)
    "bde", "bc", "banco central", "bcentral", "banco de españa",
    "banco central de chile", "bcch",
]

# ─── Zona horaria de referencia ────────────────────────────────────────────────

REFERENCE_TIMEZONE = "America/Santiago"
