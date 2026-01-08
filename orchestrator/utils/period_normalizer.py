import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Dict
from difflib import get_close_matches

import pandas as pd
import dateparser


# ---------------------------
# Utilidades de normalización
# ---------------------------

MONTHS = {
    "enero": 1, "ene": 1,
    "febrero": 2, "feb": 2,
    "marzo": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "mayo": 5, "may": 5,
    "junio": 6, "jun": 6,
    "julio": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "septiembre": 9, "sep": 9,
    "octubre": 10, "oct": 10,
    "noviembre": 11, "nov": 11,
    "diciembre": 12, "dic": 12,
}

def fuzzy_match_month(word: str, cutoff: float = 0.75) -> Optional[int]:
    """
    Intenta corregir errores ortográficos en nombres de meses.
    
    Args:
        word: Palabra a buscar (ej: "agousto", "feberero")
        cutoff: Umbral de similitud (0-1), por defecto 0.75
        
    Returns:
        Número del mes (1-12) o None si no hay match
    """
    word = word.lower().strip()
    word = strip_accents(word)
    
    # Primero búsqueda exacta
    if word in MONTHS:
        return MONTHS[word]
    
    # Fuzzy matching sobre nombres completos de meses (no abreviaciones)
    full_month_names = [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre"
    ]
    
    matches = get_close_matches(word, full_month_names, n=1, cutoff=cutoff)
    if matches:
        return MONTHS[matches[0]]
    
    return None

RANGE_QUARTER_MONTHS = {
    1: (1, 3),
    2: (4, 6),
    3: (7, 9),
    4: (10, 12),
}

def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = strip_accents(s)
    # Normalizar typos/variantes en "ultimo/ultima/ultimos/ultimas"
    s = re.sub(r"\bult[io]?m?[oa]s?\b", "ultimo", s)   # ultimo, ultima, ultimos, ultimas, ultmo, ultim
    s = re.sub(r"\bul[t]?im[oa]s?\b", "ultimo", s)
    s = re.sub(r"\bultim\b", "ultimo", s)
    s = re.sub(r"\bultmo\b", "ultimo", s)
    # Normalizar sinónimos de "último": reciente, más reciente -> ultimo
    s = re.sub(r"\b(mas|muy)?\s*r[eai]c[ie][ei]nte?\b", "ultimo", s)
    # Tolera errores en "nuevo": nuevo, nuebo, nueo, nevo
    s = re.sub(r"\b(el|la|lo)?\s*mas\s+n[ue][euo][bv]?[oa]\b", "ultimo", s)
    # Reescrituras útiles - tolerando typos en "año"
    s = re.sub(r"\bpresente\s+a[nñ][oiea]\b", "este ano", s)
    s = re.sub(r"\ba[nñ][oiea]\s+en\s+curso\b", "este ano", s)
    s = re.sub(r"\ba[nñ][oiea]\s+actual\b", "este ano", s)
    # Normalizar "mes de cierre" a "ultimo mes del ano"
    s = re.sub(r"\bmes\s+de\s+cierre\s+del\s+ano\b", "ultimo mes del ano", s)
    return s


# ---------------------------
# Modelo de salida
# ---------------------------

@dataclass
class StandardPeriod:
    granularity: str            # "month" | "quarter" | "year"
    target_date: str            # formato: "YYYY-MM-DD" (month), "YYYY-MM-DD" (quarter, primer día del trimestre), "YYYY" (year)


def month_range(year: int, month: int) -> str:
    """Retorna fecha en formato YYYY-MM-DD."""
    return f"{year:04d}-{month:02d}-01"

def quarter_range(year: int, q: int) -> str:
    """Retorna periodo en formato YYYY-MM-DD (primer día del trimestre)."""
    first_month = {1: 1, 2: 4, 3: 7, 4: 10}.get(q)
    if not first_month:
        first_month = 1
    return f"{year:04d}-{first_month:02d}-01"

def year_range(year: int) -> str:
    """Retorna año en formato YYYY."""
    return f"{year:04d}"


# ---------------------------
# Parsers por reglas
# ---------------------------

def parse_explicit_month(text: str) -> Optional[StandardPeriod]:
    # 1) Formatos numéricos: 08/2024, 08-2024, 2024-08
    m = re.search(r"\b(0?[1-9]|1[0-2])\s*[\/\-]\s*((19|20)\d{2})\b", text)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        target = month_range(year, month)
        return StandardPeriod("month", target)

    m = re.search(r"\b((19|20)\d{2})\s*[\/\-]\s*(0?[1-9]|1[0-2])\b", text)
    if m:
        year = int(m.group(1))
        month = int(m.group(3))
        target = month_range(year, month)
        return StandardPeriod("month", target)

    # 2) Mes en texto: "agosto 2024", "ago 2024", "mes de abril del 2021"
    m = re.search(
        r"\b(" + "|".join(sorted(MONTHS.keys(), key=len, reverse=True)) + r")\b"
        r"(?:\s+de|\s+del|\s+)?\s*((19|20)\d{2})\b",
        text
    )
    if m:
        mon_str = m.group(1)
        year = int(m.group(2))
        month = MONTHS[mon_str]
        target = month_range(year, month)
        return StandardPeriod("month", target)

    # 2b) Mes con "del ano" explícito entre mes y año: "marzo del ano 2023"
    # Nota: el texto ya viene sin acentos ("año" -> "ano") desde normalize_text/strip_accents
    m = re.search(
        r"\b(" + "|".join(sorted(MONTHS.keys(), key=len, reverse=True)) + r")\b" 
        r"(?:\s+de|\s+del)?\s+ano\s+((19|20)\d{2})\b",
        text
    )
    if m:
        mon_str = m.group(1)
        year = int(m.group(2))
        month = MONTHS[mon_str]
        target = month_range(year, month)
        return StandardPeriod("month", target)

    # 3) Fuzzy matching para meses con errores ortográficos: "agousto 2024", "del feberero del ano 2023"
    # Patrón 1: "del PALABRA del ano YYYY"
    m = re.search(r"\bdel\s+(\w{4,12})\s+del\s+ano\s+((19|20)\d{2})\b", text)
    if m:
        potential_month = m.group(1)
        year = int(m.group(2))
        month_num = fuzzy_match_month(potential_month, cutoff=0.7)
        if month_num:
            target = month_range(year, month_num)
            return StandardPeriod("month", target)
    
    # Patrón 2: "PALABRA YYYY" o "de PALABRA del YYYY"
    m = re.search(r"(?:de|del)?\s*(\w{4,10})\s+(?:de|del)?\s*((19|20)\d{2})\b", text)
    if m:
        potential_month = m.group(1)
        # Evitar falsos positivos con palabras comunes
        if potential_month not in ['ano', 'imacec', 'valor', 'dato', 'datos', 'mes', 'meses']:
            year = int(m.group(2))
            month_num = fuzzy_match_month(potential_month, cutoff=0.7)
            if month_num:
                target = month_range(year, month_num)
                return StandardPeriod("month", target)

    return None


def parse_explicit_quarter(text: str, base_date: Optional[date] = None) -> Optional[StandardPeriod]:
    """
    Detecta periodos trimestrales explícitos en español, incluyendo variantes y errores comunes:
    - 1T 2025, 1Q 2025, 1er trimestre 2025, I trimestre 2025, primer trimestre 2025, etc.
    - Soporta variantes y typos: 'trimstre', 'trimestr', etc.
    """
    ord_map = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4, "1er": 1, "2do": 2, "3er": 3, "4to": 4, "1ro": 1, "2do": 2, "3ro": 3, "4to": 4}
    roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    base = base_date or date.today()

    # Fuzzy matching para variantes de 'trimestre' (typos comunes)
    fuzzy_trimestre = ["trimestre", "trimstre", "trimetre", "trimestr", "trimesttre", "trimestte"]
    fuzzy_ordinals = ["primer", "segundo", "tercer", "cuarto", "1er", "2do", "3er", "4to", "1ro", "2do", "3ro", "4to"]
    for ord in fuzzy_ordinals:
        for tri in fuzzy_trimestre:
            pat = fr"\b({ord})\s+{tri}(?:\s+(?:de|del))?\s*((19|20)\d{{2}})\b"
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                q = ord_map.get(m.group(1).lower(), None)
                year = int(m.group(2))
                if q and year:
                    target = quarter_range(year, q)
                    return StandardPeriod("quarter", target)

    def roman_extractor(m):
        roman = m.group(1)
        roman_clean = strip_accents(roman).lower()
        q = roman_map.get(roman_clean)
        return (q, int(m.group(2)))

    def roman_extractor_no_year(m):
        roman = m.group(1)
        roman_clean = strip_accents(roman).lower()
        q = roman_map.get(roman_clean)
        return (q, None)

    patterns = [
        (r"\b([ivx]{1,3})\s*trimestre(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", roman_extractor),
        (r"\b([ivx]{1,3})\s*trimestre\b", roman_extractor_no_year),
        (r"\b([1-4])\s*[TtQq]\s*[- ]?((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
        (r"\b([1-4])[TtQq]((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
        (r"\b([1-4])\s*[TtQq]\b", lambda m: (int(m.group(1)), None)),
        (r"\b([1-4])[TtQq]\b", lambda m: (int(m.group(1)), None)),
        (r"\b[TtQq]\s*([1-4])\s*[- ]?((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
        (r"\b[TtQq]([1-4])((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
        (r"\b[TtQq]\s*([1-4])\b", lambda m: (int(m.group(1)), None)),
        (r"\b[TtQq]([1-4])\b", lambda m: (int(m.group(1)), None)),
        (r"\b([1-4])(er|ro|do|to)?\s*trimestre(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(3)))),
        (r"\b([1-4])(er|ro|do|to)?\s*trimestre\b", lambda m: (int(m.group(1)), None)),
        (r"\b(primer|segundo|tercer|cuarto|1er|2do|3er|4to|1ro|2do|3ro|4to)\s+trimestre(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", lambda m: (ord_map.get(m.group(1).lower()), int(m.group(2)))),
        (r"\b(primer|segundo|tercer|cuarto|1er|2do|3er|4to|1ro|2do|3ro|4to)\s+trimestre\b", lambda m: (ord_map.get(m.group(1).lower()), None)),
        (r"\btrimestre\s*([1-4])(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
        (r"\b([1-4])\s*trimestre(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
    ]

    for pat, extractor in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            q, year = extractor(m)
            if q is None:
                continue
            if year or year is None:
                if year is None:
                    year = base.year
                    confidence = "medium"
                target = quarter_range(year, q)
                return StandardPeriod("quarter", target)

    # 4) "primer/segundo/tercer/cuarto trimestre de/del 2025" (palabra, sin variantes)
    m = re.search(r"\b(primer|segundo|tercer|cuarto)\s+trimestre(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", text)
    if m:
        ord_map = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4}
        q = ord_map[m.group(1)]
        year = int(m.group(2))
        target = quarter_range(year, q)
        return StandardPeriod("quarter", target)

    # 4) Rangos por meses: "enero-marzo 2025", "abril a junio 2025"
    m = re.search(
        r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r"\s*(?:-|a)\s*"
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
        r"\s*((19|20)\d{2})\b",
        text
    )
    if m:
        m1 = MONTHS[m.group(1)]
        m2 = MONTHS[m.group(2)]
        year = int(m.group(3))
        # Si coincide exactamente con un trimestre canónico, lo expresamos como quarter
        for q, (qs, qe) in RANGE_QUARTER_MONTHS.items():
            if (m1, m2) == (qs, qe):
                target = quarter_range(year, q)
                return StandardPeriod("quarter", target)
        # Si no coincide, podrías optar por devolver un rango mensual genérico
        # pero aquí dejamos que otro nivel lo resuelva.
    return None


def parse_explicit_year(text: str) -> Optional[StandardPeriod]:
    m = re.search(r"\b((19|20)\d{2})\b", text)
    # Ojo: esto es muy laxo, por eso solo úsalo si NO detectaste mes/trimestre antes.
    if m:
        year = int(m.group(1))
        target = year_range(year)
        return StandardPeriod("year", target)
    return None


def parse_relative_periods(text: str, base: date) -> Optional[StandardPeriod]:
    # IMPORTANTE: Reglas MÁS ESPECÍFICAS primero
    
    # "primer trimestre del presente año" (antes de capturar solo "este ano")
    m = re.search(r"\b(primer|segundo|tercer|cuarto)\s+trimestre\s+del\s+(este ano|presente ano|ano actual|ano en curso)\b", text)
    if m:
        ord_map = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4}
        q = ord_map[m.group(1)]
        year = base.year
        target = quarter_range(year, q)
        return StandardPeriod("quarter", target)

    # "ultimo mes del ano" -> Diciembre del año base
    m = re.search(r"\bultimo\s+mes\s+del\s+ano\b", text)
    if m:
        year = base.year
        target = month_range(year, 12)
        return StandardPeriod("month", target)

    # "ultimo trimestre del ano" -> Q4 del año base
    m = re.search(r"\bultimo\s+trimestre\s+del\s+ano\b", text)
    if m:
        year = base.year
        target = quarter_range(year, 4)
        return StandardPeriod("quarter", target)

    # "ultimo ano" -> año base
    m = re.search(r"\bultimo\s+ano\b", text)
    if m:
        year = base.year
        target = year_range(year)
        return StandardPeriod("year", target)

    # "ultimo mes" -> mes anterior completo al base
    m = re.search(r"\bultimo\s+mes\b", text)
    if m:
        prev = pd.Period(base, freq="M") - 1
        year = prev.start_time.year
        month = prev.start_time.month
        target = month_range(year, month)
        return StandardPeriod("month", target)

    # "ultimo trimestre" -> trimestre anterior al base
    m = re.search(r"\bultimo\s+trimestre\b", text)
    if m:
        q_current = (base.month - 1) // 3 + 1
        year = base.year
        q = q_current - 1
        if q == 0:
            q = 4
            year -= 1
        target = quarter_range(year, q)
        return StandardPeriod("quarter", target)
    
    # "primer mes del año" (antes de capturar solo "este ano")
    m = re.search(r"\b(primer|segundo|tercer|cuarto)\s+mes\s+del\s+ano\b", text)
    if m:
        ord_to_month = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4}
        # Mapas para ordinales y romanos
        ord_map = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4, "1er": 1, "2do": 2, "3er": 3, "4to": 4, "1ro": 1, "2do": 2, "3ro": 3, "4to": 4}
        roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4}

        # 1) Notaciones cortas y variantes: 1T, 1Q, 1er trimestre, I trimestre, etc.
        # Orden de patrones: primero los más específicos y con año explícito
        patterns = [
            (r"\b([1-4])\s*[TQ]\s*[- ]?((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
            (r"\b([1-4])\s*[TQ]\b", lambda m: (int(m.group(1)), None)),
            (r"\b[TQ]\s*([1-4])\s*[- ]?((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
            (r"\b[TQ]\s*([1-4])\b", lambda m: (int(m.group(1)), None)),
            (r"\b([1-4])(er|ro|do|to)?\s*trimestre(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(3)))),
            (r"\b([1-4])(er|ro|do|to)?\s*trimestre\b", lambda m: (int(m.group(1)), None)),
            (r"\b(primer|segundo|tercer|cuarto|1er|2do|3er|4to|1ro|2do|3ro|4to)\s+trimestre(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", lambda m: (ord_map.get(m.group(1).lower()), int(m.group(2)))),
            (r"\b(primer|segundo|tercer|cuarto|1er|2do|3er|4to|1ro|2do|3ro|4to)\s+trimestre\b", lambda m: (ord_map.get(m.group(1).lower()), None)),
            (r"\b([IVX]{1,3})\s*trimestre(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", lambda m: (roman_map.get(m.group(1).upper()), int(m.group(2)))),
            (r"\b([IVX]{1,3})\s*trimestre\b", lambda m: (roman_map.get(m.group(1).upper()), None)),
            (r"\btrimestre\s*([1-4])(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
            (r"\b([1-4])\s*trimestre(?:\s+(?:de|del))?\s*((19|20)\d{2})\b", lambda m: (int(m.group(1)), int(m.group(2)))),
        ]

        for pat, extractor in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                q, year = extractor(m)
                if q and (year or year is None):
                    if year is None:
                        year = base.year
                    target = quarter_range(year, q)
                    return StandardPeriod("quarter", target)

        # Fuzzy matching para "trimestre" y ordinales (tolerancia a errores ortográficos)
        fuzzy_ordinals = ["primer", "segundo", "tercer", "cuarto", "1er", "2do", "3er", "4to", "1ro", "2do", "3ro", "4to"]
        fuzzy_trimestre = ["trimestre", "trimstre", "trimetre", "trimestr", "trimesttre", "trimestte"]
        for ord in fuzzy_ordinals:
            for tri in fuzzy_trimestre:
                pat = fr"\b({ord})\s+{tri}(?:\s+(?:de|del))?\s*((19|20)\d{{2}})\b"
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    q = ord_map.get(m.group(1).lower(), None)
                    year = int(m.group(2))
                    if q and year:
                        target = quarter_range(year, q)
                        return StandardPeriod("quarter", target)

        # 4) Rangos por meses: "enero-marzo 2025", "abril a junio 2025"
        m = re.search(
            r"\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
            r"\s*(?:-|a)\s*"
            r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
            r"\s*((19|20)\d{2})\b",
            text
        )
        if m:
            m1 = MONTHS[m.group(1)]
            m2 = MONTHS[m.group(2)]
            year = int(m.group(3))
            # Si coincide exactamente con un trimestre canónico, lo expresamos como quarter
            for q, (qs, qe) in RANGE_QUARTER_MONTHS.items():
                if (m1, m2) == (qs, qe):
                    target = quarter_range(year, q)
                    return StandardPeriod("quarter", target)
            # Si no coincide, podrías optar por devolver un rango mensual genérico
            # pero aquí dejamos que otro nivel lo resuelva.
        return None
    if re.search(r"\b(ultimo reporte del ano|comunicado del ano pasado)\b", text):
        year = base.year - 1 if "pasado" in text else base.year
        target = year_range(year)
        return StandardPeriod("year", target)

    return None


# ---------------------------
# Fallback con dateparser
# ---------------------------

def parse_with_dateparser(text: str, base: date) -> Optional[StandardPeriod]:
    settings = {
        "RELATIVE_BASE": datetime.combine(base, datetime.min.time()),
        "PREFER_DAY_OF_MONTH": "first",
        "PREFER_DATES_FROM": "past",
    }

    dt = dateparser.parse(text, languages=["es"], settings=settings)
    if not dt:
        return None

    # Heurística: si el texto menciona explícitamente un año y no mes,
    # lo tratamos como año; si menciona un mes en palabras, como mes.
    t = text
    has_month_word = any(re.search(fr"\b{k}\b", t) for k in MONTHS.keys())
    has_year = re.search(r"\b(20\d{2})\b", t) is not None

    if has_month_word and has_year:
        target = month_range(dt.year, dt.month)
        return StandardPeriod("month", target)

    if has_year and not has_month_word:
        target = year_range(dt.year)
        return StandardPeriod("year", target)

    # Si no está claro, asumimos mes del dt.
    target = month_range(dt.year, dt.month)
    return StandardPeriod("month", target)


# ---------------------------
# Función principal
# ---------------------------

def standardize_imacec_time_ref(text: str, base_date: Optional[date] = None) -> Optional[Dict]:
    """
    Devuelve un dict estandarizado con:
      granularity, start_date, end_date, period_key, confidence, source
    """
    base = base_date or date.today()
    t_norm = normalize_text(text)
    t_noacc = strip_accents(t_norm)

    # 1) Relativos primero (más específicos: "primer trimestre del presente año")
    p = parse_relative_periods(t_noacc, base)
    if p:
        return p.__dict__

    # 2) Reglas explícitas: quarter
    p = parse_explicit_quarter(t_noacc, base)
    if p:
        return p.__dict__

    # 3) Reglas explícitas: mes
    p = parse_explicit_month(t_noacc)
    if p:
        return p.__dict__

    # 4) Año explícito (solo si no se detectó algo más fino)
    p = parse_explicit_year(t_noacc)
    if p:
        return p.__dict__

    # 5) NLP fallback
    p = parse_with_dateparser(t_noacc, base)
    if p:
        return p.__dict__

    return None
