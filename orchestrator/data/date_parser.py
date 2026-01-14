import re
from typing import Optional, Tuple
from datetime import datetime, date
from calendar import monthrange

import dateparser

SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
}


def _first_day_of_month(year: int, month: int) -> date:
    return date(year, month, 1)


def _quarter_start(year: int, q: int) -> date:
    month = (q - 1) * 3 + 1
    return date(year, month, 1)


def _format_ddmmyyyy(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def parse_point_date(query: str, frequency: str, default_year: Optional[int] = None) -> Optional[str]:
    q = query.lower()
    # Try quarter
    m = re.search(r"(q|t)([1-4])\s*(\d{4})", q)
    if m:
        qn = int(m.group(2))
        yr = int(m.group(3))
        return _format_ddmmyyyy(_quarter_start(yr, qn))

    # "primer trimestre 2024"
    m = re.search(r"(primer|segundo|tercer|cuarto)\s+trimestre\s+(\d{4})", q)
    if m:
        qmap = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4}
        qn = qmap[m.group(1)]
        yr = int(m.group(2))
        return _format_ddmmyyyy(_quarter_start(yr, qn))

    # Monthly: try month name + year
    for name, mn in SPANISH_MONTHS.items():
        # Accept "noviembre 2025", "noviembre de 2025", "noviembre del 2025"
        mm = re.search(name + r"\s+(?:de\s+|del\s+)?(\d{4})", q)
        if mm:
            yr = int(mm.group(1))
            return _format_ddmmyyyy(_first_day_of_month(yr, mn))
    
    # Monthly: try month name WITHOUT year (use default_year if provided)
    if default_year is not None:
        for name, mn in SPANISH_MONTHS.items():
            if name in q:
                return _format_ddmmyyyy(_first_day_of_month(default_year, mn))

    # Annual: explicit year only (or year fallback for any frequency)
    m = re.search(r"\b(\d{4})\b", q)
    if m:
        yr = int(m.group(1))
        if frequency == "a":
            return _format_ddmmyyyy(date(yr, 1, 1))
        # For monthly/quarterly, return first observation of the year
        return _format_ddmmyyyy(date(yr, 1, 1))

    # Fallback to dateparser
    dt = dateparser.parse(query, languages=["es"])  # type: ignore
    if dt:
        # Normalize by frequency using first day of the period
        if frequency == "m":
            d = _first_day_of_month(dt.year, dt.month)
        elif frequency == "q":
            qn = (dt.month - 1) // 3 + 1
            d = _quarter_start(dt.year, qn)
        else:
            d = date(dt.year, 1, 1)
        return _format_ddmmyyyy(d)

    return None


def parse_range(query: str, frequency: str) -> Optional[Tuple[str, str]]:
    q = query.lower()
    # Common patterns: "desde X hasta Y", "entre X y Y", "de X a Y"
    patterns = [
        r"desde\s+(.*?)\s+hasta\s+(.*)",
        r"entre\s+(.*?)\s+y\s+(.*)",
        r"de\s+(.*?)\s+a\s+(.*)",
    ]
    for p in patterns:
        m = re.search(p, q)
        if m:
            start_raw = m.group(1)
            end_raw = m.group(2)

            # Extract year from end_raw or start_raw
            end_year_match = re.search(r"\b(\d{4})\b", end_raw)
            start_year_match = re.search(r"\b(\d{4})\b", start_raw)
            inferred_year = None
            
            if end_year_match:
                inferred_year = int(end_year_match.group(1))
            elif start_year_match:
                inferred_year = int(start_year_match.group(1))

            # Parse start and end with inferred year
            start = parse_point_date(start_raw, frequency, default_year=inferred_year)
            end = parse_point_date(end_raw, frequency, default_year=inferred_year)
            
            if not end:
                # Try splitting end_raw by common stop words
                for stop in [" variaci", " cambio", " diferencia", " calculo", " cálculo"]:
                    if stop in end_raw:
                        end_raw = end_raw.split(stop)[0].strip()
                        end = parse_point_date(end_raw, frequency, default_year=inferred_year)
                        if end:
                            break
            if start and end:
                return (start, end)
    
    # Year-only: if a single year is mentioned, return full-year range per frequency
    year_match = re.search(r"\b(\d{4})\b", q)
    if year_match:
        yr = int(year_match.group(1))
        if frequency == "m":
            start = _format_ddmmyyyy(_first_day_of_month(yr, 1))
            end = _format_ddmmyyyy(_first_day_of_month(yr, 12))
        elif frequency == "q":
            start = _format_ddmmyyyy(_quarter_start(yr, 1))
            end = _format_ddmmyyyy(_quarter_start(yr, 4))
        else:
            start = _format_ddmmyyyy(date(yr, 1, 1))
            end = start
        return (start, end)
    
    # Fallback: find two dates via splitting numbers/words
    parts = re.split(r"\s+(al|a|y)\s+", q)
    if len(parts) >= 2:
        start = parse_point_date(parts[0], frequency)
        end = parse_point_date(parts[-1], frequency)
        if start and end:
            return (start, end)
    
    return None
