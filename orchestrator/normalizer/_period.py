"""
Resolución y formateo de períodos temporales para el normalizador NER.

Responsabilidades:
- Parseo de fechas explícitas (meses, trimestres, años, décadas)
- Resolución de referencias relativas ("hace 3 meses", "año pasado")
- Resolución de referencias implícitas ("última cifra", "dato vigente")
- Inferencia de granularidad temporal (mensual/trimestral/anual)
- Formateo final a rangos ``[YYYY-MM-DD, YYYY-MM-DD]``

Convenciones de salida:
- Siempre se retornan rangos de 2 elementos ``[inicio, fin]``.
- En contexto trimestral los límites se ajustan al trimestre completo.
- Rangos invertidos se reordenan automáticamente.
"""

import re
from datetime import datetime, timedelta
from calendar import monthrange
from typing import Dict, List, Optional, Tuple

from orchestrator.normalizer._vocab import (
    MONTHS,
    MONTH_ALIASES,
    DECADE_WORDS,
    ROMAN_QUARTERS,
    QUARTERS_START_MONTH,
    SPANISH_NUMBER_WORDS,
    PERIOD_LATEST_TERMS,
    PERIOD_LATEST_REGEX,
    PERIOD_PREVIOUS_REGEX,
    REFERENCE_TIMEZONE,
)
from orchestrator.normalizer._text import normalize_text, fuzzy_match

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers de fecha
# ═══════════════════════════════════════════════════════════════════════════════

def reference_now() -> datetime:
    """Fecha/hora actual en zona horaria de referencia (America/Santiago)."""
    if ZoneInfo is None:
        return datetime.now()
    try:
        return datetime.now(ZoneInfo(REFERENCE_TIMEZONE)).replace(tzinfo=None)
    except Exception:
        return datetime.now()


def parse_yyyymmdd(text: str) -> Optional[datetime]:
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except Exception:
        return None


def _add_months(base: datetime, delta: int) -> datetime:
    idx = base.year * 12 + (base.month - 1) + delta
    y, m = divmod(idx, 12)
    m += 1
    return datetime(y, m, min(base.day, monthrange(y, m)[1]))


# ─── Formateo ──────────────────────────────────────────────────────────────────

def quarter_start_month(month: int) -> int:
    """Mes de inicio del trimestre al que pertenece *month* (1-indexed)."""
    return ((month - 1) // 3) * 3 + 1


def fmt_month_start(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}-01"


def fmt_month_end(dt: datetime) -> str:
    nxt = datetime(dt.year + (1 if dt.month == 12 else 0),
                   (dt.month % 12) + 1, 1)
    return (nxt - timedelta(days=1)).strftime("%Y-%m-%d")


def fmt_quarter_start(dt: datetime) -> str:
    qm = quarter_start_month(dt.month)
    return f"{dt.year:04d}-{qm:02d}-01"


def fmt_quarter_end(dt: datetime) -> str:
    qm = quarter_start_month(dt.month) + 2
    return fmt_month_end(datetime(dt.year, qm, 1))


def fmt_year_start(dt: datetime) -> str:
    return f"{dt.year:04d}-01-01"


# ─── Anclas temporales ─────────────────────────────────────────────────────────

def prev_month_anchor(dt: datetime) -> datetime:
    return datetime(dt.year - 1, 12, 1) if dt.month == 1 else datetime(dt.year, dt.month - 1, 1)


def prev_quarter_anchor(dt: datetime) -> datetime:
    qs = quarter_start_month(dt.month)
    return datetime(dt.year - 1, 10, 1) if qs == 1 else datetime(dt.year, qs - 3, 1)


def prev_year_anchor(dt: datetime) -> datetime:
    return datetime(dt.year - 1, 1, 1)


def _shift_anchor(base: datetime, granularity: str, steps: int) -> datetime:
    if granularity == "m":
        return _add_months(datetime(base.year, base.month, 1), -steps)
    if granularity == "q":
        return _add_months(
            datetime(base.year, quarter_start_month(base.month), 1),
            -(steps * 3),
        )
    return datetime(base.year - steps, 1, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Detección de referencias temporales en texto libre
# ═══════════════════════════════════════════════════════════════════════════════

def has_explicit_year(text: str) -> bool:
    return bool(re.search(r"\b(?:19|20)\d{2}\b", normalize_text(text)))


def contains_latest_ref(text: str) -> bool:
    """¿El texto contiene referencia a 'último dato', 'más reciente', etc.?"""
    n = normalize_text(text)
    if not n:
        return False
    if any(t in n for t in PERIOD_LATEST_TERMS):
        return True
    return any(re.search(p, n) for p in PERIOD_LATEST_REGEX)


def contains_previous_ref(text: str) -> bool:
    n = normalize_text(text)
    return bool(n) and any(re.search(p, n) for p in PERIOD_PREVIOUS_REGEX)


def contains_current_ref(text: str) -> bool:
    n = normalize_text(text)
    return bool(n) and bool(
        re.search(r"\beste\b|\bactual\b|\ben\s+curso\b|\bcurso\b", n)
    )


# ─── Tokens numéricos ─────────────────────────────────────────────────────────

def _parse_number_token(token: Optional[str]) -> Optional[int]:
    n = normalize_text(str(token or "")).strip()
    if not n:
        return None
    if n.isdigit():
        return int(n)
    return SPANISH_NUMBER_WORDS.get(n)


# ─── Trimestres ────────────────────────────────────────────────────────────────

def _is_trimester_like(token: str) -> bool:
    """Detecta variantes de 'trimestre' con tolerancia a typos."""
    if not token:
        return False
    n = normalize_text(token)
    if n.startswith("trimestre") or n in {"trim", "trimestr", "trimestral", "quarter", "quarterly"}:
        return True
    return fuzzy_match(n, ["trimestre", "trimestres", "trimestral", "trim"], threshold=0.7) is not None


def has_quarter_ref(text: str) -> bool:
    n = normalize_text(text)
    if any(_is_trimester_like(t) for t in re.findall(r"[a-záéíóúüñ0-9]+", n)):
        return True
    return re.search(r"\b[tq]\s*[1-4]\b|\b[tq][1-4]\b|\b[1-4]\s*(?:t|q)\b", n) is not None


# ─── Extracción de fechas desde texto libre ─────────────────────────────────

def extract_quarter_dates(text: str, reference_year: Optional[int] = None) -> List[str]:
    """Extrae fechas ``YYYY-MM-DD`` desde menciones de trimestres en *text*."""
    tokens = re.findall(r"[a-záéíóúüñ0-9]+", normalize_text(text))
    if not tokens:
        return []

    _ordinal_word = {
        "primer": 1, "primero": 1, "segundo": 2,
        "tercer": 3, "tercero": 3, "cuarto": 4,
    }

    q_tokens: List[Tuple[int, int]] = []
    y_tokens: List[Tuple[int, int]] = []

    for i, tok in enumerate(tokens):
        if re.fullmatch(r"(?:19|20)\d{2}", tok):
            y_tokens.append((i, int(tok)))
            continue

        m = re.fullmatch(r"[tq]([1-4])", tok)
        if m:
            q_tokens.append((i, int(m.group(1))))
            continue
        m = re.fullmatch(r"([1-4])[tq]", tok)
        if m:
            q_tokens.append((i, int(m.group(1))))
            continue
        m = re.fullmatch(r"([1-4])(?:er|ro|do|to)?", tok)
        if m and i + 1 < len(tokens) and _is_trimester_like(tokens[i + 1]):
            q_tokens.append((i, int(m.group(1))))
            continue
        rq = ROMAN_QUARTERS.get(tok)
        if rq and i + 1 < len(tokens) and _is_trimester_like(tokens[i + 1]):
            q_tokens.append((i, rq))
            continue
        if tok in _ordinal_word and i + 1 < len(tokens) and _is_trimester_like(tokens[i + 1]):
            q_tokens.append((i, _ordinal_word[tok]))
            continue
        if tok in {"1", "2", "3", "4"} and i + 1 < len(tokens) and _is_trimester_like(tokens[i + 1]):
            q_tokens.append((i, int(tok)))

    # ── Second pass: expand coordinated ordinal lists ──────────────────
    # Handles patterns like "primer y tercer trimestre", "1, 2 y 3 trimestre"
    # by scanning backward from each detected q_token through connectors
    # ("y", ",", "e") and noise articles ("el", "la", "del", etc.).
    # Commas are stripped by the tokenizer, so adjacent ordinals are also handled.
    _noise = {"el", "la", "los", "las", "del", "de", "al", "entre"}
    _connectors = {"y", ",", "e"}
    expanded: List[Tuple[int, int]] = []
    q_positions = {pos for pos, _ in q_tokens}

    def _try_parse_q(tok: str) -> Optional[int]:
        q = _ordinal_word.get(tok)
        if q is not None:
            return q
        pm = re.fullmatch(r"([1-4])(?:er|ro|do|to)?", tok)
        if pm:
            return int(pm.group(1))
        if tok in {"1", "2", "3", "4"}:
            return int(tok)
        return ROMAN_QUARTERS.get(tok)

    for qi, _qv in q_tokens:
        scan = qi - 1
        while scan >= 0:
            tok_s = tokens[scan]
            if tok_s in _noise:
                scan -= 1
                continue
            # Adjacent ordinal (comma stripped by tokenizer)
            adj_q = _try_parse_q(tok_s)
            if adj_q is not None and scan not in q_positions:
                expanded.append((scan, adj_q))
                q_positions.add(scan)
                scan -= 1
                continue
            if tok_s not in _connectors:
                break
            # Find the ordinal before the connector (skip noise)
            prev_scan = scan - 1
            while prev_scan >= 0 and tokens[prev_scan] in _noise:
                prev_scan -= 1
            if prev_scan < 0:
                break
            prev_tok = tokens[prev_scan]
            prev_q: Optional[int] = _try_parse_q(prev_tok)
            if prev_q is not None and prev_scan not in q_positions:
                expanded.append((prev_scan, prev_q))
                q_positions.add(prev_scan)
                scan = prev_scan - 1
            else:
                break

    q_tokens.extend(expanded)

    if not q_tokens:
        return []

    now = reference_now()
    year_fallback = reference_year if reference_year is not None else now.year
    current_quarter = ((now.month - 1) // 3) + 1
    dates: List[str] = []
    for qi, q in q_tokens:
        ry = next((y for idx, y in y_tokens if idx > qi), None)
        ly = next((y for idx, y in reversed(y_tokens) if idx < qi), None)
        y = ry if ry is not None else (ly if ly is not None else year_fallback)
        # Si el trimestre aún no ha ocurrido este año, asumir el año anterior
        if y == now.year and ry is None and ly is None and q > current_quarter:
            y -= 1
        d = f"{y:04d}-{QUARTERS_START_MONTH[q]:02d}-01"
        if d not in dates:
            dates.append(d)
    return dates


def extract_month_dates(text: str, reference_year: Optional[int] = None) -> List[str]:
    """Extrae fechas ``YYYY-MM-DD`` desde menciones de meses en *text*."""
    tokens = re.findall(r"[a-záéíóúüñ0-9]+", normalize_text(text))
    if not tokens:
        return []

    all_month_names = list(MONTHS.keys()) + list(MONTH_ALIASES.keys())
    m_tokens: List[Tuple[int, int]] = []
    y_tokens: List[Tuple[int, int]] = []

    for i, tok in enumerate(tokens):
        if re.fullmatch(r"(?:19|20)\d{2}", tok):
            y_tokens.append((i, int(tok)))
            continue
        if tok in MONTHS:
            m_tokens.append((i, MONTHS[tok]))
            continue
        if tok in MONTH_ALIASES:
            m_tokens.append((i, MONTH_ALIASES[tok]))
            continue
        if len(tok) >= 4:
            matched = fuzzy_match(tok, all_month_names, threshold=0.82)
            if matched:
                num = MONTHS.get(matched) or MONTH_ALIASES.get(matched)
                if num is not None:
                    m_tokens.append((i, num))

    if not m_tokens:
        return []

    now = reference_now()
    year_fallback = reference_year if reference_year is not None else now.year
    if not y_tokens:
        dates: List[str] = []
        for _, mn in m_tokens:
            y = year_fallback
            # Si el mes aún no ha ocurrido este año, asumir el año anterior
            if y == now.year and mn > now.month:
                y -= 1
            d = f"{y:04d}-{mn:02d}-01"
            if d not in dates:
                dates.append(d)
        return dates

    dates = []
    for mi, mn in m_tokens:
        ry = next((y for idx, y in y_tokens if idx > mi), None)
        ly = next((y for idx, y in reversed(y_tokens) if idx < mi), None)
        y = ry if ry is not None else ly
        if y is None:
            continue
        d = f"{y:04d}-{mn:02d}-01"
        if d not in dates:
            dates.append(d)
    return dates


def extract_year_dates(text: str) -> List[str]:
    """Extrae años explícitos como fechas ``YYYY-01-01``."""
    matches = re.findall(r"\b((?:19|20)\d{2})\b", normalize_text(text))
    seen: List[str] = []
    for yt in matches:
        d = f"{int(yt):04d}-01-01"
        if d not in seen:
            seen.append(d)
    return seen


def extract_decade_range(text: str) -> Optional[Tuple[int, int]]:
    """Extrae rango de década si se menciona (ej. 'los 90s' → (1990, 1999))."""
    n = normalize_text(text)
    if not n:
        return None

    def _norm_decade(raw: str) -> Optional[int]:
        v = normalize_text(raw).strip().replace("s", "")
        if not v:
            return None
        if re.fullmatch(r"(?:19|20)\d0", v):
            return int(v)
        if re.fullmatch(r"\d{2}", v):
            short = int(v)
            return (1900 if short >= 30 else 2000) + short
        return None

    for pat in [
        r"\b(?:decada|decadas|anos|anios|años)\s+(?:del?|de|los)?\s*((?:19|20)\d0|\d{2})s?\b",
        r"\blos\s+((?:19|20)\d0|\d{2})s\b",
    ]:
        m = re.search(pat, n)
        if m:
            ds = _norm_decade(m.group(1))
            if ds is not None:
                return ds, ds + 9
    for word, ds in DECADE_WORDS.items():
        if re.search(rf"\b(?:decada|decadas|anos|anios|años)\s+(?:del?|de|los)?\s*{word}\b", n):
            return ds, ds + 9
    return None


# ─── Clasificación de tipo de período ──────────────────────────────────────────

def is_year_only_ref(text: str) -> bool:
    """``True`` si el texto refiere a un año sin mes ni trimestre explícito."""
    if not text:
        return False
    n = normalize_text(text)
    if not re.search(r"\b(?:19|20)\d{2}\b", n):
        return False
    return not any(m in n for m in MONTHS) and not has_quarter_ref(text)


# ─── Resolución de referencias relativas ──────────────────────────────────────

def _extract_relative_shift(text: str) -> Optional[Tuple[str, int]]:
    """Detecta 'hace N meses/trimestres/años' → (granularidad, cantidad)."""
    n = normalize_text(text)
    if not n:
        return None
    m = re.search(
        r"\bhace\s+([a-z0-9]+)\s+(mes(?:es)?|trimestre(?:s)?|(?:ano|anio)s?)\b(?:\s+atras)?",
        n,
    )
    if not m:
        return None
    amount = _parse_number_token(m.group(1))
    if amount is None or amount <= 0:
        return None
    unit = m.group(2)
    if unit.startswith("mes"):
        return "m", amount
    if unit.startswith("trimestre"):
        return "q", amount
    return "a", amount


def _extract_relative_year_offset(text: str) -> Optional[int]:
    n = normalize_text(text)
    if not n:
        return None
    if re.search(r"\b(?:ano|anio)\s+antepasad[oa]s?\b", n):
        return 2
    if re.search(r"\b(?:ano|anio)\s+antes\s+pasad[oa]s?\b", n):
        return 2
    if re.search(r"\b(?:ano|anio)\s+pasad[oa]s?\b", n):
        return 1
    m = re.search(r"\bhace\s+([a-z0-9]+)\s+(?:ano|anio)s?\b(?:\s+atras)?", n)
    if m:
        return _parse_number_token(m.group(1))
    return None


def _has_explicit_relative_year_anchor(text: str) -> bool:
    n = normalize_text(text)
    if not n or _extract_relative_year_offset(n) is None:
        return False
    has_m = any(mn in n for mn in MONTHS) or bool(re.search(r"\bmes(?:es)?\b", n))
    return has_m or has_quarter_ref(n) or bool(re.search(r"\b(?:ano|anio)s?\b", n))


def extract_relative_anchored_dates(
    text: str,
    ref_now: Optional[datetime] = None,
) -> List[str]:
    """Resuelve referencias temporales relativas a fechas ``YYYY-MM-DD``."""
    n = normalize_text(text)
    if not n:
        return []
    now = ref_now or reference_now()

    shift = _extract_relative_shift(n)
    if shift:
        g, amount = shift
        anchor = _shift_anchor(now, g, amount)
        if g == "q":
            return [fmt_quarter_start(anchor)]
        if g == "a":
            return [fmt_year_start(anchor)]
        return [fmt_month_start(anchor)]

    offset = _extract_relative_year_offset(n)
    if offset is None:
        return []

    target_year = now.year - offset
    dates: List[str] = []

    for d in extract_quarter_dates(n, reference_year=target_year):
        if d not in dates:
            dates.append(d)
    for d in extract_month_dates(n, reference_year=target_year):
        if d not in dates:
            dates.append(d)
    if dates:
        return dates

    if re.search(r"\bmes(?:es)?\b", n):
        return [f"{target_year:04d}-{now.month:02d}-01"]
    if has_quarter_ref(n):
        return [f"{target_year:04d}-{quarter_start_month(now.month):02d}-01"]
    if re.search(r"\b(?:ano|anio)s?\b", n):
        return [f"{target_year:04d}-01-01"]
    return []


# ─── Inferencia de granularidad desde texto ────────────────────────────────────

_QUARTER_RE = r"\btrimestre(s)?\b|\b[tq]\s*[1-4]\b|\b[tq][1-4]\b"
_MONTH_RE = r"\bmes(es)?\b|\bmensual(es)?\b"
_YEAR_RE = r"\bano(s)?\b|\banual(es)?\b"


def _detect_granularity(text: str, require_latest: bool = False, require_current: bool = False) -> Optional[str]:
    """Detecta granularidad temporal en texto libre."""
    n = normalize_text(text)
    if not n:
        return None

    if require_current and not contains_current_ref(n):
        return None
    if require_latest and not _has_explicit_relative_year_anchor(n):
        has_rel = contains_latest_ref(n) or contains_previous_ref(n)
        if not has_rel:
            return None

    if has_explicit_year(n):
        return None

    if re.search(_QUARTER_RE, n):
        return "q"
    # Referencias como "mayo del ano pasado" deben clasificarse mensual
    # antes de caer en la deteccion anual por la palabra "ano".
    month_tokens = set(MONTHS.keys()) | set(MONTH_ALIASES.keys())
    if any(tok in n.split() for tok in month_tokens):
        return "m"
    if re.search(_MONTH_RE, n):
        return "m"
    if re.search(_YEAR_RE, n):
        return "a"
    return None


def infer_relative_latest_granularity(
    raw_values: List[str],
    frequency: Optional[str],
) -> Optional[str]:
    """Infiere granularidad desde referencias relativas ('último', 'anterior', etc.)."""
    default_freq = frequency if frequency in {"m", "q", "a"} else None
    has_bare_relative = False

    for raw in raw_values:
        if not raw:
            continue
        g = _detect_granularity(raw, require_latest=True)
        if g:
            return g
        if (contains_latest_ref(raw) or contains_previous_ref(raw)) and not has_explicit_year(raw):
            has_bare_relative = True

    return default_freq if has_bare_relative else None


def infer_current_granularity(
    raw_values: List[str],
    frequency: Optional[str],
) -> Optional[str]:
    """Infiere granularidad desde referencias al período actual ('este', 'actual')."""
    default_freq = frequency if frequency in {"m", "q", "a"} else None
    has_bare_current = False

    for raw in raw_values:
        if not raw:
            continue
        g = _detect_granularity(raw, require_current=True)
        if g:
            return g
        if contains_current_ref(raw) and not has_explicit_year(raw):
            has_bare_current = True

    return default_freq if has_bare_current else None


def infer_frequency_from_period(raw_values: List[str]) -> Optional[str]:
    """Infiere ``m``/``q``/``a`` desde los textos de período (para req_form point/range/latest)."""
    valid = [r for r in raw_values if r]
    if not valid:
        return None

    cg = infer_current_granularity(valid, None)
    if cg in {"m", "q", "a"}:
        return cg

    rg = infer_relative_latest_granularity(valid, None)
    if rg in {"m", "q", "a"}:
        return rg

    for raw in valid:
        shift = _extract_relative_shift(raw)
        if shift:
            return shift[0]
        offset = _extract_relative_year_offset(raw)
        if offset is not None:
            n = normalize_text(raw)
            if has_quarter_ref(n):
                return "q"
            if any(m in n for m in MONTHS) or re.search(r"\bmes(?:es)?\b", n):
                return "m"
            if re.search(r"\b(?:ano|anio)s?\b", n):
                return "a"

    if any(has_quarter_ref(r) for r in valid):
        return "q"
    if any(extract_month_dates(r) for r in valid):
        return "m"
    if any(extract_decade_range(r) for r in valid):
        return "a"
    if any(is_year_only_ref(r) for r in valid):
        return "a"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Resolución principal de período
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_single_period(text: str) -> Tuple[Optional[str], List[str]]:
    """Parsea un solo valor de período a ``YYYY-MM-DD`` (fallback simple)."""
    if not text:
        return None, []

    n = normalize_text(text)

    if contains_latest_ref(n):
        return fmt_month_start(reference_now()), []

    year_m = re.search(r"\b((?:19|20)\d{2})\b", n)
    now = reference_now()
    year = int(year_m.group(1)) if year_m else now.year

    for name, num in MONTHS.items():
        if name in n:
            # Si no hay año explícito y el mes está en el futuro, asumir año anterior
            if not year_m and year == now.year and num > now.month:
                year -= 1
            return f"{year:04d}-{num:02d}-01", []

    qm = re.search(r"(?:t|trimestre)\s*([1-4])", n)
    if qm:
        return f"{year:04d}-{QUARTERS_START_MONTH[int(qm.group(1))]:02d}-01", []

    if year_m:
        return f"{year:04d}-01-01", []

    iso = re.search(r"((?:19|20)\d{2})-(\d{2})", n)
    if iso:
        return f"{int(iso.group(1)):04d}-{int(iso.group(2)):02d}-01", []

    return None, [text]


def resolve_period(
    raw_values: List[str],
    calc_mode: Optional[str],
    base_normalized: Dict[str, Optional[str]],
    req_form: Optional[str],
    frequency: Optional[str] = None,
) -> Optional[List[str]]:
    """Resuelve ``period`` a un rango ``[inicio, fin]`` según *req_form*.

    Args:
        raw_values: textos crudos de período del NER.
        calc_mode: modo de cálculo ('original', 'yoy', etc.).
        base_normalized: entidades ya normalizadas (para fallback).
        req_form: forma de la solicitud ('latest', 'point', 'range').
        frequency: código de frecuencia inferido ('m', 'q', 'a').
    """
    candidates: List[str] = []
    req = (req_form or "").strip().lower()
    now = reference_now()
    has_year_only = False
    is_quarterly = (frequency == "q") or any(has_quarter_ref(r) for r in raw_values if r)

    # 1. Intentar resolución de referencias relativas.
    for raw in raw_values:
        if not raw:
            continue
        anchored = extract_relative_anchored_dates(raw, ref_now=now)
        for d in anchored:
            if d not in candidates:
                candidates.append(d)
        if anchored:
            shift = _extract_relative_shift(raw)
            if shift and shift[0] == "a":
                has_year_only = True
            else:
                offset = _extract_relative_year_offset(raw)
                n = normalize_text(raw)
                has_m = any(mn in n for mn in MONTHS) or bool(re.search(r"\bmes(?:es)?\b", n))
                if offset is not None and re.search(r"\b(?:ano|anio)s?\b", n) and not has_m and not has_quarter_ref(raw):
                    has_year_only = True

    # 2. Referencia al período actual ("este trimestre", "este año").
    cg = infer_current_granularity(raw_values, frequency)
    if cg:
        if cg == "q":
            return [fmt_quarter_start(now), fmt_quarter_end(now)]
        if cg == "a":
            return [fmt_year_start(now), f"{now.year:04d}-12-31"]
        return [fmt_month_start(now), fmt_month_end(now)]

    # 3. Décadas.
    decades = [dr for dr in (extract_decade_range(r) for r in raw_values if r) if dr]
    if decades and req in {"point", "range"}:
        sy = min(d[0] for d in decades)
        ey = max(d[1] for d in decades)
        return [f"{sy:04d}-01-01", f"{ey:04d}-12-31"]

    # 4. Extracción explícita de trimestres / meses / años.
    for raw in raw_values:
        if not raw:
            continue
        extracted = extract_quarter_dates(raw, reference_year=now.year) if is_quarterly else []
        if not extracted:
            n = normalize_text(raw)
            if not is_quarterly or any(mn in n for mn in MONTHS) or any(a in n.split() for a in MONTH_ALIASES):
                extracted = extract_month_dates(raw, reference_year=now.year)
        for d in extracted:
            if d not in candidates:
                candidates.append(d)

        if not extracted:
            yd = extract_year_dates(raw)
            for d in yd:
                if d not in candidates:
                    candidates.append(d)
            n = normalize_text(raw)
            if yd and not any(mn in n for mn in MONTHS) and not has_quarter_ref(raw):
                has_year_only = True

        if not extracted and not extract_year_dates(raw):
            if (contains_latest_ref(raw) or contains_previous_ref(raw)) and not has_explicit_year(raw):
                continue
            single, _ = normalize_single_period(raw)
            if single and single not in candidates:
                candidates.append(single)

    # 5. Fallback: inferencia relativa sin fechas explícitas.
    if not candidates:
        rg = infer_relative_latest_granularity(raw_values, frequency)
        if rg:
            if rg == "q":
                pq = prev_quarter_anchor(now)
                return [fmt_quarter_start(pq), fmt_quarter_end(pq)]
            if rg == "a":
                py = prev_year_anchor(now)
                return [fmt_year_start(py), f"{py.year:04d}-12-31"]
            pm = prev_month_anchor(now)
            return [fmt_month_start(pm), fmt_month_end(pm)]

        bp = base_normalized.get("period")
        if bp:
            candidates.append(bp)

    # 6. Fallback: latest sin candidatos.
    if not candidates:
        if req == "latest":
            if is_quarterly:
                pq = prev_quarter_anchor(now)
                return [fmt_quarter_start(pq), fmt_quarter_end(pq)]
            pm = prev_month_anchor(now)
            return [fmt_month_start(pm), fmt_month_end(pm)]
        return None

    # 7. Parsear candidatos y construir rango final.
    parsed = [p for p in (parse_yyyymmdd(c) for c in candidates) if p]
    if not parsed:
        return None

    # Deduplicar por trimestre si el contexto es trimestral.
    if is_quarterly:
        seen_q: set = set()
        deduped: List[datetime] = []
        for d in parsed:
            qd = datetime(d.year, quarter_start_month(d.month), 1)
            key = (qd.year, qd.month)
            if key not in seen_q:
                seen_q.add(key)
                deduped.append(qd)
        parsed = deduped

    if req == "latest":
        if has_year_only and all(d.month == 1 and d.day == 1 for d in parsed):
            py = prev_year_anchor(now)
            return [fmt_year_start(py), f"{py.year:04d}-12-31"]
        if is_quarterly:
            pq = prev_quarter_anchor(now)
            return [fmt_quarter_start(pq), fmt_quarter_end(pq)]
        pm = prev_month_anchor(now)
        return [fmt_month_start(pm), fmt_month_end(pm)]

    if req == "range":
        s = sorted(parsed)
        first = fmt_month_start(s[0])
        if has_year_only and all(d.month == 1 and d.day == 1 for d in s):
            last = f"{s[-1].year:04d}-12-31"
        elif is_quarterly:
            last = fmt_quarter_end(s[-1])
        else:
            last = fmt_month_end(s[-1])
        return [first, last]

    # point (default)
    first = parsed[0]
    if has_year_only:
        return [fmt_year_start(first), f"{first.year:04d}-12-31"]
    if is_quarterly:
        return [fmt_quarter_start(first), fmt_quarter_end(first)]
    return [fmt_month_start(first), fmt_month_end(first)]
