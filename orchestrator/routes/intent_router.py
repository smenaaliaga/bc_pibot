from typing import Any, Iterable, Optional, Dict
import logging
import os
import json
import datetime
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# IMACEC deterministic intents (sin depender de orchestrator_old)
# ------------------------------------------------------------


def _load_imacec_defaults() -> Optional[Dict[str, str]]:
    try:
        cfg_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "..",
            "series",
            "config_default.json",
        )
        data = json.loads(open(cfg_path, "r", encoding="utf-8").read())
        d = data.get("defaults", {})
        block = d.get("IMACEC", {})
        if not block:
            return None
        return {
            "cod_serie": block.get("cod_serie"),
            "freq": block.get("freq_por_defecto"),
        }
    except Exception:
        return None


def _fetch_imacec_series(year: int) -> Optional[Dict[str, Any]]:
    defaults = _load_imacec_defaults()
    if not defaults or not defaults.get("cod_serie"):
        return None
    sid = defaults["cod_serie"]
    freq = defaults.get("freq") or None
    firstdate = f"{year-1}-01-01"
    lastdate = f"{year}-12-31"
    try:
        from get_series import get_series_api_rest_bcch  # type: ignore

        return get_series_api_rest_bcch(
            series_id=sid,
            firstdate=firstdate,
            lastdate=lastdate,
            target_frequency=freq,
            agg="avg",
        )
    except Exception as e:
        logger.error(f"[IMACEC_FETCH_ERROR] {e}")
        return None


def _format_latest_imacec(data: Dict[str, Any]) -> Optional[str]:
    obs = (data or {}).get("observations") or []
    if not obs:
        return None
    last = obs[-1]
    date = last.get("date", "")
    val = last.get("value")
    yoy = last.get("yoy_pct")
    parts = [f"Último IMACEC disponible ({date})"]
    if val is not None:
        try:
            parts.append(f"valor={float(val):.2f}")
        except Exception:
            parts.append(f"valor={val}")
    if yoy is not None:
        try:
            parts.append(f"variación anual={float(yoy):.1f}%")
        except Exception:
            parts.append(f"variación anual={yoy}%")
    return " | ".join(parts)


# Helpers PIB
def _fetch_pib_series(year: int) -> Optional[Dict[str, Any]]:
    try:
        cfg_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "series", "config_default.json")
        data_cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
        d = data_cfg.get("defaults", {})
        block = d.get("PIB_TOTAL", {})
        sid = block.get("cod_serie")
        freq = block.get("freq_por_defecto")
    except Exception:
        sid, freq = None, None
    if not sid:
        return None
    firstdate = f"{year-1}-01-01"
    lastdate = f"{year}-12-31"
    try:
        from get_series import get_series_api_rest_bcch  # type: ignore

        return get_series_api_rest_bcch(
            series_id=sid,
            firstdate=firstdate,
            lastdate=lastdate,
            target_frequency=freq or "T",
            agg="avg",
        )
    except Exception as e:
        logger.error(f"[PIB_FETCH_ERROR] {e}")
        return None


_IMACEC_LATEST_PATTERNS = [
    r"ultimo valor del imacec",
    r"\búltimo imacec\b",
    r"\bvalor del imacec\b",
    r"\bimacec\b.*ultimo",
    r"\bimacec\b.*valor",
    r"actividad economica.*ultimo",
    r"actividad económica.*ultimo",
]


def _is_imacec_latest(question: str) -> bool:
    q = (question or "").lower()
    return any(re.search(pat, q, flags=re.IGNORECASE) for pat in _IMACEC_LATEST_PATTERNS)


# ----------------------------
# IMACEC: mes específico / intervalo
# ----------------------------
_MONTH_MAP = {
    # nombres completos
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
    # abreviaturas comunes
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dic": 12,
}


def _detect_specific_month_year(question: str) -> Optional[tuple[int, int]]:
    q = (question or "").lower()
    # patrón: "marzo 2024", "sep 2023"
    for name, num in _MONTH_MAP.items():
        m = re.search(rf"{name}\s+(20\d{{2}})", q)
        if m:
            try:
                return int(num), int(m.group(1))
            except Exception:
                return None
    # numérico: mm/yyyy o mm-aaaa
    m2 = re.search(r"\b(0?[1-9]|1[0-2])[/-](20\d{2})\b", q)
    if m2:
        return int(m2.group(1)), int(m2.group(2))
    return None


def _detect_month_interval(question: str) -> Optional[tuple[int, int, int]]:
    q = (question or "").lower()
    m = re.search(r"(0?[1-9]|1[0-2])[/-](20\d{2})\s*(?:a|-|–|hasta)\s*(0?[1-9]|1[0-2])[/-](20\d{2})", q)
    if m:
        m_start, y_start, m_end, y_end = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if y_start == y_end:
            return m_start, m_end, y_start
        return None
    return None


def _format_imacec_obs_list(obs: list[dict]) -> str:
    lines = []
    for o in obs:
        date = o.get("date", "")
        val = o.get("value")
        yoy = o.get("yoy_pct")
        parts = [date]
        if val is not None:
            try:
                parts.append(f"valor={float(val):.2f}")
            except Exception:
                parts.append(f"valor={val}")
        if yoy is not None:
            try:
                parts.append(f"yoy={float(yoy):.1f}%")
            except Exception:
                parts.append(f"yoy={yoy}%")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


# ----------------------------
# IMACEC: toggles de métrica (pct vs yoy)
# ----------------------------
_TOGGLE_MONTHLY_PAT = re.compile(r"(variaci[oó]n mensual|mes a mes|m/m|mensual)", re.IGNORECASE)
_TOGGLE_ANNUAL_PAT = re.compile(r"(variaci[oó]n anual|interanual|a/a|anual)", re.IGNORECASE)
_FREQ_TO_T = re.compile(r"(trimestral|trimestre|a trimestral|a trimestres)", re.IGNORECASE)
_FREQ_TO_A = re.compile(r"(anual|a anual|año|años)", re.IGNORECASE)
_METHODOLOGY_PAT = re.compile(r"(qué es|que es|cómo se calcula|metodolog[ií]a|definici[oó]n)\s+(el\s+)?(imacec|pib)", re.IGNORECASE)

try:
    from orchestrator.data import data_flow as _df  # type: ignore
except Exception:
    _df = None  # type: ignore


def _toggle_imacec_metric(question: str) -> Optional[str]:
    if _df is None:
        return None
    ctx = getattr(_df, "_last_data_context", {}) or {}
    if (ctx.get("domain") or "").upper() != "IMACEC":
        return None
    data = ctx.get("data_full") or {}
    obs = data.get("observations") or []
    if not obs:
        return None
    q = (question or "").lower()
    want_monthly = bool(_TOGGLE_MONTHLY_PAT.search(q))
    want_annual = bool(_TOGGLE_ANNUAL_PAT.search(q))
    if not (want_monthly or want_annual):
        return None
    # último dato
    last = obs[-1]
    date = last.get("date", "")
    metric_key = "pct" if want_monthly else "yoy_pct"
    metric_label = "variación mensual" if want_monthly else "variación anual"
    val = last.get(metric_key)
    if val is None:
        return None
    try:
        val_txt = f"{float(val):.1f}%"
    except Exception:
        val_txt = str(val)
    return f"IMACEC {metric_label} ({date}): {val_txt}"


def _change_frequency_from_context(question: str) -> Optional[str]:
    if _df is None:
        return None
    if not (_FREQ_TO_T.search(question or "") or _FREQ_TO_A.search(question or "")):
        return None
    ctx = getattr(_df, "_last_data_context", {}) or {}
    data = ctx.get("data_full")
    domain = (ctx.get("domain") or "").upper()
    if not data or domain not in {"IMACEC", "PIB"}:
        return None
    freq_eff = (ctx.get("freq") or "").upper()
    # Determinar target
    target_freq = "T" if _FREQ_TO_T.search(question or "") else "A"
    if freq_eff == target_freq:
        return f"La serie ya está en frecuencia {target_freq}."
    try:
        from get_series import get_series_api_rest_bcch  # type: ignore
        sid = ctx.get("series_id") or data.get("meta", {}).get("series_id")
        if not sid:
            return None
        # usar rango del último año en contexto si existe
        year = ctx.get("year") or None
        fd = f"{int(year)-1}-01-01" if year else None
        ld = f"{int(year)}-12-31" if year else None
        data_new = get_series_api_rest_bcch(
            series_id=sid,
            firstdate=fd,
            lastdate=ld,
            target_frequency=target_freq,
            agg="avg",
        )
        obs = data_new.get("observations") or []
        if not obs:
            return "No encontré datos para la frecuencia solicitada."
        last = obs[-1]
        val = last.get("value")
        yoy = last.get("yoy_pct")
        date = last.get("date", "")
        parts = [f"{domain} frecuencia {target_freq} ({date})"]
        if val is not None:
            try:
                parts.append(f"valor={float(val):,.2f}".replace(",", "_").replace("_", "."))
            except Exception:
                parts.append(f"valor={val}")
        if yoy is not None:
            try:
                parts.append(f"variación anual={float(yoy):.1f}%")
            except Exception:
                parts.append(f"variación anual={yoy}%")
        # actualizar contexto compartido
        try:
            _df._last_data_context.update({"freq": target_freq, "data_full": data_new, "series_id": sid})
        except Exception:
            pass
        return " | ".join(parts)
    except Exception:
        logger.debug("change_frequency_from_context failed", exc_info=True)
        return None


def _methodology_response(question: str, domain: str) -> Optional[str]:
    if not _METHODOLOGY_PAT.search(question or ""):
        return None
    d = domain.upper()
    if d == "IMACEC":
        return (
            "El IMACEC es el índice mensual de actividad económica. "
            "Mide la variación de la actividad real de la economía chilena mes a mes. "
            "Se basa en el valor agregado de sectores productivos y se publica mensualmente por el Banco Central de Chile. "
            "Puedo buscar datos o explicarte algún sector específico si lo necesitas."
        )
    if d == "PIB":
        return (
            "El PIB es el producto interno bruto. Representa el valor agregado generado en la economía en un período (trimestre o año). "
            "Se calcula por enfoques de producción, gasto e ingreso y se publica trimestralmente (PIB total) por el Banco Central de Chile. "
            "Puedo mostrarte cifras recientes o desglosarlo por componentes si lo indicas."
        )
    return "Puedo explicar la metodología del indicador si me indicas cuál (IMACEC o PIB)."


# ----------------------------
# IMACEC: solicitud de gráfico (usa último contexto de data_flow)
# ----------------------------
_CHART_PAT = re.compile(r"gr[aá]fic[oa]|plot|visual", re.IGNORECASE)


def _detect_chart_request(question: str) -> bool:
    return bool(_CHART_PAT.search(question or ""))


def _chart_marker_from_context(domain: str) -> Optional[str]:
    if _df is None:
        return None
    ctx = getattr(_df, "_last_data_context", {}) or {}
    data = ctx.get("data_full")
    if not data:
        return None
    try:
        marker = _df._emit_chart_marker(domain, data)  # type: ignore[attr-defined]
        return marker
    except Exception:
        logger.debug("Could not build chart marker from context", exc_info=True)
        return None


def _handle_pib_quarter_year(match: Any) -> Iterable[str]:
    try:
        qtr_raw = match.group(1)
        year = int(match.group(2))
    except Exception:
        return iter(["No se pudo interpretar el trimestre y año solicitados para el PIB."])
    # map texto a número si corresponde
    q_map = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4, "primero": 1}
    try:
        q_num = int(qtr_raw)
    except Exception:
        q_num = q_map.get(qtr_raw.lower(), None)
    if q_num not in {1, 2, 3, 4}:
        return iter(["Trimestre no reconocido (use T1..T4 o texto)."])
    data = _fetch_pib_series(year)
    if not data:
        return iter(["No fue posible obtener el PIB para ese año/trimestre."])
    obs = data.get("observations") or []
    month_map = {1: "01", 2: "04", 3: "07", 4: "10"}
    target_date = f"{year}-{month_map[q_num]}-01"
    match_obs = next((o for o in obs if o.get("date") == target_date), None)
    if not match_obs:
        return iter(["No encontré ese trimestre del PIB."])
    parts = [f"PIB T{q_num} {year} ({target_date})"]
    v = match_obs.get("value")
    y = match_obs.get("yoy_pct")
    if v is not None:
        try:
            parts.append(f"valor={float(v):,.1f}".replace(",", "_").replace("_", "."))
        except Exception:
            parts.append(f"valor={v}")
    if y is not None:
        try:
            parts.append(f"variación anual={float(y):.1f}%")
        except Exception:
            parts.append(f"variación anual={y}%")
    try:
        if _df is not None:
            _df._last_data_context.update(
                {"series_id": data.get("meta", {}).get("series_id"), "domain": "PIB", "year": year, "freq": data.get("meta", {}).get("freq_effective"), "data_full": data}
            )
    except Exception:
        pass
    return iter([" | ".join(parts)])


# ----------------------------
# Configurable intents (catalog/intents.json)
# ----------------------------
@lru_cache(maxsize=1)
def _load_intents_config() -> list[dict]:
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "catalog", "intents.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        logger.error(f"[INTENTS_CFG_LOAD_ERROR] path={path} e={e}")
        return []
    months = "|".join(map(re.escape, _MONTH_MAP.keys()))
    for it in cfg:
        pats = []
        for p in it.get("patterns", []) or []:
            pats.append(p.replace("{MONTHS}", months))
        it["_compiled"] = [re.compile(p, flags=re.IGNORECASE) for p in pats]
    cfg.sort(key=lambda x: int(x.get("priority", 9999)))
    return cfg


def _intent_requires_domain_ok(intent: dict, question: str) -> bool:
    req = (intent.get("requiresDomain") or "").upper().strip()
    q = (question or "").lower()
    if not req:
        return True
    if req == "IMACEC":
        return "imacec" in q
    if req == "PIB":
        return re.search(r"\bpib\b", q) is not None
    return True


def _handle_intent_imacec_month_interval(match: Any) -> Iterable[str]:
    try:
        m1_name = match.group(1).lower()
        m2_name = match.group(2).lower()
        year = int(match.group(3))
    except Exception:
        return iter(["No se pudo interpretar el intervalo de meses."])
    m1 = _MONTH_MAP.get(m1_name)
    m2 = _MONTH_MAP.get(m2_name)
    if not (m1 and m2):
        return iter(["Meses no reconocidos en el intervalo solicitado."])
    if m1 > m2:
        m1, m2 = m2, m1
    data = _fetch_imacec_series(year)
    if not data:
        return iter(["No fue posible conectar con la API tras reintentos. Por favor, intenta nuevamente."])
    obs = data.get("observations") or []
    sel = []
    for o in obs:
        try:
            dt = datetime.date.fromisoformat(o.get("date", ""))
        except Exception:
            continue
        if dt.year == year and m1 <= dt.month <= m2:
            sel.append(o)
    if not sel:
        return iter(["No encontré ese rango del IMACEC."])
    msg = _format_imacec_obs_list(sel)
    try:
        if _df is not None:
            _df._last_data_context.update(
                {"series_id": data.get("meta", {}).get("series_id"), "domain": "IMACEC", "year": year, "freq": data.get("meta", {}).get("freq_effective"), "data_full": data}
            )
    except Exception:
        pass
    return iter([msg])


def _handle_intent_imacec_month_specific(match: Any) -> Iterable[str]:
    try:
        m_name = match.group(1).lower()
        year = int(match.group(2))
    except Exception:
        return iter(["No se pudo interpretar el mes y año solicitados."])
    month = _MONTH_MAP.get(m_name)
    if not month:
        return iter(["No se reconoció el mes solicitado."])
    data = _fetch_imacec_series(year)
    if not data:
        return iter(["No fue posible conectar con la API tras reintentos. Por favor, intenta nuevamente."])
    target_date = f"{year}-{month:02d}-01"
    obs = data.get("observations") or []
    match_obs = next((o for o in obs if o.get("date") == target_date), None)
    if not match_obs:
        return iter(["No encontré ese mes del IMACEC."])
    msg = _format_imacec_obs_list([match_obs])
    try:
        if _df is not None:
            _df._last_data_context.update(
                {"series_id": data.get("meta", {}).get("series_id"), "domain": "IMACEC", "year": year, "freq": data.get("meta", {}).get("freq_effective"), "data_full": data}
            )
    except Exception:
        pass
    return iter([msg])


def _dispatch_config_intents(question: str) -> Optional[Iterable[str]]:
    cfg = _load_intents_config()
    if not cfg:
        return None
    q = question or ""
    for it in cfg:
        if not _intent_requires_domain_ok(it, q):
            continue
        for rx in it.get("_compiled", []) or []:
            m = rx.search(q)
            if not m:
                continue
            handler_name = it.get("handler")
            if handler_name == "handle_imacec_month_interval":
                return _handle_intent_imacec_month_interval(m)
            if handler_name == "handle_imacec_month_specific":
                return _handle_intent_imacec_month_specific(m)
            if handler_name == "handle_pib_quarter_year":
                return _handle_pib_quarter_year(m)
    return None


# ----------------------------
# Vector fallback helper
# ----------------------------
_VECTOR_HINT_PAT = re.compile(r"otra serie|busca otra|sugerir serie|buscar serie", re.IGNORECASE)


def _maybe_vector_fallback(question: str, domain: str) -> Optional[Iterable[str]]:
    if domain.upper() == "OTHER" or _VECTOR_HINT_PAT.search(question or ""):
        return _vector_fallback(question)
    return None


# ----------------------------
# Vector fallback para sugerir series (catalog/search)
# ----------------------------
def _vector_fallback(question: str) -> Optional[Iterable[str]]:
    try:
        from search import search_serie_pg_vector  # type: ignore
    except Exception:
        return None
    try:
        matches = search_serie_pg_vector(question, top_k=3) or []
    except Exception:
        return None
    if not matches:
        return None
    lines = ["Sugerencias de series similares:"]
    for m in matches:
        sid = m.get("code") or m.get("series_id") or ""
        title = m.get("title") or m.get("name") or ""
        score = m.get("score")
        if score is not None:
            try:
                line = f"- {sid}: {title} (sim={float(score):.2f})"
            except Exception:
                line = f"- {sid}: {title}"
        else:
            line = f"- {sid}: {title}"
        lines.append(line)
    return iter(["\n".join(lines)])


# ----------------------------
# Helpers PIB (quarter detection)
# ----------------------------
_Q_PATTERN = re.compile(r"\bT([1-4])\b", re.IGNORECASE)
_Q_TEXT_MAP = {
    "primer trimestre": 1,
    "segundo trimestre": 2,
    "tercer trimestre": 3,
    "cuarto trimestre": 4,
}


def _extract_quarter_local(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.lower()
    for k, v in _Q_TEXT_MAP.items():
        if k in t:
            return v
    m = _Q_PATTERN.search(text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def route_intents(
    classification: Any,
    question: str,
    history_text: str,
    intent_classifier: Optional[Any] = None,  # kept for future extensibility
    memory: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> Optional[Iterable[str]]:
    """
    Decide whether to short-circuit con intents deterministas (IMACEC/PIB).
    Retorna iterable de chunks o None para seguir flujo normal.
    """
    try:
        domain = getattr(classification, "data_domain", "") or ""
    except Exception:
        domain = ""

    # Solo manejar IMACEC aquí; otros dominios siguen el flujo normal
    if domain.upper() != "IMACEC":
        return None

    # Último valor IMACEC
    if _is_imacec_latest(question):
        try:
            year = datetime.date.today().year
            data = _fetch_imacec_series(year)
            if data is None:
                return iter(["No pude obtener el IMACEC en este momento."])
            msg = _format_latest_imacec(data)
            if msg:
                try:
                    if memory and session_id and hasattr(memory, "set_facts"):
                        facts = {"imacec_last": msg}
                        last_obs = (data.get("observations") or [])[-1]
                        facts["imacec_last_date"] = last_obs.get("date", "")
                        if last_obs.get("value") is not None:
                            facts["imacec_last_value"] = str(last_obs.get("value"))
                        if last_obs.get("yoy_pct") is not None:
                            facts["imacec_last_yoy"] = str(last_obs.get("yoy_pct"))
                        facts["imacec_freq"] = (data.get("meta", {}) or {}).get("freq_effective", "")
                        facts["imacec_metric"] = "annual"
                        memory.set_facts(session_id, facts)  # type: ignore
                except Exception:
                    logger.debug("Could not store IMACEC last facts", exc_info=True)
                return iter([msg])
        except Exception:
            logger.exception("intent_router imacec_latest failed")
            return iter(["No pude obtener el IMACEC en este momento."])

    # Mes específico
    month_year = _detect_specific_month_year(question)
    if month_year:
        m, y = month_year
        data = _fetch_imacec_series(y)
        if data:
            target_date = f"{y}-{m:02d}-01"
            obs = data.get("observations") or []
            match = next((o for o in obs if o.get("date") == target_date), None)
            if match:
                msg = _format_imacec_obs_list([match])
                try:
                    if memory and session_id and hasattr(memory, "set_facts"):
                        memory.set_facts(
                            session_id,
                            {
                                "imacec_month": target_date,
                                "imacec_month_value": str(match.get("value", "")),
                                "imacec_freq": (data.get("meta", {}) or {}).get("freq_effective", ""),
                                "imacec_metric": "annual",
                            },
                        )
                except Exception:
                    logger.debug("Could not store IMACEC month fact", exc_info=True)
                return iter([msg])
        return iter(["No encontré ese mes del IMACEC."])

    # Intervalo de meses en el mismo año
    month_interval = _detect_month_interval(question)
    if month_interval:
        m_start, m_end, y = month_interval
        data = _fetch_imacec_series(y)
        if data:
            obs = data.get("observations") or []
            sel = []
            for o in obs:
                try:
                    dt = datetime.date.fromisoformat(o.get("date", ""))
                except Exception:
                    continue
                if dt.year == y and m_start <= dt.month <= m_end:
                    sel.append(o)
            if sel:
                msg = _format_imacec_obs_list(sel)
                try:
                    if memory and session_id and hasattr(memory, "set_facts"):
                        memory.set_facts(
                            session_id,
                            {
                                "imacec_range": f"{y}-{m_start:02d}..{y}-{m_end:02d}",
                                "imacec_freq": (data.get("meta", {}) or {}).get("freq_effective", ""),
                                "imacec_metric": "annual",
                            },
                        )
                except Exception:
                    logger.debug("Could not store IMACEC range fact", exc_info=True)
                return iter([msg])
        return iter(["No encontré ese rango del IMACEC."])

    # Toggle métrica (variación mensual vs anual) usando último dato cacheado
    toggle_msg = _toggle_imacec_metric(question)
    if toggle_msg:
        try:
            if memory and session_id and hasattr(memory, "set_facts"):
                memory.set_facts(
                    session_id,
                    {
                        "imacec_metric": "monthly" if _TOGGLE_MONTHLY_PAT.search(question or "") else "annual",
                        "imacec_freq": (getattr(_df, "_last_data_context", {}) or {}).get("freq", ""),
                    },
                )
        except Exception:
            logger.debug("Could not store IMACEC metric fact", exc_info=True)
        return iter([toggle_msg])

    # Solicitud de gráfico con contexto actual
    if _detect_chart_request(question):
        marker = _chart_marker_from_context("IMACEC")
        if marker:
            try:
                if memory and session_id and hasattr(memory, "set_facts"):
                    memory.set_facts(session_id, {"chart_last": "IMACEC"})
            except Exception:
                logger.debug("Could not store chart fact", exc_info=True)
            return iter([marker])
        return iter(["No tengo suficientes datos en memoria para generar el gráfico del IMACEC."])

    # Metodológico IMACEC
    m_resp = _methodology_response(question, domain)
    if m_resp:
        return iter([m_resp])

    # Intents configurables (IMACEC) desde catalog/intents.json
    cfg_iter = _dispatch_config_intents(question)
    if cfg_iter is not None:
        return cfg_iter

    # Cambio de frecuencia usando serie en contexto
    freq_msg = _change_frequency_from_context(question)
    if freq_msg:
        return iter([freq_msg])

    # ----------------------------
    # PIB determinista
    # ----------------------------
    if domain.upper() == "PIB":
        try:
            from . import data_flow as _df  # type: ignore
        except Exception:
            _df = None  # type: ignore

    # Solicitud de gráfico con contexto actual PIB
        if _detect_chart_request(question):
            marker = _chart_marker_from_context("PIB")
            if marker:
                try:
                    if memory and session_id and hasattr(memory, "set_facts"):
                        memory.set_facts(session_id, {"chart_last": "PIB"})
                except Exception:
                    logger.debug("Could not store chart fact", exc_info=True)
                return iter([marker])

        # Metodológico PIB
        m_resp_pib = _methodology_response(question, "PIB")
        if m_resp_pib:
            return iter([m_resp_pib])

        # Usar contexto existente para último valor/trimestre
        if _df is not None:
            ctx = getattr(_df, "_last_data_context", {}) or {}
            if (ctx.get("domain") or "").upper() == "PIB":
                data = ctx.get("data_full") or {}
                obs = data.get("observations") or []
                if obs:
                    q = (question or "").lower()
                    # Rango específico: año o intervalo
                    year_req = None
                    m_year = re.search(r"(20\\d{2})", q)
                    if m_year:
                        try:
                            year_req = int(m_year.group(1))
                        except Exception:
                            year_req = None
                    # Último valor (más variantes)
                    want_latest = any(
                        pat in q
                        for pat in [
                            "ultimo valor del pib",
                            "último valor del pib",
                            "pib más reciente",
                            "pib mas reciente",
                            "pib reciente",
                            "pib ultimo",
                            "pib último",
                        ]
                    )

                    # Filtrar por año si se pidió
                    obs_filtered = obs
                    if year_req:
                        obs_filtered = [o for o in obs if str(o.get("date", ""))[:4] == str(year_req)]
                    if not obs_filtered:
                        obs_filtered = obs

                    last = obs_filtered[-1]
                    date = last.get("date", "")
                    val = last.get("value")
                    yoy = last.get("yoy_pct")
                    parts = [f"PIB {date[:4]} ({date})"]
                    if val is not None:
                        try:
                            parts.append(f"valor={float(val):,.1f}".replace(",", "_").replace("_", "."))
                        except Exception:
                            parts.append(f"valor={val}")
                    if yoy is not None:
                        try:
                            parts.append(f"variación anual={float(yoy):.1f}%")
                        except Exception:
                            parts.append(f"variación anual={yoy}%")
                    # Trimestre específico si se pidió Tn
                    qtr = _extract_quarter_local(question)
                    if qtr:
                        month_map = {1: "01", 2: "04", 3: "07", 4: "10"}
                        t_month = month_map.get(qtr)
                        if t_month:
                            target_date = f"{ctx.get('year') or date[:4]}-{t_month}-01"
                            match = next((o for o in obs if o.get("date") == target_date), None)
                            if match:
                                parts = [f"PIB T{qtr} {target_date[:4]}"]
                                v2 = match.get("value")
                                y2 = match.get("yoy_pct")
                                if v2 is not None:
                                    try:
                                        parts.append(f"valor={float(v2):,.1f}".replace(",", "_").replace("_", "."))
                                    except Exception:
                                        parts.append(f"valor={v2}")
                                if y2 is not None:
                                    try:
                                        parts.append(f"variación anual={float(y2):.1f}%")
                                    except Exception:
                                        parts.append(f"variación anual={y2}%")
                                return iter([" | ".join(parts)])
                    # store facts
                    try:
                        if memory and session_id and hasattr(memory, "set_facts"):
                            memory.set_facts(session_id, {"pib_last": " | ".join(parts), "pib_last_date": date, "pib_freq": ctx.get("freq", "")})
                    except Exception:
                        logger.debug("Could not store PIB facts", exc_info=True)
                    try:
                        if memory and session_id and hasattr(memory, "set_facts"):
                            memory.set_facts(session_id, {"pib_last": " | ".join(parts), "pib_last_date": date})
                    except Exception:
                        logger.debug("Could not store PIB facts", exc_info=True)

                    # Si se pidió rango/año explícito, devolver los puntos del año
                    if year_req:
                        lines = []
                        for o in obs:
                            d = o.get("date", "")
                            if not str(d).startswith(str(year_req)):
                                continue
                            v = o.get("value")
                            y = o.get("yoy_pct")
                            row = [d]
                            if v is not None:
                                try:
                                    row.append(f"valor={float(v):,.1f}".replace(',', '_').replace('_','.'))
                                except Exception:
                                    row.append(f"valor={v}")
                            if y is not None:
                                try:
                                    row.append(f"yoy={float(y):.1f}%")
                                except Exception:
                                    row.append(f"yoy={y}%")
                            lines.append(" | ".join(row))
                        if lines:
                            return iter([" | ".join(parts)] + ["\n".join(lines)])
                    return iter([" | ".join(parts)])

        # Fallback: intentar fetch rápido del año actual
        try:
            year = datetime.date.today().year
            try:
                cfg_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "series", "config_default.json")
                data_cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
                d = data_cfg.get("defaults", {})
                block = d.get("PIB_TOTAL", {})
                sid_pib = block.get("cod_serie")
                freq_pib = block.get("freq_por_defecto")
            except Exception:
                sid_pib, freq_pib = None, None
            sid = sid_pib
            if sid:
                from get_series import get_series_api_rest_bcch  # type: ignore

                data_pib = get_series_api_rest_bcch(
                    series_id=sid,
                    firstdate=f"{year-1}-01-01",
                    lastdate=f"{year}-12-31",
                    target_frequency=freq_pib or "T",
                    agg="avg",
                )
                obs = data_pib.get("observations") or []
                if obs:
                    last = obs[-1]
                    parts = [f"Último PIB ({last.get('date','')})"]
                    if last.get("value") is not None:
                        try:
                            parts.append(f"valor={float(last.get('value')):,.1f}".replace(",", "_").replace("_", "."))
                        except Exception:
                            parts.append(f"valor={last.get('value')}")
                    if last.get("yoy_pct") is not None:
                        try:
                            parts.append(f"variación anual={float(last.get('yoy_pct')):,.1f}%")
                        except Exception:
                            parts.append(f"variación anual={last.get('yoy_pct')}%")
                    try:
                        if memory and session_id and hasattr(memory, "set_facts"):
                            memory.set_facts(session_id, {"pib_last": " | ".join(parts), "pib_last_date": last.get("date", ""), "pib_freq": freq_pib or "T"})
                    except Exception:
                        logger.debug("Could not store PIB facts", exc_info=True)
                    return iter([" | ".join(parts)])
        except Exception:
            logger.debug("PIB intent fallback failed", exc_info=True)

    # Vector fallback si no hubo match y se sugiere otra serie o domain=OTHER
    vec = _maybe_vector_fallback(question, domain or "")
    if vec is not None:
        return vec

    return None
