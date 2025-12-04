"""
DATA flow (metodología y tablas) desacoplado del legacy.

Incluye:
- Fase 1: respuesta metodológica (sin cifras).
- Fetch de series BCCh con reintentos.
- Fase 2: tabla comparativa año-1 vs año, metadatos, descarga CSV y marcador de gráfico.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import get_settings

# Opcional: metadatos de series
try:
    from get_series import get_series_metadata  # type: ignore
except Exception:
    get_series_metadata = None  # type: ignore

logger = logging.getLogger(__name__)
_settings = get_settings()

# Contexto simple para compartir meta entre fases
_last_data_context: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Prompts de fase DATA (copiados/ajustados del legacy)
# ---------------------------------------------------------------------------
_DATA_SYSTEM = (
    "Eres el asistente económico del Banco Central de Chile (PIBot). "
    "Respondes SIEMPRE en español.\n\n"
    "Estás en el modo de respuesta orientada a DATOS. "
    "El usuario podría pedir valores numéricos, tablas o variaciones. "
    "Esta versión puede tener acceso limitado a datos; si no puedes entregar "
    "valores reales, dilo explícitamente y describe los pasos para obtenerlos "
    "desde la serie adecuada. No inventes cifras."
)

_DATA_HUMAN = (
    "Historial de la conversación (puede estar vacío):\n"
    "{history}\n\n"
    "Consulta actual del usuario:\n"
    "{question}\n\n"
    "Clasificación técnica de la consulta (NO la muestres tal cual al usuario):\n"
    "query_type={query_type}, data_domain={data_domain}, is_generic={is_generic}, "
    "default_key={default_key}\n"
    "Árbol IMACEC={imacec_tree}\n"
    "Árbol PIB={pibe_tree}\n\n"
    "Instrucción de modo (no la muestres, solo síguela):\n"
    "{mode_instruction}\n\n"
    "Responde con un texto breve, sin números si no los tienes, "
    "explicando metodología y cómo obtendrías y presentarías los datos "
    "para el periodo que menciona (por ejemplo, 2025), sin inventar cifras."
)

_data_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _DATA_SYSTEM),
        ("human", _DATA_HUMAN),
    ]
)

_llm_data = ChatOpenAI(
    model=_settings.openai_model,
    temperature=0.1,
    streaming=True,
)

# ---------------------------------------------------------------------------
# Helpers de año/frecuencia y tablas/CSV
# ---------------------------------------------------------------------------
_YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")


def _infer_domain_from_history(history_text: str) -> Optional[str]:
    """Si el clasificador no trae dominio, intenta deducirlo del historial."""
    h = (history_text or "").lower()
    if "imacec" in h:
        return "IMACEC"
    if "pib" in h and "regional" in h:
        return "PIB_REGIONAL"
    if "pib" in h:
        return "PIB"
    return None


def _extract_year(text: str) -> Optional[int]:
    m = _YEAR_PATTERN.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _get_latest_year_from_data(data: Dict[str, Any]) -> Optional[int]:
    obs = (data or {}).get("observations") or []
    years = []
    for o in obs:
        d = o.get("date")
        if not d:
            continue
        try:
            years.append(int(str(d)[:4]))
        except Exception:
            continue
    return max(years) if years else None


def _format_series_metadata_block(series_id: str) -> str:
    if not get_series_metadata:
        return ""
    md = get_series_metadata(series_id)
    if not md:
        return ""
    freq = (md.get("default_frequency") or "").strip()
    unit = (md.get("unit") or "").strip()
    code = (md.get("code") or "").strip()
    title = (md.get("title") or "").strip()
    url = (md.get("source_url") or "").strip()
    metodo = (md.get("metodologia") or "").strip()
    lines = [
        f"1. Código: {code}",
        f"2. Título: {title}",
        f"3. Frecuencia: {freq}",
        f"4. Unidad: {unit}",
        f"5. Gráfico: {url}",
        f"6. Metodología: {metodo}",
    ]
    return "\n".join(lines) + "\n\n"


def _build_year_table(data: Dict[str, Any], year: int) -> str:
    try:
        from get_series import build_year_comparison_table_text  # type: ignore
    except Exception as e:
        logger.error(f"[DATA_TABLE] No se pudo importar función canónica: {e}")
        return ""

    table_text = build_year_comparison_table_text(data, year)
    lines = table_text.split("\n")
    body_lines = lines[3:] if len(lines) > 3 else []
    for line in body_lines[:12]:
        logger.info(f"[DATA_TABLE_CONTENT] {line}")
    if not body_lines:
        logger.warning(f"[DATA_TABLE] Sin observaciones para {year}")
    return table_text


def _export_table_to_csv(table_text: str, filename_base: str) -> str:
    lines = [l for l in table_text.split("\n") if l.strip()]
    if not lines:
        raise ValueError("Tabla vacía para exportar")
    header_idx = sep_idx = None
    for i, ln in enumerate(lines):
        if "|" in ln and header_idx is None and not ln.lower().startswith("comparación "):
            header_idx = i
        elif header_idx is not None and re.match(r"^[\-\s|]+$", ln):
            sep_idx = i
            break
    if header_idx is None or sep_idx is None:
        raise ValueError("No se pudo identificar encabezado/separador en la tabla Markdown")
    header_cols = [c.strip() for c in lines[header_idx].split("|")]
    data_rows = []
    for ln in lines[sep_idx + 1 :]:
        if "|" not in ln:
            continue
        cols = [c.strip() for c in ln.split("|")]
        if len(cols) != len(header_cols):
            continue
        data_rows.append(cols)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    export_dir = os.path.join(root, "logs", "exports")
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, f"{filename_base}.csv")
    import csv

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_cols)
        for row in data_rows:
            w.writerow(row)
    return filepath


def _emit_csv_download_marker(table_text: str, filename_base: str, preferred_filename: Optional[str] = None) -> str:
    if not (table_text and table_text.strip()):
        return ""
    try:
        csv_path = _export_table_to_csv(table_text, filename_base)
    except Exception as _e_csv:
        logger.error(f"[CSV_EXPORT_ERROR] base={filename_base} e={_e_csv}")
        return ""
    fname = preferred_filename or os.path.basename(csv_path)
    block = [
        "##CSV_DOWNLOAD_START",
        f"path={csv_path}",
        f"filename={fname}",
        "mimetype=text/csv",
        "label=Descargar CSV",
        "##CSV_DOWNLOAD_END",
        "",
    ]
    return "\n".join(block)


def _emit_chart_marker(domain: str, data: Dict[str, Any]) -> Optional[str]:
    if not data:
        return None
    obs = data.get("observations") or []
    if not obs:
        return None
    metric_type = _last_data_context.get("metric_type", "annual")
    try:
        import csv
        import time as _t

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        export_dir = os.path.join(root, "logs", "exports")
        os.makedirs(export_dir, exist_ok=True)
        ts = int(_t.time())
        filename_base = f"{domain.lower()}_chart_{ts}.csv"
        path = os.path.join(export_dir, filename_base)
        year_ctx = _last_data_context.get("year") or _get_latest_year_from_data(data)
        rows_out = []
        for o in obs:
            d = o.get("date")
            if not d:
                continue
            try:
                y = int(str(d)[:4])
            except Exception:
                continue
            if year_ctx and y != int(year_ctx):
                continue
            val = o.get("pct") if metric_type == "monthly" else o.get("yoy_pct")
            if val is None:
                continue
            rows_out.append((d, val))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if metric_type == "monthly":
                w.writerow(["date", "pct"])
            else:
                w.writerow(["date", "yoy_pct"])
            for d, val in rows_out:
                w.writerow([d, val])
    except Exception as _e_chart:
        logger.error(f"[CHART_EXPORT_ERROR] domain={domain} e={_e_chart}")
        return None
    title = (
        f"{domain.upper()} - Variación mensual (%) {year_ctx}"
        if metric_type == "monthly"
        else f"{domain.upper()} - Variación anual (%) {year_ctx}"
    )
    columns_line = "columns=date,pct" if metric_type == "monthly" else "columns=date,yoy_pct"
    block = [
        "##CHART_START",
        "type=line",
        f"title={title}",
        f"data_path={path}",
        columns_line,
        f"domain={domain}",
        "##CHART_END",
        "",
    ]
    logger.info(
        f"[CHART_MARKER_EMITTED] path={path} domain={domain} rows={len(rows_out)} year_ctx={year_ctx} cols={columns_line}"
    )
    return "\n".join(block)


# ---------------------------------------------------------------------------
# Fetch helpers y tablas adicionales
# ---------------------------------------------------------------------------
def _load_defaults_for_domain(domain: str) -> Optional[Dict[str, str]]:
    try:
        cfg_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "..",
            "series",
            "config_default.json",
        )
        data = json.loads(open(cfg_path, "r", encoding="utf-8").read())
        d = data.get("defaults", {})
        key_map = {"IMACEC": "IMACEC", "PIB": "PIB_TOTAL", "PIB_REGIONAL": "PIB_REGIONAL"}
        block = d.get(key_map.get(domain, ""), {})
        if not block:
            return None
        return {
            "cod_serie": block.get("cod_serie"),
            "freq": block.get("freq_por_defecto"),
        }
    except Exception:
        return None


def _get_series_with_retry(
    series_id: str,
    firstdate: Optional[str],
    lastdate: Optional[str],
    target_frequency: Optional[str],
    agg: str = "avg",
    retries: int = 2,
    backoff: float = 1.0,
) -> Optional[Dict[str, Any]]:
    try:
        from get_series import get_series_api_rest_bcch
    except Exception as e:
        logger.error(f"[DATA_FETCH] import get_series_api_rest_bcch falló: {e}")
        return None
    for attempt in range(retries + 1):
        try:
            return get_series_api_rest_bcch(
                series_id=series_id,
                firstdate=firstdate,
                lastdate=lastdate,
                target_frequency=target_frequency,
                agg=agg,
            )
        except Exception as e:
            if attempt < retries:
                delay = backoff * (2**attempt)
                logger.warning(
                    f"[DATA_FETCH_RETRY] sid={series_id} intento={attempt+1}/{retries+1} en {delay:.1f}s | error={e}"
                )
                try:
                    time.sleep(delay)
                except Exception:
                    pass
                continue
            logger.error(f"[DATA_FETCH] Error final obteniendo serie {series_id}: {e}")
            return None


def _fetch_series_for_year(domain: str, year: int) -> Optional[Dict[str, Any]]:
    defaults = _load_defaults_for_domain(domain)
    if not defaults or not defaults.get("cod_serie"):
        logger.warning(f"[DATA_FETCH] defaults ausentes para domain={domain}")
        return None
    series_id = defaults["cod_serie"]
    freq = defaults.get("freq") or defaults.get("freq_por_defecto") or None
    firstdate = f"{year-1}-01-01"
    lastdate = f"{year}-12-31"
    logger.info(
        f"[YEAR_DETECT] domain={domain} year_detected={year} series_id={series_id} rango={firstdate}->{lastdate} freq_default={freq}"
    )
    return _get_series_with_retry(series_id, firstdate, lastdate, freq, agg="avg")


def _fetch_series_for_year_by_series_id(series_id: str, year: int, target_freq: Optional[str]) -> Optional[Dict[str, Any]]:
    firstdate = f"{year-1}-01-01"
    lastdate = f"{year}-12-31"
    return _get_series_with_retry(series_id, firstdate, lastdate, target_freq, agg="avg")


def _format_last_yoy_from_table(data: Dict[str, Any], year: int) -> Optional[str]:
    obs = (data or {}).get("observations") or []
    rows = []
    for o in obs:
        d = o.get("date")
        if not d:
            continue
        try:
            y = int(str(d)[:4])
        except Exception:
            continue
        if y != year:
            continue
        yoy = o.get("yoy_pct")
        if yoy is None:
            continue
        rows.append((str(d), float(yoy)))
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    last_date, last_yoy = rows[-1]
    try:
        month = last_date[5:7]
        txt_date = f"{month}/{year}"
    except Exception:
        txt_date = str(last_date)
    return f"La última variación anual disponible es {last_yoy:.1f}% (periodo {txt_date})."


def _sanitize_llm_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<bound method BaseMessage.text of AIMessageChunk\([^>]+\)>", "", text)
    text = re.sub(r"content='' additional_kwargs=\{\} response_metadata=\{[^}]*\} id='run--[0-9a-f-]+'", "", text)
    text = re.sub(r"([a-záéíóúñ])([A-ZÁÉÍÓÚÑ])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_year_change_only_table(data: Dict[str, Any], year: int) -> str:
    obs = (data or {}).get("observations") or []
    rows = []
    for o in obs:
        d = o.get("date")
        if not d:
            continue
        try:
            y = int(str(d)[:4])
        except Exception:
            continue
        if y != year:
            continue
        yoy = o.get("yoy_pct")
        if yoy is None:
            continue
        rows.append((str(d), float(yoy)))
    lines: List[str] = []
    lines.append(f"Comparación {year-1} vs {year}")
    lines.append("Periodo | Variación anual")
    lines.append("----|-----------------")
    for d, yoy in sorted(rows, key=lambda r: r[0]):
        lines.append(f"{d} | {yoy:.1f}")
    for ln in lines[3:6]:
        logger.info(f"[DATA_TABLE_CONTENT_YOY_ONLY] {ln}")
    return "\n".join(lines)


def _build_latest_only_table(data: Dict[str, Any]) -> str:
    obs = (data or {}).get("observations") or []
    last = None
    for o in obs:
        if o.get("yoy_pct") is not None or o.get("value") is not None:
            last = o
    if not last:
        return "No se encontró una observación reciente."
    date = last.get("date") or "(sin fecha)"
    yoy = last.get("yoy_pct")
    val = last.get("value")
    if yoy is not None:
        metric = f"{float(yoy):.1f}%"
        header_metric = "Variación anual"
    elif val is not None:
        try:
            metric = f"{float(val):,.2f}".replace(",", "_").replace("_", ".")
        except Exception:
            metric = str(val)
        header_metric = "Valor"
    else:
        metric = "--"
        header_metric = "Valor"
    lines = [
        f"Último periodo | {header_metric}",
        "---------------|----------------",
        f"{date} | {metric}",
    ]
    for ln in lines:
        logger.info(f"[DATA_TABLE_CONTENT_LATEST_ONLY] {ln}")
    return "\n".join(lines)


def _build_year_yoy_simple_table(data: Dict[str, Any], year: int) -> str:
    obs = (data or {}).get("observations") or []
    rows = []
    for o in obs:
        d = o.get("date")
        if not d:
            continue
        try:
            y = int(str(d)[:4])
        except Exception:
            continue
        if y != year:
            continue
        yoy = o.get("yoy_pct")
        if yoy is None:
            continue
        rows.append((str(d), float(yoy)))
    lines: List[str] = []
    lines.append("Periodo | Variación anual")
    lines.append("----|-----------------")
    for d, yoy in sorted(rows, key=lambda r: r[0]):
        lines.append(f"{d} | {yoy:.1f}")
    for ln in lines[2:5]:
        logger.info(f"[DATA_TABLE_CONTENT_YEAR_SIMPLE] {ln}")
    return "\n".join(lines)


def _handle_vector_search_other_series(question: str) -> Optional[List[Dict[str, Any]]]:
    try:
        from search import search_serie_pg_vector  # type: ignore
    except Exception:
        search_serie_pg_vector = None
    if search_serie_pg_vector is None:
        return None
    if not re.search(r"otra serie|otra.*serie|consultar otra serie", (question or "").lower()):
        return None
    try:
        matches = search_serie_pg_vector(question, top_k=3) or []
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH] Error búsqueda vectorial: {e}")
        return None
    if not matches:
        return None
    return matches


# ---------------------------------------------------------------------------
# Fase 1: metodológica (sin tablas)
# ---------------------------------------------------------------------------
def stream_phase(classification: Any, question: str, history_text: str) -> Iterable[str]:
    try:
        vars_in = {
            "history": history_text,
            "question": question,
            "query_type": getattr(classification, "query_type", None),
            "data_domain": getattr(classification, "data_domain", None),
            "is_generic": getattr(classification, "is_generic", None),
            "default_key": getattr(classification, "default_key", None),
            "imacec_tree": asdict(classification.imacec) if getattr(classification, "imacec", None) else None,
            "pibe_tree": asdict(classification.pibe) if getattr(classification, "pibe", None) else None,
            "mode_instruction": "Genera una respuesta metodológica clara. No inventes cifras.",
        }
        chain = _data_prompt | _llm_data
        def _chunk_to_text(chunk: Any) -> str:
            for attr in ("content", "text"):
                val = getattr(chunk, attr, None)
                if val is None:
                    continue
                if callable(val) and attr != "text":
                    try:
                        val = val()
                    except Exception:
                        val = str(val)
                if isinstance(val, str):
                    return val
                return str(val)
            return ""
        for chunk in chain.stream(vars_in):
            content = _chunk_to_text(chunk)
            if content.startswith("<bound method"):
                parts = content.split(">", 1)
                content = parts[1].strip() if len(parts) == 2 else ""
            if content:
                yield content
        return
    except Exception as e:
        logger.error(f"[DATA_FLOW] stream_phase falló: {e}")
    yield "No pude generar la respuesta de datos ahora mismo."


# ---------------------------------------------------------------------------
# Fase 2: con tabla (ya sin delegar al legacy)
# ---------------------------------------------------------------------------
def _summarize_with_llm(domain: str, year: int, table_text: str) -> str:
    system_msg = (
        "Eres el asistente económico del Banco Central de Chile (PIBot). Responde SIEMPRE en español. "
        "Estás en la FASE 2 de una respuesta orientada a DATOS. "
        "SOLO debes generar una frase introductoria neutral y TRES preguntas de seguimiento, "
        "sin conclusiones ni interpretaciones, sin repetir la tabla ni valores numéricos."
    )
    table_description = (
        f"Tabla de comparación año anterior vs año actual para el dominio '{domain}' y el año {year}. "
        "La tabla ya fue mostrada al usuario y contiene columnas de periodo, año anterior, año actual y variación anual."
    )
    human_msg = (
        f"Dominio: {domain}\nAño consultado: {year}\n"
        f"Resumen de tabla ya mostrada al usuario: {table_description}\n"
        "Genera una frase introductoria neutral y luego exactamente tres preguntas de seguimiento."
    )
    chain = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", human_msg),
        ]
    ) | _llm_data
    out: List[str] = []
    def _chunk_to_text(chunk: Any) -> str:
        for attr in ("content", "text"):
            val = getattr(chunk, attr, None)
            if val is None:
                continue
            if callable(val) and attr != "text":
                try:
                    val = val()
                except Exception:
                    val = str(val)
            if isinstance(val, str):
                return val
            return str(val)
        return ""
    for chunk in chain.stream({}):
        content = _chunk_to_text(chunk)
        if content.startswith("<bound method"):
            parts = content.split(">", 1)
            content = parts[1].strip() if len(parts) == 2 else ""
        if content:
            out.append(content)
    return _sanitize_llm_text("".join(out))


def _stream_data_phase_with_table(
    classification: Any,
    question: str,
    history_text: str,
    domain: str,
    year: int,
    data: Dict[str, Any],
) -> Iterable[str]:
    meta = data.get("meta", {}) or {}
    _last_data_context.update(
        {
            "series_id": meta.get("series_id"),
            "domain": domain,
            "year": year,
            "freq": meta.get("freq_effective"),
            "data_full": data,
        }
    )

    table_text = _build_year_table(data, year)
    lines_count = table_text.count("\n") + 1
    logger.info(f"[DATA_TABLE] domain={domain} year={year} lines={lines_count}")
    yield "\n" + table_text + "\n\n"

    series_id = meta.get("series_id")
    if series_id:
        md_block = _format_series_metadata_block(series_id)
        if md_block.strip():
            logger.info(f"[SERIES_META] series_id={series_id}")
            yield md_block + "\n"

    summary_full = _summarize_with_llm(domain, year, table_text)
    for pat in [r"no puedo proporcionar cifras", r"no puedo entregar cifras", r"no puedo proporcionar valores"]:
        summary_full = re.sub(pat, "", summary_full, flags=re.IGNORECASE)
    for s in re.split(r"(?<=[.!?])\s+", summary_full.strip()):
        if s:
            yield s + "\n"

    try:
        marker = _emit_csv_download_marker(
            table_text, f"{domain.lower()}_{year}", preferred_filename=f"{domain.lower()}_{year}.csv"
        )
        if marker:
            yield "\n" + marker + "\n"
    except Exception as _e_footer:
        logger.error(f"[CSV_MARKER_ERROR] domain={domain} year={year} e={_e_footer}")


def _detect_chart_request(question: str) -> bool:
    q = (question or "").lower()
    if re.search(r"grafico|gráfico", q) and ("imacec" in q or "pib" in q or "serie" in q):
        return True
    if re.search(r"realiza un gráfico|realiza un grafico|mostrar gráfico", q):
        return True
    return False


def stream_data_flow_full(
    classification: Any,
    question: str,
    history_text: str,
) -> Iterable[str]:
    """Fetch de datos y tabla; si falla, cae a fase metodológica."""
    domain = getattr(classification, "data_domain", "") or "IMACEC"
    if not domain or domain == "OTHER":
        inferred = _infer_domain_from_history(history_text) or _last_data_context.get("domain")
        if inferred:
            logger.info(f"[DATA_DOMAIN_INFERRED] domain={inferred} (fallback from history/context)")
            domain = inferred
    year = _extract_year(question)
    if year is None:
        try:
            year = int(time.strftime("%Y"))
        except Exception:
            year = None

    data = None
    if year:
        data = _fetch_series_for_year(domain, year)
    if data is None and getattr(classification, "default_key", None) and year:
        sid = getattr(classification, "default_key", None)
        if sid:
            data = _fetch_series_for_year_by_series_id(str(sid), year, None)

    if not data:
        # Fallback metodológico si no hay datos
        for chunk in stream_phase(classification, question, history_text):
            if chunk:
                yield chunk
        yield "\nNo pude obtener la serie de datos para construir la tabla.\n"
        return

    if _detect_chart_request(question):
        marker = _emit_chart_marker(domain, data)
        if marker:
            yield "\n" + marker + "\n"

    for chunk in _stream_data_phase_with_table(classification, question, history_text, domain, year or 0, data):
        if chunk:
            yield chunk
