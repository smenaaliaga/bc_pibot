# -*- coding: utf-8 -*-
"""
get_series.py
-------------
Capa de acceso a series del Banco Central de Chile (BCCh) vía API REST,
con cálculo de variaciones y cacheo en Redis.

Funciones principales
=====================

   - get_series_api_rest_bcch(...)
   - Llama a la API REST del BCCh (SieteRestWS).
   - Normaliza las observaciones (fecha, valor, status).
   - Opcionalmente remuestrea a otra frecuencia (D/M/Q/A) con agregación (avg/sum/first/last).
   - Calcula:
       * pct     → variación % respecto del período anterior.
       * yoy_pct → variación % respecto del mismo período del año anterior.
   - Almacena el resultado completo en Redis (meta + observaciones enriquecidas).
   - Devuelve un dict con:
       {
         "meta": {...},
         "observations": [
            {"date": "YYYY-MM-DD", "value": x, "status": "...", "pct": ..., "yoy_pct": ...},
            ...
         ],
         "observations_raw": [...]
       }

   La clave de Redis incluye:
       - series_id
       - target_frequency
       - agg

2) get_series_from_redis(...)
   - Recupera desde Redis lo almacenado por get_series_api_rest_bcch.
   - Si no existe en Redis:
       * Por defecto, puede llamar a get_series_api_rest_bcch para poblar (si use_fallback=True).
       * O devolver None si use_fallback=False.
   - Permite filtrar por rango de fechas (fecha inicio / fecha fin).
   - Devuelve el mismo tipo de estructura que get_series_api_rest_bcch, pero filtrada al período.


"""

import json
import re
import datetime
import os
from typing import Optional, Dict, Any, List

import pandas as pd
import requests
from dateutil import parser as dateparser
from urllib.parse import urlencode, unquote
import inspect as _inspect
import os as _os
from config import LOG_EXPOSE_API_LINKS
# import redis -> hacerlo opcional para no fallar en import
try:
    import redis  # type: ignore
except Exception:  # redis no instalado o problemático
    redis = None  # type: ignore[assignment]
# relativedelta para ajustar rangos según frecuencia
try:
    from dateutil.relativedelta import relativedelta  # type: ignore
except Exception:
    relativedelta = None  # type: ignore[assignment]

from config import BCCH_USER, BCCH_PASS, LOG_LEVEL, get_settings
# Flag opcional para habilitar/deshabilitar cacheo
USE_REDIS_CACHE = os.getenv("USE_REDIS_CACHE", "1").lower() in {"1", "true", "yes", "y", "on"}
REDIS_SERIES_TTL = os.getenv("REDIS_SERIES_TTL")
# from logger import get_logger, Phase  -> fallback si no existe logger.py
try:
    from logger import get_logger, Phase  # type: ignore
except Exception:
    import logging, contextlib, time as _time, os as _os, datetime as _dt

    def get_logger(name: str, level: str = "INFO") -> logging.Logger:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        root = _os.path.abspath(_os.path.dirname(__file__))
        log_dir = _os.path.join(root, "logs")
        _os.makedirs(log_dir, exist_ok=True)
        fname = _os.path.join(log_dir, f"pibot_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = logging.FileHandler(fname, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        return logger

    class Phase(contextlib.ContextDecorator):
        def __init__(self, logger: logging.Logger, name: str, extra: Optional[dict] = None):
            self.logger = logger
            self.name = name
            self.extra = extra or {}
            self._t0: Optional[float] = None

        def __enter__(self):
            self._t0 = _time.perf_counter()
            self.logger.info(f"[FASE] start: {self.name} | {self.extra}")
            return self

        def __exit__(self, exc_type, exc, tb):
            t1 = _time.perf_counter()
            if exc:
                self.logger.error(f"[FASE] error: {self.name} ({t1 - (self._t0 or t1):.3f}s) | error={exc}")
            else:
                self.logger.info(f"[FASE] end: {self.name} ({t1 - (self._t0 or t1):.3f}s)")
            return False

try:
    from logger import get_logger as _project_get_logger  # type: ignore
except Exception:
    _project_get_logger = None  # type: ignore

_DEF_LOGGER_NAME = __name__
if _project_get_logger:
    logger = _project_get_logger(_DEF_LOGGER_NAME, level=LOG_LEVEL)
else:
    # Reusar logger del orquestador (archivo único por sesión)
    try:
        import orchestrator as _orch_mod
        if hasattr(_orch_mod, 'logger'):
            logger = _orch_mod.logger  # type: ignore[assignment]
        else:
            logger = get_logger(_DEF_LOGGER_NAME, level=LOG_LEVEL)
    except Exception:
        logger = get_logger(_DEF_LOGGER_NAME, level=LOG_LEVEL)

# Eliminar lógica de duplicación de master: no agregar nuevos handlers
try:
    current_file = getattr(logger, '_session_log_path', None)
    if not current_file:
        # Intentar detectar baseFilename del handler
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                current_file = getattr(h, 'baseFilename', None)
                break
    if current_file:
        logger.info(f"[LOG_REUSE] Using single session log file {current_file} (get_series)")
except Exception as _e_master:
    logger.error(f"[LOG_REUSE] No se pudo confirmar handler único: {_e_master}")

BCCH_BASE = "https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx"

# ---------------------------------------------------------------------------
# Configuración Redis
# ---------------------------------------------------------------------------


def _get_redis_client() -> Optional[Any]:
    """
    Devuelve un cliente Redis configurado a partir de config.py.

    Se asume que en config.py existe, por ejemplo:
        REDIS_URL = "redis://localhost:6379/0"

    Si no está configurado, devuelve None y el código funciona sin cacheo.
    """

    if not USE_REDIS_CACHE:
        logger.info("Cache Redis deshabilitado por USE_REDIS_CACHE=0/false.")
        return None

    if redis is None:
        logger.warning("Paquete 'redis' no instalado; cacheo deshabilitado.")
        return None

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        try:
            from config import REDIS_URL as _cfg_url  # lazy import para compatibilidad
            redis_url = _cfg_url
        except Exception:
            redis_url = None

    if not redis_url:
        logger.warning("REDIS_URL no configurado en config.py; cacheo deshabilitado.")
        return None

    try:
        client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=2,
        )
        # Pequeña prueba de conexión
        client.ping()
        return client
    except Exception as e:
        logger.error(f"No se pudo conectar a Redis con REDIS_URL={redis_url!r}: {e}")
        return None


_redis_client: Optional[Any] = _get_redis_client()
_redis_status_logged: bool = False


def _ensure_redis_client() -> Optional[Any]:
    """Inicializa el cliente Redis bajo demanda si no existe o falló antes."""
    global _redis_client
    global _redis_status_logged
    if _redis_client is None:
        _redis_client = _get_redis_client()
        if _redis_client is not None:
            logger.info("[redis] Cliente inicializado para cacheo de series.")
            _redis_status_logged = True
    elif not _redis_status_logged:
        try:
            _redis_client.ping()
            logger.info("[redis] Cliente ya disponible para cacheo de series.")
        except Exception as _e:
            logger.warning(f"[redis] Cliente presente pero ping falló: {_e}")
        _redis_status_logged = True
    return _redis_client


def _make_cache_key(
    series_id: str,
    firstdate: Optional[str],
    lastdate: Optional[str],
    target_frequency: Optional[str],
    agg: str,
) -> str:
    """
    Construye la clave de Redis para una combinación (serie + fechas + frecuencia + agg).
    """
    fdate = firstdate or "auto"
    ldate = lastdate or "auto"
    freq = (target_frequency or "orig").upper()
    agg = (agg or "avg").lower()
    return f"bcch:series:{series_id}:{fdate}:{ldate}:{freq}:{agg}"


# ---------------------------------------------------------------------------
# Trazabilidad de cálculos (decorador)
# ---------------------------------------------------------------------------

from functools import wraps
from typing import Any as _Any

def _summarize_arg(name: str, val: _Any):
    try:
        if isinstance(val, pd.DataFrame):
            return {"type": "DataFrame", "shape": list(val.shape)}
        if isinstance(val, dict):
            out = {"type": "dict"}
            if "observations" in val and isinstance(val["observations"], list):
                out["observations_len"] = len(val["observations"])  # type: ignore[index]
            if "meta" in val and isinstance(val["meta"], dict):
                out["meta_freq"] = (val.get("meta", {}) or {}).get("freq_effective")
            return out
        if isinstance(val, list):
            return {"type": "list", "len": len(val)}
        return val
    except Exception:
        return str(val)[:200]


def calc_trace(fn):
    @wraps(fn)
    def _wrap(*args, **kwargs):
        try:
            args_summary = {f"arg{idx}": _summarize_arg(f"arg{idx}", a) for idx, a in enumerate(args)}
            args_summary.update({k: _summarize_arg(k, v) for k, v in kwargs.items()})
            logger.info(f"[CALC_TRACE_ENTER] file=get_series.py func={fn.__name__} args={args_summary}")
        except Exception:
            pass
        result = fn(*args, **kwargs)
        try:
            if isinstance(result, str):
                res_sum = {"type": "str", "lines": result.count("\n") + 1 if result else 0}
            elif isinstance(result, dict):
                res_sum = {"type": "dict", "keys": list(result.keys())[:8]}
                try:
                    res_sum["observations_len"] = len(result.get("observations", []) or [])
                except Exception:
                    pass
            elif isinstance(result, pd.DataFrame):
                res_sum = {"type": "DataFrame", "shape": list(result.shape)}
            else:
                res_sum = type(result).__name__
            logger.info(f"[CALC_TRACE_EXIT] file=get_series.py func={fn.__name__} result={res_sum}")
        except Exception:
            pass
        return result
    return _wrap


def _caller_info_for_log(skip_filename_suffix: str = "get_series.py") -> Dict[str, Any]:
    """Extrae archivo y función del primer frame fuera de este módulo para trazas.

    Retorna dict con keys: file, func, module. Si no se encuentra, usa current.
    """
    try:
        stack = _inspect.stack()
        for fr in stack[1:]:
            fname = _os.path.basename(fr.filename)
            if not fname.endswith(skip_filename_suffix):
                return {
                    "file": fname,
                    "func": fr.function,
                    "module": _inspect.getmodule(fr.frame).__name__ if _inspect.getmodule(fr.frame) else None,
                }
        # Fallback: usar el primer frame
        fr0 = stack[1]
        return {
            "file": _os.path.basename(fr0.filename),
            "func": fr0.function,
            "module": _inspect.getmodule(fr0.frame).__name__ if _inspect.getmodule(fr0.frame) else None,
        }
    except Exception:
        return {"file": skip_filename_suffix, "func": "<unknown>", "module": None}
# Utilidades para fechas y frecuencias
# ---------------------------------------------------------------------------


def _infer_freq_from_code(series_id: str) -> str:
    if not series_id:
        return "U"
    last = series_id.strip().split(".")[-1].upper()
    return last if last in {"D", "M", "Q", "A"} else "U"


def _parse_index_date(s: str) -> datetime.date:
    if not s:
        raise ValueError("indexDateString vacío")
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return datetime.date.fromisoformat(s)
    return dateparser.parse(s, dayfirst=True).date()


@calc_trace
def _resample(
    df: pd.DataFrame,
    target_freq: Optional[str],
    agg: str,
    original_freq: str,
) -> pd.DataFrame:
    if not target_freq or target_freq.upper() in {"", original_freq.upper()}:
        return df

    target = target_freq.upper()
    if target not in {"D", "M", "Q", "A"}:
        return df

    # Use 'ME' for month-end to avoid pandas deprecation warning
    rule_map = {"D": "D", "M": "ME", "Q": "Q-DEC", "A": "A-DEC"}
    rule = rule_map[target]

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("date")

    if agg == "sum":
        grouped = df.resample(rule).sum(numeric_only=True)
    elif agg == "last":
        grouped = df.resample(rule).last()
    elif agg == "first":
        grouped = df.resample(rule).first()
    else:
        grouped = df.resample(rule).mean(numeric_only=True)

    grouped = grouped.dropna(how="all").reset_index()

    # Para frecuencia trimestral Q: asegurar que periodo anterior (mismo trimestre año previo) exista
    # Esto se maneja luego en cálculo yoy, pero verificamos aquí que las fechas sean fin de período
    return grouped


def _normalize_observations(obs: List[Dict[str, Any]]) -> pd.DataFrame:
    import pandas as pd
    rows = []
    for o in obs:
        # Normalizar fecha: usar indexDateString si existe, sino date
        date_raw = o.get("indexDateString") or o.get("date")
        d = None
        if date_raw:
            try:
                d = _parse_index_date(date_raw)
            except Exception:
                d = None
        v_raw = o.get("value", None)
        try:
            v = float(str(v_raw).replace(",", ".")) if v_raw is not None else None
        except Exception:
            v = None
        rows.append(
            {
                "date": d,
                "value": v,
                "status": o.get("statusCode", ""),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["value", "date"]).sort_values("date")
    # Asegurar que la columna 'date' sea datetime para permitir resampleo
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Cálculo de variaciones (PCT y YOY_PCT)
# ---------------------------------------------------------------------------


@calc_trace
def _compute_variations(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Agrega columnas:
      - pct      → variación % respecto al período anterior.
      - yoy_pct  → variación % respecto al mismo período del año anterior.
    """
    if df.empty:
        df["pct"] = []
        df["yoy_pct"] = []
        return df

    df = df.sort_values("date").copy()

    # Variación respecto del período anterior
    df["pct"] = df["value"].pct_change() * 100.0

    # Definir lag para YoY según frecuencia
    freq = (freq or "").upper()
    if freq == "M":
        lag = 12
    elif freq in {"Q", "T"}:
        lag = 4
    elif freq == "A":
        lag = 1
    elif freq == "D":
        lag = 365  # aproximación
    else:
        lag = None

    # Cálculo robusto de variación anual: emparejar por trimestre y año
    import numpy as np
    if freq in {"Q", "T"}:
        # Extraer año y trimestre
        df["_year"] = df["date"].apply(lambda x: int(str(x)[:4]))
        df["_month"] = df["date"].apply(lambda x: int(str(x)[5:7]))
        df["_quarter"] = df["_month"].apply(lambda m: ((m - 1) // 3) + 1)
        yoy_list = []
        for idx, row in df.iterrows():
            y, q = row["_year"], row["_quarter"]
            prev = df[(df["_year"] == y - 1) & (df["_quarter"] == q)]
            if not prev.empty and prev.iloc[0]["value"] != 0:
                yoy = (row["value"] / prev.iloc[0]["value"] - 1.0) * 100.0
            else:
                yoy = np.nan
            yoy_list.append(yoy)
        df["yoy_pct"] = yoy_list
        df.drop(["_year", "_month", "_quarter"], axis=1, inplace=True)
    elif lag is not None and lag > 0:
        prev = df["value"].shift(lag)
        df["yoy_pct"] = (df["value"] / prev - 1.0) * 100.0
    else:
        df["yoy_pct"] = None

    return df


# ---------------------------------------------------------------------------
# Llamada a BCCh + almacenamiento en Redis
# ---------------------------------------------------------------------------


def get_series_api_rest_bcch(
    series_id: str,
    target_date: Optional[str] = None,
    target_frequency: Optional[str] = None,
    agg: str = "avg",
) -> Dict[str, Any]:
    """
    Llama a la API REST del BCCh, calcula variaciones y almacena el resultado en Redis.

    Parámetros
    ----------
    series_id : str
        Identificador de la serie BCCh (ej: "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M").
        Para IMACEC, PIB y PIB regional, normalmente se obtienen desde los JSON
        de configuración por defecto (config_default.json).
    target_date : str, opcional
        Fecha de referencia (YYYY-MM-DD) para calcular flags de posición; la serie se trae completa.
    target_frequency : str, opcional
        Frecuencia objetivo: "D", "M", "Q" o "A". Si None, se mantiene la original.
    agg : str
        Tipo de agregación en caso de remuestreo: "avg", "sum", "first", "last".

    Devuelve
    --------
    dict con claves:
        - "meta": {...}
        - "observations": lista de dicts con date, value, status, pct, yoy_pct
        - "observations_raw": observaciones tal como vinieron del BCCh
    """
    global _redis_client
    if not BCCH_USER or not BCCH_PASS:
        raise RuntimeError("BCCH_USER/BCCH_PASS no configurados (.env)")

    # Ignorar firstdate: siempre traemos la serie completa. target_date se usa solo para flags.
    target_date = None if target_date in (None, "", "auto") else target_date

    params = {
        "user": BCCH_USER,
        "pass": BCCH_PASS,
        "function": "GetSeries",
        "timeseries": series_id,
    }
    # No se envían firstdate/lastdate: se trae la serie completa.

    headers = {"Accept": "application/json"}
    log_params = {**params}
    log_params["pass"] = "***"

    # Construir URL de request para log (opcionalmente expuesta) con traza de origen
    try:
        caller = _caller_info_for_log()
        args_summary = {
            "series_id": series_id,
            "firstdate": "auto",  # siempre auto
            "lastdate": target_date or "auto",
            "target_frequency": (target_frequency or None),
            "agg": agg,
        }
        url_masked = f"{BCCH_BASE}?{urlencode(log_params)}"
        # URL expuesta: versión codificada y versión legible (sin %xx)
        url_plain_encoded = f"{BCCH_BASE}?{urlencode(params)}"
        url_plain_readable = unquote(url_plain_encoded)
        # Etiqueta de dominio por inspección del series_id
        sid_up = (series_id or "").upper()
        if "PIB" in sid_up:
            tag = "PIB"
        elif "IMC" in sid_up or "IMACEC" in sid_up:
            tag = "IMACEC"
        else:
            tag = "SERIE"
        # Registrar siempre la versión enmascarada (sin credenciales)
        logger.info(
            f"[API_REQUEST_URL:{tag}] caller_file={caller.get('file')} caller_func={caller.get('func')} "
            f"params={args_summary} url={url_masked}"
        )
        # Opcional: exponer link plano completo sólo si está habilitado explícitamente
        if LOG_EXPOSE_API_LINKS:
            logger.info(
                f"[API_REQUEST_URL_PLAIN:{tag}] caller_file={caller.get('file')} caller_func={caller.get('func')} "
                f"params={args_summary} url={url_plain_readable}"
            )

    except Exception as _e_url:
        logger.error(f"[API_REQUEST_URL_ERROR] No se pudo construir URL de log: {_e_url}")

    # Cache sin depender de fechas: siempre se guarda la serie completa para la combinación serie+freq+agg
    cache_key = _make_cache_key(series_id, None, None, target_frequency, agg)
    logger.info(
        f"[get_series_api_rest_bcch] Consultando serie de datos | "
        f"series_id='{series_id}' | cache_key='{cache_key}' | params={log_params}"
    )

    # --- Llamada a API BCCh ---
    with Phase(
        logger,
        "Fase 3: Llamada a BCCh",
        {
            "series_id": series_id,
            "firstdate": None,
            "lastdate": None,
            "target_frequency": target_frequency,
            "agg": agg,
        },
    ):
        r = requests.get(BCCH_BASE, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("Codigo") != 0:
            desc = data.get("Descripcion", "Error desconocido")
            raise RuntimeError(f"BCCh API error: {desc}")

        series = data.get("Series", {})
        obs = series.get("Obs", [])

        logger.info(
            f"[get_series_api_rest_bcch] Observaciones recibidas: {len(obs)} | "
            f"descripEsp='{series.get('descripEsp', '')}'"
        )

        observations_raw = [
            {
                "indexDateString": o.get("indexDateString", ""),
                "value": o.get("value", None),
                "status": o.get("statusCode", ""),
            }
            for o in obs
        ]

        df = _normalize_observations(obs)
        original_freq = _infer_freq_from_code(series_id)

        with Phase(
            logger,
            "Fase 3.1: Remuestreo",
            {
                "series_id": series_id,
                "original_frequency": original_freq,
                "target_frequency": target_frequency,
                "agg": agg,
            },
        ):
            df_out = _resample(df, target_frequency, agg, original_freq)
            logger.info(
                f"[get_series_api_rest_bcch] Filas salida remuestreo: {len(df_out)}"
            )

        # Determinar frecuencia efectiva al final del proceso
        effective_freq = (target_frequency or original_freq or "U").upper()

        # Calcular variaciones
        with Phase(
            logger,
            "Fase 3.2: Cálculo variaciones PCT / YOY_PCT",
            {"series_id": series_id, "frequency": effective_freq},
        ):
            df_enriched = _compute_variations(df_out, effective_freq)
            logger.info(
                f"[get_series_api_rest_bcch] Filas con variaciones calculadas: {len(df_enriched)}"
            )

        meta = {
            "series_id": series.get("seriesId", series_id),
            "descripEsp": series.get("descripEsp", ""),
            "descripIng": series.get("descripIng", ""),
            "original_frequency": original_freq,
            "target_frequency": target_frequency or original_freq,
            "freq_effective": effective_freq,
            "agg": agg,
            "firstdate": "primer dato disponible",
            "lastdate": target_date or "último dato disponible",
            "target_date": target_date or "auto",
        }

        observations = []
        for _, row in df_enriched.iterrows():
            d = row["date"]
            v = row["value"]
            pct = row.get("pct", None)
            yoy = row.get("yoy_pct", None)
            observations.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "value": None if pd.isna(v) else float(v),
                    "status": "",  # status original se pierde en el remuestreo
                    "pct": None if pd.isna(pct) else float(pct),
                    "yoy_pct": None if pd.isna(yoy) else float(yoy),
                }
            )

        result = {
            "meta": meta,
            "observations": observations,
            "observations_raw": observations_raw,
        }

    # --- Verificación de lastdate vs rango disponible (primera y última fecha) ---
    try:
        # Fechas disponibles en las observaciones devueltas
        first_available_str = None
        last_available_str = None
        if observations:
            dates_iter = [o.get("date") for o in observations if isinstance(o.get("date"), str) and o.get("date")]
            if dates_iter:
                first_available_str = min(dates_iter)
                last_available_str = max(dates_iter)

        # Clasificar la posición de lastdate respecto del rango disponible
        lastdate_position = "unknown"
        is_auto = target_date in (None, "", "auto")
        if is_auto:
            lastdate_position = "auto"
        elif first_available_str and last_available_str:
            try:
                d_first = dateparser.parse(first_available_str).date()
                d_latest = dateparser.parse(last_available_str).date()
                d_req = dateparser.parse(target_date).date()  # type: ignore[arg-type]
                if d_req > d_latest:
                    lastdate_position = "gt_latest"
                elif d_req == d_latest:
                    lastdate_position = "eq_latest"
                elif d_req >= d_first and d_req < d_latest:
                    lastdate_position = "within_range"
                elif d_req < d_first:
                    lastdate_position = "lt_first"
            except Exception:
                # Fallback lexicográfico si el parse falla
                if str(target_date) > str(last_available_str):
                    lastdate_position = "gt_latest"
                elif str(target_date) == str(last_available_str):
                    lastdate_position = "eq_latest"
                elif str(first_available_str) <= str(target_date) < str(last_available_str):
                    lastdate_position = "within_range"
                elif str(target_date) < str(first_available_str):
                    lastdate_position = "lt_first"

        # Booleans derivados
        lastdate_ge_latest = lastdate_position in {"eq_latest", "gt_latest"}
        lastdate_lt_first = lastdate_position == "lt_first"
        lastdate_within_range = lastdate_position == "within_range"

        # Anotar en meta
        meta_ref = result.setdefault("meta", {})
        meta_ref["first_available_date"] = first_available_str or ""
        meta_ref["last_available_date"] = last_available_str or ""
        meta_ref["lastdate_position"] = lastdate_position
        meta_ref["lastdate_ge_latest"] = bool(lastdate_ge_latest)
        meta_ref["lastdate_within_range"] = bool(lastdate_within_range)
        meta_ref["lastdate_lt_first"] = bool(lastdate_lt_first)

        logger.info(
            "[get_series_api_rest_bcch] lastdate check | "
            f"first_available={first_available_str} last_available={last_available_str} "
            f"requested={target_date or 'auto'} position={lastdate_position}"
        )
    except Exception as _e_lastchk:
        logger.debug(f"[get_series_api_rest_bcch] No se pudo verificar lastdate vs último dato: {_e_lastchk}")

    # --- Log de salida a test/log.text ---
    try:
        sample_api = ", ".join(
            f"{r['date']}={r['value']}" for r in observations[:5]
        )
        _append_test_log(
            (
                f"source=api | serie={series_id} | rango=auto→{target_date or 'auto'} | "
                f"freq={meta.get('freq_effective', '')} | agg={agg} | rows={len(observations)} | sample=[{sample_api}]"
            )
        )
    except Exception as _e:
        logger.error(f"Fallo escribiendo log de test para API: {_e}")

    # --- Almacenar en Redis ---
    client = _ensure_redis_client()
    if client is not None:
        try:
            payload = json.dumps(result, ensure_ascii=False)
            client.set(cache_key, payload)
            if REDIS_SERIES_TTL and REDIS_SERIES_TTL.isdigit():
                try:
                    client.expire(cache_key, int(REDIS_SERIES_TTL))
                except Exception:
                    logger.warning(f"[get_series_api_rest_bcch] No se pudo aplicar TTL a key='{cache_key}'")
            logger.info(
                f"[get_series_api_rest_bcch] Serie almacenada en Redis | key='{cache_key}'"
            )
        except Exception as e:
            logger.error(
                f"[get_series_api_rest_bcch] Error almacenando en Redis key='{cache_key}': {e}"
            )
            _redis_client = None  # permitir reintentos en llamadas futuras
    else:
        logger.info(
            "[get_series_api_rest_bcch] Redis no disponible; no se cachea la serie."
        )

    return result


# ---------------------------------------------------------------------------
# Lectura desde Redis con filtro por fechas
# ---------------------------------------------------------------------------


def get_series_from_redis(
    series_id: str,
    firstdate: Optional[str] = None,
    lastdate: Optional[str] = None,
    target_frequency: Optional[str] = None,
    agg: str = "avg",
    use_fallback: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Recupera la serie desde Redis (ya enriquecida con pct y yoy_pct) y filtra
    por rango de fechas.

    Parámetros
    ----------
    series_id : str
        Id de la serie BCCh.
    firstdate : str, opcional
        Fecha inicial (YYYY-MM-DD). Si None, no se aplica filtro inferior.
    lastdate : str, opcional
        Fecha final (YYYY-MM-DD). Si None, no se aplica filtro superior.
    target_frequency : str, opcional
        Frecuencia objetivo usada cuando se cacheó.
    agg : str
        Tipo de agregación usado cuando se cacheó.
    use_fallback : bool
        Si True y la clave no existe en Redis, se llama a get_series_api_rest_bcch
        para poblarla y se devuelve el resultado (ya filtrado).
        Si False y la clave no existe, devuelve None.

    Devuelve
    --------
    dict o None
        Mismo formato que get_series_api_rest_bcch, pero con "observations"
        filtradas al período solicitado.
    """
    global _redis_client
    fd = None if firstdate in (None, "", "auto") else firstdate
    ld = None if lastdate in (None, "", "auto") else lastdate

    # Cache siempre con fechas auto (serie completa); fd/ld solo para filtro posterior
    cache_key = _make_cache_key(series_id, None, None, target_frequency, agg)
    logger.info(
        f"[get_series_from_redis] Intentando recuperar serie desde Redis | "
        f"series_id='{series_id}' | cache_key='{cache_key}'"
    )

    client = _ensure_redis_client()
    if client is None:
        logger.warning(
            "[get_series_from_redis] Redis no disponible; "
            "usando fallback a get_series_api_rest_bcch."
        )
        return (
            get_series_api_rest_bcch(
                series_id=series_id,
                target_date=ld,
                target_frequency=target_frequency,
                agg=agg,
            )
            if use_fallback
            else None
        )

    try:
        raw = client.get(cache_key)
    except Exception as e:
        logger.error(
            f"[get_series_from_redis] Error obteniendo clave '{cache_key}' desde Redis: {e}"
        )
        _redis_client = None  # forzar reintento en próximas peticiones
        if not use_fallback:
            return None
        return get_series_api_rest_bcch(
            series_id=series_id,
            target_date=ld,
            target_frequency=target_frequency,
            agg=agg,
        )
    if raw is None:
        logger.info(
            f"[get_series_from_redis] Clave no encontrada en Redis | key='{cache_key}'"
        )
        if not use_fallback:
            return None
        # Poblar Redis llamando a la API
        return get_series_api_rest_bcch(
            series_id=series_id,
            target_date=ld,
            target_frequency=target_frequency,
            agg=agg,
        )

    try:
        data = json.loads(raw)
        logger.info(
            f"[get_series_from_redis] Cache hit | key='{cache_key}' | rows={len((data or {}).get('observations', []) or [])}"
        )
    except Exception as e:
        logger.error(
            f"[get_series_from_redis] Error parseando JSON desde Redis | key='{cache_key}' | error={e}"
        )
        if not use_fallback:
            return None
        return get_series_api_rest_bcch(
            series_id=series_id,
            target_date=ld,
            target_frequency=target_frequency,
            agg=agg,
        )

    obs = data.get("observations", [])

    # Recalcular flags de posición según el target_date (ld)
    try:
        first_available_str = None
        last_available_str = None
        if obs:
            dates_iter = [o.get("date") for o in obs if isinstance(o.get("date"), str) and o.get("date")]
            if dates_iter:
                first_available_str = min(dates_iter)
                last_available_str = max(dates_iter)

        lastdate_position = data.get("meta", {}).get("lastdate_position", "unknown")
        if ld:
            try:
                d_first = dateparser.parse(first_available_str).date() if first_available_str else None
                d_latest = dateparser.parse(last_available_str).date() if last_available_str else None
                d_req = dateparser.parse(ld).date()
                if d_latest and d_req > d_latest:
                    lastdate_position = "gt_latest"
                elif d_latest and d_req == d_latest:
                    lastdate_position = "eq_latest"
                elif d_first and d_latest and d_req >= d_first and d_req < d_latest:
                    lastdate_position = "within_range"
                elif d_first and d_req < d_first:
                    lastdate_position = "lt_first"
            except Exception:
                pass

        meta_ref = data.setdefault("meta", {})
        meta_ref["first_available_date"] = first_available_str or meta_ref.get("first_available_date", "")
        meta_ref["last_available_date"] = last_available_str or meta_ref.get("last_available_date", "")
        meta_ref["lastdate_position"] = lastdate_position
        if ld:
            meta_ref["target_date"] = ld
    except Exception:
        pass
    if not fd and not ld:
        # Log simple de lectura completa
        try:
            sample_redis_all = ", ".join(
                f"{r['date']}={r['value']}" for r in obs[:5]
            )
            _append_test_log(
                (
                    f"source=redis | serie={series_id} | rango=cache | "
                    f"freq={data.get('meta', {}).get('freq_effective', '')} | agg={agg} | "
                    f"rows={len(obs)} | sample=[{sample_redis_all}]"
                )
            )
        except Exception as _e:
            logger.error(f"Fallo escribiendo log de test para Redis(all): {_e}")
        return data

    # Filtrado por rango de fechas
    def _parse_date_str(s: str) -> datetime.date:
        return datetime.date.fromisoformat(s)

    fd_date = _parse_date_str(fd) if fd else None
    ld_date = _parse_date_str(ld) if ld else None

    filtered_obs = []
    for o in obs:
        d = _parse_date_str(o["date"])
        if fd_date and d < fd_date:
            continue
        if ld_date and d > ld_date:
            continue
        filtered_obs.append(o)

    data["observations"] = filtered_obs
    # Actualizar meta para reflejar el nuevo rango
    meta = data.get("meta", {})
    if fd:
        meta["firstdate"] = fd
    if ld:
        meta["lastdate"] = ld
    data["meta"] = meta

    logger.info(
        f"[get_series_from_redis] Observaciones devueltas tras filtro: {len(filtered_obs)}"
    )

    # Log de lectura filtrada
    try:
        sample_redis = ", ".join(
            f"{r['date']}={r['value']}" for r in filtered_obs[:5]
        )
        _append_test_log(
            (
                f"source=redis | serie={series_id} | rango={fd or 'auto'}→{ld or 'auto'} | "
                f"freq={data.get('meta', {}).get('freq_effective', '')} | agg={agg} | "
                f"rows={len(filtered_obs)} | sample=[{sample_redis}]"
            )
        )
    except Exception as _e:
        logger.error(f"Fallo escribiendo log de test para Redis(filtered): {_e}")

    return data


# ---------------------------------------------------------------------------
# Wrapper de conveniencia: salida por frecuencia y tipo de cálculo
# y logging a test/log.txt con timestamp
# ---------------------------------------------------------------------------
def get_current_test_log_file() -> str:
    """Compat: antes escribía a un archivo separado; ahora reusa el logger principal."""
    return ""

def _append_test_log(message: str) -> None:
    """En lugar de crear archivos adicionales, registramos en el logger principal."""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    logger.info(f"[TEST_LOG] {ts} {message}")


@calc_trace
def fetch_series_with_calc(
    series_id: str,
    firstdate: Optional[str] = None,
    lastdate: Optional[str] = None,
    frequency: Optional[str] = None,
    calc_type: str = "original",
    agg: str = "avg",
) -> Dict[str, Any]:
    """
    Obtiene una serie del BCCh y devuelve la salida según el tipo de cálculo solicitado.

    Parámetros:
      - series_id: código BCCh de la serie.
      - firstdate / lastdate: rango YYYY-MM-DD (o None/"auto").
      - frequency: "M" (mensual), "Q" (trimestral), "A" (anual) o "D".
      - calc_type: "original" (valor), "yoy"/"YPCT" (variación interanual), "mom"/"PCT" (variación periodo a periodo).
      - agg: agregación para remuestreo (default "avg").

    Devuelve un dict con meta y observations. Mantiene los campos originales
    (value, pct, yoy_pct) y agrega "selected" con el valor elegido según calc_type.
    Además, escribe un registro en test/log_<timestamp>.text con timestamp y un resumen.
    """
    # Normalizar frecuencia y aceptar alias en español
    freq_map = {
        "D": "D", "DIARIA": "D", "DIARIO": "D",
        "M": "M", "MENSUAL": "M",
        "Q": "Q", "TRIMESTRAL": "Q", "TRIMESTRE": "Q",
        "A": "A", "ANUAL": "A", "ANIO": "A", "AÑO": "A",
    }
    freq_key = (frequency or "").strip().upper()
    freq = freq_map.get(freq_key, None)
    if frequency and not freq:
        logger.warning(f"frequency inválida '{frequency}', usando la original de la serie")

    # Determinar calc_type normalizado y métrica
    metric_key = "value"
    raw_calc = (calc_type or "original").strip().lower()
    calc_type_norm = "original"
    if any(k in raw_calc for k in ["ypct", "yoy", "interanual", "año", "anio", "anual anterior", "año anterior", "anio anterior"]):
        metric_key = "yoy_pct"
        calc_type_norm = "ypct"
    elif any(k in raw_calc for k in ["pct", "mom", "mensual", "mes", "mes a mes", "periodo a periodo", "m/m"]):
        metric_key = "pct"
        calc_type_norm = "pct"

    # Título descriptivo del cálculo en el log
    _append_test_log(
        f"Calculo de serie {series_id} con frecuencia {freq or 'original'} en formato {calc_type_norm.upper()}"
    )

    # Ajustar rango de fetch para garantizar cálculo (prefetch de períodos previos)
    fd_fetch, ld_fetch = firstdate, lastdate
    try:
        if firstdate:
            fd_dt = datetime.date.fromisoformat(firstdate)
            if calc_type_norm == "ypct":
                # Necesita mismo período del año anterior
                if relativedelta is not None:
                    if (freq or "M") == "M":
                        fd_fetch = (fd_dt - relativedelta(years=1)).isoformat()
                    elif (freq or "M") == "Q":
                        fd_fetch = (fd_dt - relativedelta(years=1)).isoformat()
                    elif (freq or "M") == "A":
                        fd_fetch = (fd_dt - relativedelta(years=1)).isoformat()
                    elif (freq or "M") == "D":
                        fd_fetch = (fd_dt - datetime.timedelta(days=365)).isoformat()
                else:
                    # Fallback básico
                    fd_fetch = (fd_dt - datetime.timedelta(days=365)).isoformat()
            elif calc_type_norm == "pct":
                # Necesita período inmediatamente anterior
                if relativedelta is not None:
                    if (freq or "M") == "M":
                        fd_fetch = (fd_dt - relativedelta(months=1)).isoformat()
                    elif (freq or "M") == "Q":
                        fd_fetch = (fd_dt - relativedelta(months=3)).isoformat()
                    elif (freq or "M") == "A":
                        fd_fetch = (fd_dt - relativedelta(years=1)).isoformat()
                    elif (freq or "M") == "D":
                        fd_fetch = (fd_dt - datetime.timedelta(days=1)).isoformat()
                else:
                    fd_fetch = (fd_dt - datetime.timedelta(days=31)).isoformat()
    except Exception as _adj_e:
        logger.warning(f"No fue posible ajustar el rango para calc={calc_type_norm}: {_adj_e}")

    data = get_series_api_rest_bcch(
        series_id=series_id,
        target_date=lastdate,
        target_frequency=freq,
        agg=agg,
    )

    obs_in = data.get("observations", [])
    obs_out: List[Dict[str, Any]] = []
    for o in obs_in:
        selected_val = o.get(metric_key, None)
        obs_out.append({**o, "selected": selected_val})

    # Filtrar al rango solicitado (primario), si corresponde
    def _in_range(date_str: str) -> bool:
        try:
            d = datetime.date.fromisoformat(date_str)
            if firstdate and d < datetime.date.fromisoformat(firstdate):
                return False
            if lastdate and d > datetime.date.fromisoformat(lastdate):
                return False
            return True
        except Exception:
            return True

    if firstdate or lastdate:
        obs_out = [o for o in obs_out if _in_range(o.get("date", ""))]

    result = {
        "meta": {
            **data.get("meta", {}),
            "calc_type": calc_type_norm,
            "metric_selected": metric_key,
            # Reflejar el rango solicitado en meta
            "firstdate": firstdate or (data.get("meta", {}).get("firstdate")),
            "lastdate": lastdate or (data.get("meta", {}).get("lastdate")),
        },
        "observations": obs_out,
        "observations_raw": data.get("observations_raw", []),
    }

    # Logging a test/log con resumen
    sample = ", ".join(
        f"{r['date']}={r['selected']}" for r in obs_out[:5]
    )
    _append_test_log(
        (
            f"serie={series_id} | rango={firstdate or 'auto'}→{lastdate or 'auto'} | "
            f"freq={freq or data.get('meta', {}).get('freq_effective', '')} | "
            f"calc={calc_type_norm} | rows={len(obs_out)} | sample=[{sample}]"
        )
    )

    return result


# Alias público solicitado: función get_series con parámetro calc_type

def get_series(
    series_id: str,
    firstdate: Optional[str] = None,
    lastdate: Optional[str] = None,
    target_frequency: Optional[str] = None,
    agg: str = "avg",
    calc_type: str = "ORIGINAL",
) -> Dict[str, Any]:
    """
    API de alto nivel para obtener series con el tipo de cálculo deseado.

    calc_type admite: "ORIGINAL", "YPCT" (interanual), "PCT" (mes a mes).
    target_frequency admite: D/M/Q/A.
    """
    # Reusar la implementación existente aceptando alias en español/inglés
    # Detect if PIB and force quarterly frequency for correct YoY/periods
    force_quarterly = False
    sid_up = (series_id or "").upper()
    if "PIB" in sid_up and (target_frequency or "M").upper() == "M":
        force_quarterly = True
    freq_to_use = "Q" if force_quarterly else (target_frequency or None)
    return fetch_series_with_calc(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        frequency=freq_to_use,
        calc_type=(calc_type or "ORIGINAL"),
        agg=agg,
    )


def format_series_openai(data: Dict[str, Any], calc_type: str = "ORIGINAL") -> Dict[str, Any]:
    """
    Devuelve una estructura tabular compacta para UI/API según calc_type.

    - ORIGINAL: columns [date, value]
    - YPCT:     columns [date, yoy_pct]
    - PCT:      columns [<año-1>, <año>, variacion_pct]
      Empareja meses entre los dos últimos años disponibles (si es posible). Si no
      puede emparejar, retorna fallback [date, pct].
    """
    obs = data.get("observations", []) or []
    calc = (calc_type or "ORIGINAL").strip().upper()

    # Utilidades de redondeo a 1 decimal conservando None
    def _r1(x: Any) -> Any:
        try:
            return None if x is None else round(float(x), 1)
        except Exception:
            return x

    if calc == "YPCT":
        # Construir tabla emparejando meses entre último año y su anterior
        # Columnas: <año-1>, <año>, variacion_pct
        if not obs:
            return {"columns": ["date", "yoy_pct"], "rows": []}

        # Determinar año objetivo desde meta.lastdate o último dato
        meta = data.get("meta", {})
        lastdate = meta.get("lastdate")
        try:
            curr_year = int(str(lastdate).split("-")[0]) if lastdate else max(int(o.get("date","0000-01-01").split("-")[0]) for o in obs)
        except Exception:
            curr_year = max(int(o.get("date","0000-01-01").split("-")[0]) for o in obs)
        prev_year = curr_year - 1

        # Mapa año->mes->valor
        by_year_month: Dict[int, Dict[int, float]] = {prev_year: {}, curr_year: {}}
        for o in obs:
            try:
                y = int(str(o.get("date", "0000-01-01")).split("-")[0])
                m = int(str(o.get("date", "0000-01-01")).split("-")[1])
            except Exception:
                continue
            if y not in (prev_year, curr_year):
                continue
            v = o.get("value")
            if v is None:
                continue
            by_year_month.setdefault(y, {})[m] = float(v)

        months = sorted(set(by_year_month.get(prev_year, {}).keys()) & set(by_year_month.get(curr_year, {}).keys()))
        rows = []
        for m in months:
            pv = by_year_month[prev_year].get(m)
            cv = by_year_month[curr_year].get(m)
            if pv is None or cv is None:
                continue
            variacion = (cv / pv - 1.0) * 100.0 if pv else None
            rows.append({str(prev_year): _r1(pv), str(curr_year): _r1(cv), "variacion_pct": _r1(variacion)})

        if rows:
            return {"columns": [str(prev_year), str(curr_year), "variacion_pct"], "rows": rows}
        # Fallback si no hay meses emparejados
        rows = [{"date": o.get("date"), "yoy_pct": _r1(o.get("yoy_pct"))} for o in obs]
        return {"columns": ["date", "yoy_pct"], "rows": rows}

    if calc == "PCT":
        if not obs:
            return {"columns": ["date", "pct"], "rows": []}

        # Detectar últimos dos años presentes
        def _year(d: str) -> int:
            return int(str(d).split("-")[0])
        def _month(d: str) -> int:
            try:
                return int(str(d).split("-")[1])
            except Exception:
                return 0

        years = sorted({ _year(o.get("date", "0000-01-01")) for o in obs })
        if len(years) >= 2:
            curr_year = years[-1]
            prev_year = years[-2]
            by_year_month: Dict[int, Dict[int, float]] = {prev_year: {}, curr_year: {}}
            for o in obs:
                try:
                    y = _year(o.get("date", "0000-01-01"))
                    m = _month(o.get("date", "0000-01-01"))
                except Exception:
                    continue
                if y not in (prev_year, curr_year):
                    continue
                v = o.get("value")
                if v is None:
                    continue
                by_year_month.setdefault(y, {})[m] = float(v)

            months = sorted(set(by_year_month.get(prev_year, {}).keys()) & set(by_year_month.get(curr_year, {}).keys()))
            rows = []
            for m in months:
                pv = by_year_month[prev_year].get(m)
                cv = by_year_month[curr_year].get(m)
                if pv is None or cv is None:
                    continue
                variacion = (cv / pv - 1.0) * 100.0 if pv else None
                rows.append({str(prev_year): _r1(pv), str(curr_year): _r1(cv), "variacion_pct": _r1(variacion)})

            # Si hubo emparejamientos, usar el formato de años
            if rows:
                return {"columns": [str(prev_year), str(curr_year), "variacion_pct"], "rows": rows}

        # Fallback: devolver date + pct
        rows = [{"date": o.get("date"), "pct": _r1(o.get("pct"))} for o in obs]
        return {"columns": ["date", "pct"], "rows": rows}

    # ORIGINAL por defecto
    rows = [{"date": o.get("date"), "value": _r1(o.get("value"))} for o in obs]
    return {"columns": ["date", "value"], "rows": rows}


def get_series_for_api(
    series_id: str,
    firstdate: Optional[str] = None,
    lastdate: Optional[str] = None,
    target_frequency: Optional[str] = None,
    agg: str = "avg",
    calc_type: str = "ORIGINAL",
) -> Dict[str, Any]:
    """
    Conveniencia: obtiene la serie con el cálculo solicitado y devuelve además
    el formato compacto para consumo de APIs/UI.

    Retorna un dict con:
      - data: salida completa de get_series (incluye meta/observations)
      - formatted: estructura tabular compacta para el calc_type indicado
    """
    data = get_series(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=target_frequency,
        agg=agg,
        calc_type=calc_type,
    )
    formatted = format_series_openai(data, calc_type=calc_type)
    return {"data": data, "formatted": formatted}


@calc_trace
def build_year_comparison_table(data: Dict[str, Any], year: int) -> Dict[str, Any]:
    """Construye tabla de comparación Año anterior vs Año actual (solo periodos con dato del año actual).

    - Incluye únicamente los periodos (meses o trimestres) para los que existe dato en el año actual.
    - Para cada periodo agrega valor del año anterior si existe.
    - Calcula variación interanual (yoy_pct) si no viene en los datos.
    - Redondea la variación a 1 decimal (pero preserva valores originales para value_prev/value_curr).
    """
    obs = data.get('observations', []) or []
    prev_year = year - 1
    freq = (data.get('meta', {}) or {}).get('freq_effective', 'M').upper()

    from typing import Tuple
    def _quarter_and_label(dt: datetime.date) -> Tuple[str, str]:
        # Robustly assign quarter and label for quarterly data
        if freq in {"Q", "T"}:
            # If date is first of quarter, assign quarter accordingly
            month = dt.month
            if month in (1, 2, 3):
                q = 1
            elif month in (4, 5, 6):
                q = 2
            elif month in (7, 8, 9):
                q = 3
            else:
                q = 4
            label = f"{q}T {dt.year}"
            key = f"{dt.year}-Q{q}"
            return key, label
        meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        label = meses[dt.month - 1]
        key = f"{dt.year}-{dt.month:02d}"
        return key, label

    prev_map: Dict[str, Dict[str, Any]] = {}
    curr_map: Dict[str, Dict[str, Any]] = {}
    for r in obs:
        try:
            d = datetime.date.fromisoformat(r.get('date', '0000-01-01'))
        except Exception:
            continue
        key, label = _quarter_and_label(d)
        r['_period_label'] = label
        if d.year == prev_year:
            prev_map[key] = r
        elif d.year == year:
            curr_map[key] = r

    # Solo periodos presentes en el año actual
    def quarter_sort_key(k):
        # k is like '2025-Q3'
        if freq in {"Q", "T"} and '-Q' in k:
            y, q = k.split('-Q')
            return (int(y), int(q))
        # fallback for months
        if '-' in k:
            y, m = k.split('-')
            return (int(y), int(m))
        return (9999, 99)
    periods_curr = sorted(curr_map.keys(), key=quarter_sort_key)

    rows: List[Dict[str, Any]] = []
    for p in periods_curr:
        py = prev_map.get(p, {})
        cy = curr_map.get(p, {})
        pv = py.get('value')
        cv = cy.get('value')
        # Use label from current year if available, else from previous
        period_label = cy.get('_period_label') or py.get('_period_label') or p
        # Calculate YoY only if both values exist and are not None
        yoy = cy.get('yoy_pct')
        if yoy is None and (pv is not None and cv is not None and pv != 0):
            try:
                yoy = (float(cv) - float(pv)) / float(pv) * 100.0
            except Exception:
                yoy = None
        # Redondear variación a 1 decimal si existe
        if yoy is not None:
            try:
                yoy = round(float(yoy), 1)
            except Exception:
                pass
        rows.append({
            'period': period_label,
            'value_prev': pv,
            'value_curr': cv,
            'yoy_pct': yoy,
        })

    return {"year_prev": prev_year, "year_curr": year, "frequency": freq, "rows": rows}


@calc_trace
def build_year_comparison_table_text(data: Dict[str, Any], year: int) -> str:
    """Markdown de la comparación interanual truncada al último periodo disponible del año actual.

    Columnas:
    Mes | Año anterior | Año actual | Variación anual
    La variación se redondea a 1 decimal.
    """
    prev_year = year - 1
    table = build_year_comparison_table(data, year)
    freq = str(table.get("frequency") or (data.get("meta", {}) or {}).get("freq_effective") or "M").upper()
    if freq in {"Q", "T"}:
        period_label = "Trimestre"
    elif freq == "M":
        period_label = "Mes"
    else:
        period_label = "Periodo"
    header_lines = [
        f"Comparación {prev_year} vs {year}",
        f"{period_label} | Año anterior | Año actual | Variación anual",
        "---|---|---|---",
    ]
    body_lines: List[str] = []

    def _r1(x: Any) -> Any:
        try:
            return "" if x is None else round(float(x), 1)
        except Exception:
            return x

    for row in table.get('rows', []):
        period_lbl = row.get('period', '')
        pv = row.get('value_prev')
        cv = row.get('value_curr')
        yoy = row.get('yoy_pct')
        # Redondear a 1 decimal en la representación textual
        pv_txt = _r1(pv)
        cv_txt = _r1(cv)
        yoy_txt = _r1(yoy)
        body_lines.append(
            f"{period_lbl} | {pv_txt} | {cv_txt} | {yoy_txt}"
        )
    return "\n".join(header_lines + body_lines)