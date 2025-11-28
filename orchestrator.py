"""
orchestrator.py
---------------
Orquestador principal de PIBot con:

- Clasificación de consultas vía function calling (prompt.classify_query).
- Streaming real hacia Streamlit.
- Flujo en dos fases para consultas de DATOS (IMACEC / PIB / PIB_REGIONAL):
    1) Primera fase: respuesta metodológica / definición (streaming).
    2) Segunda fase: respuesta orientada a datos (streaming).
- Streaming también para consultas METODOLÓGICAS.
- Logging detallado por etapas en /logs para trazar errores y tiempos.
"""

from __future__ import annotations

import os
import time
import datetime as _dt
import logging
import re
import json
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Any, Tuple
from functools import lru_cache

try:
    from langchain_core.prompts import ChatPromptTemplate  # original import
except Exception:  # fallback stub if langchain not installed
    class ChatPromptTemplate:  # type: ignore
        def __init__(self, messages):
            self.messages = messages
        @classmethod
        def from_messages(cls, messages):
            return cls(messages)
        def __or__(self, other):  # pipe operator fallback
            class _Chain:
                def __init__(self, prompt, llm):
                    self.prompt = prompt
                    self.llm = llm
                def stream(self, vars_dict):  # mimic langchain stream interface
                    # Produce a minimal placeholder chunk
                    yield type("Chunk", (), {"content": "[Stub LLM: dependencia langchain no disponible]"})
            return _Chain(self, other)

try:
    from langchain_openai import ChatOpenAI  # original import
except Exception:  # fallback stub
    class ChatOpenAI:  # type: ignore
        def __init__(self, model: str, temperature: float = 0.0, streaming: bool = True):
            self.model = model
            self.temperature = temperature
            self.streaming = streaming
        def stream(self, vars_dict):
            yield type("Chunk", (), {"content": "[Stub ChatOpenAI: modelo=" + self.model + "]"})

from config import get_settings
try:
    from config import LOG_LEVEL
except ImportError:
    LOG_LEVEL = "INFO"

# Intentar usar el logger del proyecto; si falla, usamos uno propio
try:
    from logger import get_logger as _project_get_logger  # type: ignore
except Exception:
    _project_get_logger = None  # type: ignore[assignment]


_SESSION_LOG_PATH: Optional[str] = None

def _fallback_get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Logger de respaldo cuando no se puede usar logger.py del proyecto.

    - Escribe en /logs/pibot_YYYYMMDD.log
    - También escribe a consola (útil cuando corres streamlit)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # ya configurado

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    root = os.path.abspath(os.path.dirname(__file__))
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Un único archivo por EJECUCIÓN con timestamp de inicio.
    # Si ya existe ruta en esta sesión, reutilizar sin crear uno nuevo.
    global _SESSION_LOG_PATH
    if _SESSION_LOG_PATH and os.path.isfile(_SESSION_LOG_PATH):
        fname = _SESSION_LOG_PATH
        mode = 'a'
    else:
        session_ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(log_dir, f"pibot_{session_ts}.log")
        _SESSION_LOG_PATH = fname
        mode = 'w'

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Abrir en modo según sesión (w solo primera vez, luego a)
    fh = logging.FileHandler(fname, mode=mode, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# Si existe el logger del proyecto, úsalo. Si no, usa el fallback.
if _project_get_logger:
    logger = _project_get_logger(__name__, level=LOG_LEVEL)
else:
    logger = _fallback_get_logger(__name__, level=LOG_LEVEL)

# Capturar ruta del handler de archivo (sesión)
_SESSION_LOG_PATH: Optional[str] = None
for _h in logger.handlers:
    if isinstance(_h, logging.FileHandler):
        _SESSION_LOG_PATH = getattr(_h, 'baseFilename', None)
        break
if _SESSION_LOG_PATH:
    logger.info(f"[LOG_SESSION] path={_SESSION_LOG_PATH}")

# Guardar ruta del archivo de sesión para reutilización desde otros módulos
logger._session_log_path = _SESSION_LOG_PATH or fname  # type: ignore[attr-defined]

# No crear más handlers ni archivos adicionales.

# Utilidad pública para que otros módulos recuperen la ruta única de log
def get_current_test_log_file() -> str:
    return getattr(logger, "_session_log_path", "")

# Línea de utilidad para registrar el intent clasificado/manejado
def _log_intent(name: str) -> None:
    try:
        logger.info(f"[INTENT= {name}]")
    except Exception:
        pass

# ---------------------------------------------------------------
# Intent trace helpers (log calls and arguments per handler)
# ---------------------------------------------------------------

def _summ_arg(val: Any) -> str:
    try:
        if isinstance(val, dict):
            keys = list(val.keys())
            return f"dict(keys={keys[:5]}{'...' if len(keys)>5 else ''})"
        if isinstance(val, list):
            return f"list(len={len(val)})"
        if isinstance(val, tuple):
            return f"tuple(len={len(val)})"
        s = str(val)
        return s if len(s) <= 120 else s[:117] + '...'
    except Exception:
        return "<unrepr>"

def _summ_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, a in enumerate(args):
        out[f"arg{i}"] = _summ_arg(a)
    for k, v in kwargs.items():
        out[k] = _summ_arg(v)
    return out

def _call_with_trace(intent: str, func: Any, *args: Any, **kwargs: Any):
    try:
        fname = getattr(func, "__name__", str(func))
        logger.info(f"[INTENT_TRACE_CALL] intent={intent} func={fname} args={_summ_args(args, kwargs)}")
    except Exception:
        pass
    return func(*args, **kwargs)

from prompt import ClassificationResult, classify_query  # después de configurar logger
# ---------------------------------------------------------------------------
# API explícita: clasificación LLM + router modular de intents
# ---------------------------------------------------------------------------

def response_api_openai_type(
    question: str,
    history: List[Dict[str, str]],
) -> Tuple[ClassificationResult, str]:
    """Encapsula la clasificación con LLM y la construcción de history_text.

    - Emite logs de Fase C1 (start/end) con tiempos y salida estructurada.
    - Devuelve (classification, history_text) para consumo del router.
    """
    t_class_start = time.perf_counter()
    logger.info(
        f"[FASE] start: Fase C1: Clasificación de consulta (OpenAI - function calling) | model='{_settings.openai_model}'"
    )
    try:
        classification = classify_query(question)
    except Exception as e:
        t_class_end = time.perf_counter()
        logger.error(
            f"[FASE] error: Fase C1: Clasificación de consulta (OpenAI - function calling) ({t_class_end - t_class_start:.3f}s) | question='{question}' | error={e}"
        )
        raise
    t_class_end = time.perf_counter()
    logger.info(
        "[FASE] end: Fase C1: Clasificación de consulta (OpenAI - function calling) "
        f"({t_class_end - t_class_start:.3f}s) | "
        f"query_type={classification.query_type} | "
        f"data_domain={classification.data_domain} | "
        f"is_generic={classification.is_generic} | "
        f"default_key={classification.default_key} | "
        f"error={classification.error}"
    )
    history_text = _build_history_text(history)
    return classification, history_text


def intent_response(
    classification: ClassificationResult,
    question: str,
    history_text: str,
) -> Optional[Iterable[str]]:
    """Router modular: intenta manejar la consulta con intents configurables
    y rutas deterministas tempranas. Si maneja, devuelve un iterable de chunks;
    si no, retorna None para continuar con la lógica legacy sin cambios.
    """
    # 1) Intents por configuración
    dispatched = _dispatch_config_intents(classification, question, history_text)
    if dispatched is not None:
        logger.info("[ROUTE] CONFIG_INTENT_MATCHED")
        return dispatched
    # 2) Ruta temprana: contribución sectores IMACEC (determinista)
    if _detect_imacec_sector_contribution(question):
        logger.info("[ROUTE] IMACEC_SECTOR_CONTRIBUTION")
        return _stream_imacec_sector_contribution(question)
    return None

# Nuevo: plantillas de respuesta centralizadas
try:
    from answer import (
        get_methodological_instruction,
        get_data_first_phase_instruction,
        get_generic_instruction,
        get_processing_banner,
        get_data_second_phase_instruction,
        get_csv_download_footer,
    )
except Exception:
    # Fallback mínimo por si no está disponible answer.py en algún entorno
    def get_methodological_instruction() -> str:
        return (
            "Responde en modo METHODOLOGICAL: céntrate en definiciones, "
            "metodología y contexto. No intentes entregar tablas de datos ni "
            "cifras concretas."
        )

    def get_data_first_phase_instruction() -> str:
        return (
            "Responde como una PRIMERA FASE de una consulta de datos: "
            "entrega una definición breve del indicador, explica para qué sirve "
            "y cómo se relaciona con el período o año que menciona el usuario. "
            "No hables aún de tablas ni de pasos técnicos detallados."
        )

    def get_generic_instruction() -> str:
        return (
            "La consulta no encaja claramente en IMACEC/PIB/PIB regional o no es "
            "estrictamente de datos. Responde de manera general explicando el tema "
            "económico relevante, sin inventar cifras."
        )

    def get_processing_banner() -> str:
        return ("\n\n---\n\nProcesando los datos solicitados, esto puede tomar unos segundos...\n\n")

    def get_data_second_phase_instruction() -> str:
        return (
            "Responde en modo DATA (fase 2). Con los datos calculados que te pasen (variación PCT u otros), elabora una conclusión breve (3 a 5 líneas) que resuma tendencias y matices sin repetir la tabla. Luego, sugiere preguntas para continuar: 1) ¿Quieres cambiar la frecuencia de la serie anterior? 2) ¿Necesitas consultar por otra serie? 3) Propón una pregunta adicional relevante generada por ti en base al resumen. Incluye estas preguntas textualmente al final, para sostener el flujo conversacional."
        )

    def get_csv_download_footer(csv_path: Optional[str] = None) -> str:  # fallback silencioso
        return ""

# Nuevo: metadatos de series
try:
    from metadata import get_series_metadata  # deterministic lookup desde catalog
except Exception:
    get_series_metadata = None  # type: ignore


def _format_series_metadata_block(series_id: str) -> str:
    if not get_series_metadata:
        return ""
    md = get_series_metadata(series_id)
    if not md:
        return ""
    freq = (md.get('default_frequency') or '').strip()
    unit = (md.get('unit') or '').strip()
    code = (md.get('code') or '').strip()
    title = (md.get('title') or '').strip()
    url = (md.get('source_url') or '').strip()
    metodo = (md.get('metodologia') or '').strip()
    lines = [
        f"1. Código: {code}",
        f"2. Título: {title}",
        f"3. Frecuencia: {freq}",
        f"4. Unidad: {unit}",
        f"5. Gráfico: {url}",
        f"6. Metodología: {metodo}",
    ]
    return "\n".join(lines) + "\n\n"

# Contexto de última serie usada para soportar 'misma serie' y cambios de frecuencia
_last_data_context: Dict[str, Any] = {
    "series_id": None,
    "domain": None,
    "year": None,
    "freq": None,
    "data_full": None,  # dataset completo actual (último fetch)
    "data_full_original_annual": None,  # copia del dataset cuando se estableció métrica anual por primera vez
    "metric_type": "annual",  # 'annual' (yoy_pct) o 'monthly' (pct)
}

_last_vector_matches: List[Dict[str, Any]] = []  # almacena últimas coincidencias vectoriales
_last_vector_year: Optional[int] = None  # año detectado en la consulta que disparó el vector search

# Inferir dominio desde código de serie (heurística mínima)
def _infer_domain_from_series_id(series_id: Optional[str]) -> Optional[str]:
    sid = (series_id or '').upper()
    if not sid:
        return None
    if 'IMC' in sid or 'IMACEC' in sid:
        return 'IMACEC'
    if 'PIB' in sid:
        return 'PIB'
    if 'REG' in sid and 'PIB' in sid:
        return 'PIB_REGIONAL'
    return None

try:
    from search import search_serie_pg_vector  # vector search para otra serie
except Exception:
    search_serie_pg_vector = None  # type: ignore

# Resolver genérico de series por clave de catálogo (JSON) con fallback vectorial
def resolve_series_for_key(question: str, key: str, vector_fallback: bool = True):
    """
    Resuelve (series_id, frequency, agg) para una `key` usando el catálogo JSON.
    Si `vector_fallback` es True y no hay match en JSON, intenta búsqueda vectorial.
    Si es False, retorna (None, None, None) directamente tras intento JSON.
    """
    try:
        from series_default import resolve_series_from_text
        _sd = resolve_series_from_text(question, default_key=key)
        if _sd:
            series_id = _sd.series_id
            freq = _sd.frequency
            agg = getattr(_sd, 'agg', 'avg')
            logger.info(f"[DEFAULT_RESOLVE] key={key} variant={_sd.variant} sid={series_id} freq={freq} agg={agg}")
            return series_id, freq, agg
        else:
            logger.info(f"[DEFAULT_RESOLVE] sin resolución por catálogo para key={key}")
    except Exception as _e_sd:
        logger.warning(f"[DEFAULT_RESOLVE] excepción resolviendo catálogo key={key}: {_e_sd}")

    if not vector_fallback:
        return None, None, None

    # Fallback vectorial: intenta sugerir una serie cercana
    try:
        if search_serie_pg_vector:
            logger.info(f"[VECTOR_FALLBACK] key={key} intentando búsqueda vectorial")
            vec = search_serie_pg_vector(question, top_k=1) or []
            if vec and isinstance(vec[0], dict):
                sid = vec[0].get("cod_serie") or vec[0].get("series_id")
                if sid:
                    logger.info(f"[VECTOR_FALLBACK] key={key} selected sid={sid}")
                    return sid, "M", "avg"
            logger.info(f"[VECTOR_FALLBACK] key={key} sin resultados vectoriales")
        else:
            logger.info(f"[VECTOR_FALLBACK] key={key} búsqueda vectorial no disponible (función search_serie_pg_vector ausente)")
    except Exception as e:
        logger.warning(f"[VECTOR_FALLBACK] key={key} error={e}")

    return None, None, None

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

_settings = get_settings()

_llm_method = ChatOpenAI(
    model=_settings.openai_model,
    temperature=0.1,
    streaming=True,
)

_llm_data = ChatOpenAI(
    model=_settings.openai_model,
    temperature=0.1,
    streaming=True,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_METHOD_SYSTEM = (
    "Eres el asistente económico del Banco Central de Chile (PIBot). "
    "Respondes SIEMPRE en español. "
    "Tu foco en este modo es explicar definiciones, metodología y contexto de "
    "indicadores como IMACEC, PIB y PIB regional.\n\n"
    "No inventes cifras ni valores numéricos: si el usuario pide datos, "
    "limítate a explicar qué mide el indicador, cómo se interpreta y cómo se "
    "relaciona con la consulta."
)

_METHOD_HUMAN = (
    "Historial de la conversación (puede estar vacío):\n"
    "{history}\n\n"
    "Consulta actual del usuario:\n"
    "{question}\n\n"
    "Clasificación técnica de la consulta (NO la expongas literal al usuario):\n"
    "- query_type: {query_type}\n"
    "- data_domain: {data_domain}\n"
    "- is_generic: {is_generic}\n"
    "- default_key: {default_key}\n\n"
    "Instrucción de modo:\n"
    "{mode_instruction}\n\n"
    "Responde de forma clara, en 2 a 5 párrafos cortos, explicando:\n"
    "- Qué es el indicador relevante (IMACEC / PIB / PIB regional / otro).\n"
    "- Para qué se usa y cómo se interpreta.\n"
    "- Cómo se relaciona con la pregunta del usuario.\n"
    "Evita mencionar términos internos como 'query_type', 'default_key' o similares."
)

_method_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _METHOD_SYSTEM),
        ("human", _METHOD_HUMAN),
    ]
)

_DATA_SYSTEM = (
    "Eres el asistente económico del Banco Central de Chile (PIBot). "
    "Respondes SIEMPRE en español.\n\n"
    "Estás en el modo de respuesta orientada a DATOS. "
    "El usuario quiere valores de series económicas (IMACEC, PIB, PIB regional, etc.). "
    "Puedes describir qué harías para obtener los datos y cómo los presentarías "
    "(tablas, variaciones, gráficos), pero NO tienes acceso directo a la API de datos "
    "del BCCh en esta versión, así que NO inventes números concretos.\n\n"
    "Si no puedes entregar valores reales, dilo explícitamente y describe los pasos "
    "para obtenerlos desde la serie adecuada."
)

_DATA_HUMAN = (
    "Historial de la conversación (puede estar vacío):\n"
    "{history}\n\n"
    "Consulta actual del usuario:\n"
    "{question}\n\n"
    "Clasificación técnica de la consulta (NO la expongas literal al usuario):\n"
    "- query_type: {query_type}\n"
    "- data_domain: {data_domain}\n"
    "- is_generic: {is_generic}\n"
    "- default_key: {default_key}\n"
    "- arbol_imacec: {imacec_tree}\n"
    "- arbol_pibe: {pibe_tree}\n\n"
    "Instrucción de modo:\n"
    "El usuario ya recibió una breve definición del indicador. "
    "Ahora describe cómo obtendrías y presentarías los datos "
    "para el período que menciona (por ejemplo, 2025), sin inventar cifras."
)

_data_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _DATA_SYSTEM),
        ("human", _DATA_HUMAN),
    ]
)

# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

_YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")

def _extract_year(text: str) -> Optional[int]:
    match = _YEAR_PATTERN.search(text or "")
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return None
    return None

def _load_defaults_for_domain(domain: str) -> Optional[Dict[str, str]]:
    try:
        cfg_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "series", "config_default.json")
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
    """Envuelve get_series_api_rest_bcch con reintentos y backoff exponencial."""
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
                delay = backoff * (2 ** attempt)
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
    from get_series import get_series_api_rest_bcch  # import local para evitar ciclos
    defaults = _load_defaults_for_domain(domain)
    if not defaults or not defaults.get("cod_serie"):
        logger.warning(f"[DATA_FETCH] defaults ausentes para domain={domain}")
        return None
    series_id = defaults["cod_serie"]
    freq = defaults.get("freq") or None
    firstdate = f"{year-1}-01-01"  # incluir año previo para comparación y yoy
    lastdate = f"{year}-12-31"
    logger.info(f"[YEAR_DETECT] domain={domain} year_detected={year} series_id={series_id} rango={firstdate}->{lastdate} freq_default={freq}")
    # Reintentos con backoff para mitigar fallos transitorios
    retries, backoff = 2, 1.0
    for attempt in range(retries + 1):
        try:
            data = get_series_api_rest_bcch(
                series_id=series_id,
                firstdate=firstdate,
                lastdate=lastdate,
                target_frequency=freq,
                agg="avg",
            )
            return data
        except Exception as e:
            if attempt < retries:
                delay = backoff * (2 ** attempt)
                logger.warning(f"[DATA_FETCH_RETRY] domain={domain} sid={series_id} intento={attempt+1}/{retries+1} en {delay:.1f}s | error={e}")
                try:
                    time.sleep(delay)
                except Exception:
                    pass
                continue
            logger.error(f"[DATA_FETCH] Error obteniendo serie {series_id} domain={domain} year={year}: {e}")
            return None

def _fetch_series_for_year_by_series_id(series_id: str, year: int, target_freq: Optional[str]) -> Optional[Dict[str, Any]]:
    """Obtiene datos para un series_id explícito en el rango [year-1, year]."""
    try:
        from get_series import get_series_api_rest_bcch
    except Exception as e:
        logger.error(f"[DATA_FETCH] import get_series_api_rest_bcch falló: {e}")
        return None
    firstdate = f"{year-1}-01-01"
    lastdate = f"{year}-12-31"
    # Reintentos con backoff
    retries, backoff = 2, 1.0
    for attempt in range(retries + 1):
        try:
            data = get_series_api_rest_bcch(
                series_id=series_id,
                firstdate=firstdate,
                lastdate=lastdate,
                target_frequency=target_freq,
                agg="avg",
            )
            return data
        except Exception as e:
            if attempt < retries:
                delay = backoff * (2 ** attempt)
                logger.warning(f"[DATA_FETCH_RETRY] sid={series_id} intento={attempt+1}/{retries+1} en {delay:.1f}s | error={e}")
                try:
                    time.sleep(delay)
                except Exception:
                    pass
                continue
            logger.error(f"[DATA_FETCH] Error obteniendo serie {series_id} year={year} tf={target_freq}: {e}")
            return None

def _build_year_table(data: Dict[str, Any], year: int) -> str:
    """Tabla Markdown canónica de comparación año anterior vs año actual.

    Delegamos en get_series.build_year_comparison_table_text para unificar lógica
    y manejo de meses/trimestres faltantes. Mantiene encabezados exactos:
    "Mes | Año anterior | Año actual | Variación porcentual".
    """
    try:
        from get_series import build_year_comparison_table_text  # import local para evitar ciclos
    except Exception as e:
        logger.error(f"[DATA_TABLE] No se pudo importar función canónica: {e}")
        return ""

    table_text = build_year_comparison_table_text(data, year)

    # Logging detallado de contenido de tabla (solo filas, no encabezados)
    lines = table_text.split("\n")
    body_lines = lines[3:] if len(lines) > 3 else []
    for line in body_lines[:12]:  # limitar spam
        logger.info(f"[DATA_TABLE_CONTENT] {line}")
    if not body_lines:
        logger.warning(f"[DATA_TABLE] Sin observaciones para {year}")

    return table_text

def _summarize_with_llm(domain: str, year: int, table_text: str) -> str:
    instruction = get_data_second_phase_instruction()
    # Nuevo mensaje de sistema alineado con fase 2: SIN conclusiones ni repetición de tabla
    system_msg = (
        "Eres el asistente económico del Banco Central de Chile (PIBot). Responde SIEMPRE en español. "
        "Estás en la FASE 2 de una respuesta orientada a DATOS.")
    system_msg += (
        " SOLO debes generar una frase introductoria neutral y TRES preguntas de seguimiento, "
        "sin ningún tipo de conclusión, interpretación, evaluación, resumen numérico ni juicio sobre los datos. "
        "NO describas tendencias, estabilidad, alzas, bajas, cambios ni uses adjetivos como 'fuerte', 'débil', 'estable', etc. "
        "NO repitas la tabla, NO muestres filas ni columnas, NO reescribas valores numéricos ni porcentajes. "
        "NO menciones años futuros al {year} ni proyectes resultados. "
        "La salida final debe ser SOLO texto plano: una breve frase introductoria neutral y luego exactamente tres preguntas en líneas separadas."
    ).format(year=year)

    # En lugar de pasar la tabla completa, entregamos solo una descripción textual
    table_description = (
        "Tabla de comparación año anterior vs año actual para el dominio '{domain}' y el año {year}. "
        "La tabla ya fue mostrada al usuario y contiene columnas de periodo, año anterior, año actual y variación anual. "
        "NO vuelvas a mostrarla ni describir sus celdas en detalle."
    ).format(domain=domain, year=year)

    human_msg = (
        f"Dominio: {domain}\nAño consultado: {year}\n"
        f"Resumen de tabla ya mostrada al usuario: {table_description}\n\n"
        f"Instrucción fase 2 desde answer.py:\n{instruction}"
    )

    chain = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", human_msg),
    ]) | _llm_data
    out: List[str] = []
    for chunk in chain.stream({}):
        raw_content = getattr(chunk, "content", None) or getattr(chunk, "text", None)
        if callable(raw_content):
            try:
                raw_content = raw_content()
            except Exception:
                raw_content = str(raw_content)
        content = str(raw_content) if raw_content else ""
        if content.startswith("<bound method"):
            parts = content.split(">", 1)
            content = parts[1].strip() if len(parts) == 2 else ""
        if content:
            out.append(content)
    summary = "".join(out)

    # Normalizar espacios y asegurar que solo haya contenido textual plano
    summary = summary.strip()
    logger.info(f"[PHASE_2_SUMMARY] chars={len(summary)}")
    return summary

def _stream_data_phase_with_table(
    classification: ClassificationResult,
    question: str,
    history_text: str,
    domain: str,
    year: int,
    data: Dict[str, Any],
) -> Iterable[str]:
    """Emite tabla de comparación y luego metadatos + resumen fase 2 en streaming.

    Orden nuevo solicitado:
    1) Tabla
    2) Bloque de metadatos deterministas
    3) Resumen analítico + preguntas (stream por oración)
    """
    # Guardar contexto de última serie
    meta = data.get("meta", {}) or {}
    _last_data_context.update({
        "series_id": meta.get("series_id"),
        "domain": domain,
        "year": year,
        "freq": meta.get("freq_effective"),
        "data_full": data,
    })

    table_text = _build_year_table(data, year)
    # Evitar f-string con expresión que incluye literal con barra invertida para Python<=3.9
    lines_count = table_text.count("\n") + 1
    logger.info(f"[DATA_TABLE] domain={domain} year={year} lines={lines_count}")
    yield "\n" + table_text + "\n\n"

    # Metadatos antes del resumen
    series_id = meta.get("series_id")
    if (series_id):
        md_block = _format_series_metadata_block(series_id)
        if md_block.strip():
            logger.info(f"[SERIES_META] series_id={series_id}")
            yield md_block + "\n"

    summary_full = _summarize_with_llm(domain, year, table_text)
    disclaimer_patterns = [
        r"no puedo proporcionar cifras", r"no puedo entregar cifras", r"no puedo proporcionar valores"
    ]
    for pat in disclaimer_patterns:
        summary_full = re.sub(pat, "", summary_full, flags=re.IGNORECASE)

    sentences = re.split(r"(?<=[.!?])\s+", summary_full.strip())
    for s in sentences:
        if s:
            yield s + "\n"
    # CSV marker para todas las respuestas DATA con tabla
    try:
        marker = _emit_csv_download_marker(table_text, f"{domain.lower()}_{year}", preferred_filename=f"{domain.lower()}_{year}.csv")
        if marker:
            yield "\n" + marker + "\n"
    except Exception as _e_footer:
        logger.error(f"[CSV_MARKER_ERROR] domain={domain} year={year} e={_e_footer}")

# ---------------------------------------------------------------------------
# Funciones auxiliares restauradas (streaming y normalización)
# ---------------------------------------------------------------------------

def _normalize(s: Optional[str]) -> str:
    """Normaliza cadenas para comparaciones de tipo/dominio."""
    return (s or "").strip().upper()


def _build_history_text(history: Optional[List[Dict[str, str]]]) -> str:
    """Convierte el historial en texto plano para el prompt.

    Limita a los últimos 20 turnos y limpia saltos de línea excesivos.
    """
    if not history:
        return ""
    lines: List[str] = []
    for turn in history[-20:]:  # solo últimos 20
        role = turn.get("role", "user")
        content = (turn.get("content") or "").replace("\r", " ").replace("\n", " ").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _stream_methodological_phase(
    classification: ClassificationResult,
    question: str,
    history_text: str,
    mode_instruction: str,
) -> Iterable[str]:
    """Genera chunks de respuesta metodológica usando el LLM configurado.

    Para activar RAG, descomenta el bloque indicado y comenta el bloque OpenAI.
    """
    # --- ACTIVAR RAG: descomenta estas líneas y comenta el bloque OpenAI de abajo ---
        # Para ACTIVAR RAG desde este branch, reemplaza la línea inner_iter por:
        # inner_iter=(
        #     (lambda: (
        #         (yield __import__('rag').generate_methodological_response(
        #             question=question,
        #             classification={
        #                 "query_type": classification.query_type,
        #                 "data_domain": classification.data_domain,
        #                 "is_generic": classification.is_generic,
        #                 "default_key": classification.default_key,
        #             },
        #             history_text=history_text,
        #         ))
        #     ))()
        # )
    # Bloque actual (OpenAI) - default
    try:
        chain = _method_prompt | _llm_method
        vars_in = {
            "history": history_text,
            "question": question,
            "query_type": classification.query_type,
            "data_domain": classification.data_domain,
            "is_generic": classification.is_generic,
            "default_key": classification.default_key,
            "mode_instruction": mode_instruction,
            "imacec_tree": asdict(classification.imacec) if classification.imacec else None,
            "pibe_tree": asdict(classification.pibe) if classification.pibe else None,
        }
        for chunk in chain.stream(vars_in):
            raw_content = getattr(chunk, "content", None) or getattr(chunk, "text", None)
            # Evitar representación de métodos ('<bound method BaseMessage.text...>')
            if callable(raw_content):
                try:
                    raw_content = raw_content()
                except Exception:
                    raw_content = str(raw_content)
            content = str(raw_content) if raw_content else ""
            if content.startswith("<bound method"):
                # Extraer todo después del primer '>'
                parts = content.split(">", 1)
                content = parts[1].strip() if len(parts) == 2 else ""
            if content:
                yield content
    except Exception as e:
        logger.error(f"[STREAM_METHOD] error generando respuesta metodológica: {e}")
        yield "Ocurrió un problema generando la respuesta metodológica. Intenta nuevamente más tarde."


def _wrap_phase_stream(
    phase_name: str,
    description: str,
    inner_iter: Iterable[str],
) -> Iterable[str]:
    """Envuelve un iterable de chunks con logging de inicio/fin y timing."""
    t0 = time.perf_counter()
    logger.info(f"[FASE] start: {phase_name} | {description}")
    try:
        for chunk in inner_iter:
            yield chunk
    except Exception as e:
        logger.error(f"[FASE] error: {phase_name} | desc={description} | error={e}")
        yield "Se produjo un error interno al generar esta fase."
    finally:
        t1 = time.perf_counter()
        logger.info(f"[FASE] end: {phase_name} ({t1 - t0:.3f}s) | {description}")

def _detect_frequency_change(question: str) -> Optional[Dict[str, Any]]:
    """Detecta intención de cambio de frecuencia y si refiere a la misma serie.

    Reglas ampliadas:
    - same_series True si explícito ("misma serie", "serie anterior", "misma")
      o si se menciona el dominio (IMACEC / PIB) y el contexto previo coincide.
    - target_freq mapea textos comunes.
    - Retorna None si no se detecta ninguna intención.
    """
    q_raw = question or ''
    q = q_raw.lower()
    # Nueva regla: solo consideramos cambio de frecuencia si aparece la palabra
    # 'frecuencia' o expresiones explícitas de cambio ('cambio de frecuencia', 'cambiar la frecuencia').
    if not re.search(r"frecuencia|cambio de frecuencia|cambiar la frecuencia", q):
        return None
    explicit_same = any(kw in q for kw in ["misma serie", "serie anterior", "misma"])
    # dominio mencionado
    domain_mention = None
    if "imacec" in q:
        domain_mention = "IMACEC"
    elif re.search(r"\bpib\b", q):
        domain_mention = "PIB"
    context_domain = _last_data_context.get("domain")
    same_series = explicit_same or (domain_mention and context_domain and domain_mention == context_domain)
    # mapear textos a código
    if "trimestral" in q or re.search(r"\b(trimestral|trimestre|quarter)\b", q):
        tf = "T"
    elif "mensual" in q or re.search(r"\bmes(mes)?\b", q):
        tf = "M"
    elif "anual" in q or "año" in q or "anio" in q:
        tf = "A"
    elif "diaria" in q or "diario" in q:
        tf = "D"
    else:
        tf = None
    if not tf:
        return None
    try:
        logger.info(
            f"[FREQ_CHANGE_DETECTED] target={tf} | same_series={same_series} | domain_mention={domain_mention} | context_domain={context_domain} | "
            f"last_series={_last_data_context.get('series_id')} | last_freq={_last_data_context.get('freq')}"
        )
    except Exception:
        pass
    return {"target_freq": tf, "same_series": same_series, "domain_mention": domain_mention}

def _resample_existing_observations(data: Dict[str, Any], target_freq: str) -> Optional[Dict[str, Any]]:
    """Remuestrea datos ya obtenidos (sin nueva llamada a API) a otra frecuencia.

    - Usa promedio para agregaciones (mensual->trimestral/anual, trimestral->anual).
    - Mantiene meta original ajustando freq_effective.
    - Recalcula pct / yoy_pct tras remuestreo.
    """
    try:
        import pandas as _pd
    except Exception:
        logger.error("[FREQ_CHANGE] pandas no disponible para remuestreo in-memory")
        return None
    obs = data.get("observations", []) or []
    if not obs:
        return None
    df = _pd.DataFrame(obs)
    try:
        df["date"] = _pd.to_datetime(df["date"])
    except Exception:
        return None
    df = df.sort_values("date")
    orig_freq = (data.get("meta", {}) or {}).get("freq_effective", "")
    tf = target_freq.upper()
    if tf == orig_freq.upper():
        return data  # ya está en esa frecuencia
    rule_map = {"M": "M", "Q": "Q-DEC", "T": "Q-DEC", "A": "A-DEC"}
    if tf not in rule_map:
        return None
    rule = rule_map[tf]
    # Promedio siempre según requerimiento
    df_res = df.set_index("date").resample(rule).mean(numeric_only=True)
    df_res = df_res.dropna(how="all").reset_index()
    # Recalcular variaciones
    df_res_sorted = df_res.sort_values("date")
    df_res_sorted["pct"] = df_res_sorted["value"].pct_change() * 100.0
    # yoy según frecuencia destino
    lag = {"M":12, "Q":4, "T":4, "A":1}.get(tf, None)
    if lag:
        prev = df_res_sorted["value"].shift(lag)
        df_res_sorted["yoy_pct"] = (df_res_sorted["value"] / prev - 1.0) * 100.0
    else:
        df_res_sorted["yoy_pct"] = None
    observations_new = []
    for _, row in df_res_sorted.iterrows():
        observations_new.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "value": None if _pd.isna(row["value"]) else float(row["value"]),
            "status": "",
            "pct": None if _pd.isna(row["pct"]) else float(row["pct"]),
            "yoy_pct": None if _pd.isna(row.get("yoy_pct")) else float(row.get("yoy_pct")),
        })
    meta_new = {**(data.get("meta", {}) or {})}
    meta_new["freq_effective"] = tf
    meta_new["target_frequency"] = tf
    out = {"meta": meta_new, "observations": observations_new, "observations_raw": data.get("observations_raw", [])}
    return out

def _stream_frequency_change_table_only(
    original_data: Dict[str, Any],
    target_freq: str,
    domain: str,
    year: Optional[int],
) -> Iterable[str]:
    """Emite título + tabla + metadatos + recomendaciones (sin introducción).

    - Si target_freq == original_freq → reusa tabla original (sin recomputar).
    - Para IMACEC/PIB: mantener comparación año-1 vs año cuando sea posible.
    - Si no hay year almacenado, intentar inferir último año disponible.
    - Reemplaza el encabezado "Comparación X vs Y" por un título específico cuando es trimestral.
    """
    stored_meta = (original_data.get("meta") or {})
    orig_freq = stored_meta.get("freq_effective") or stored_meta.get("target_frequency")
    # Revertir a original
    if target_freq.upper() == (orig_freq or '').upper():
        data_use = original_data
        logger.info(f"[FREQ_CHANGE_APPLY] reuse_original | orig_freq={orig_freq} -> target={target_freq}")
    else:
        logger.info(f"[FREQ_CHANGE_APPLY] resample | orig_freq={orig_freq} -> target={target_freq}")
        data_use = _resample_existing_observations(original_data, target_freq) or original_data
    # Actualizar contexto global
    _last_data_context.update({
        "freq": data_use.get("meta", {}).get("freq_effective"),
        "data_full": original_data,  # mantenemos original para futuras reversiones
    })
    # Determinar año
    if year is None:
        year = _get_latest_year_from_data(original_data)
    if year is None:
        yield "No se pudo determinar el año para construir la tabla tras el cambio de frecuencia."
        return
    # Construir tabla según tipo de respuesta previa (si original tenía comparación completa o simple one-row)
    obs = data_use.get("observations", []) or []
    # Si frecuencia anual → construir tabla yoy simple
    if data_use.get("meta", {}).get("freq_effective") == "A":
        table_text = _build_year_yoy_simple_table(data_use, year)
    else:
        # Comparación año anterior vs actual si tenemos datos del año anterior
        years_present = {int(str(o.get("date"))[:4]) for o in obs if o.get("date")}
        if (year - 1) in years_present:
            table_text = _build_year_table(data_use, year)
        else:
            # Último período solo
            table_text = _build_latest_only_table(data_use)
    # Ajustar título cuando la frecuencia objetivo es trimestral
    eff_freq = (data_use.get("meta", {}).get("freq_effective") or target_freq or "").upper()
    desired_title = None
    if eff_freq in ("Q", "T"):
        desired_title = "Variación porcentual anual trimestral"
    elif eff_freq == "M":
        desired_title = "Variación porcentual anual mensual"
    elif eff_freq == "A":
        desired_title = "Variación porcentual anual"

    if desired_title:
        lines_tbl = table_text.split("\n") if table_text else []
        if lines_tbl and lines_tbl[0].strip().lower().startswith("comparación "):
            lines_tbl[0] = desired_title
            table_text = "\n".join(lines_tbl)
        else:
            # Prependemos el título si no existe una fila de comparación
            table_text = desired_title + "\n\n" + table_text

    _freq_lines_count = table_text.count("\n") + 1
    logger.info(f"[FREQ_CHANGE_TABLE_ONLY] domain={domain} year={year} freq_target={target_freq} lines={_freq_lines_count}")

    # Reemplazar encabezado específico (Mes | / Trimestre |) por Periodo |
    _tbl_lines = table_text.split("\n")
    for _i, _l in enumerate(_tbl_lines):
        if _l.startswith("Mes | ") or _l.startswith("Mes | Año"):
            _tbl_lines[_i] = _l.replace("Mes |", "Periodo |")
        elif _l.startswith("Trimestre |"):
            _tbl_lines[_i] = _l.replace("Trimestre |", "Periodo |")
    table_text = "\n".join(_tbl_lines)

    # Emitir tabla con título
    yield "\n" + table_text + "\n\n"

    # Metadatos de la serie, si están disponibles
    series_id = stored_meta.get("series_id") or _last_data_context.get("series_id")
    if series_id:
        md_block = _format_series_metadata_block(str(series_id))
        if md_block.strip():
            logger.info(f"[SERIES_META] series_id={series_id}")
            yield md_block + "\n"

    # Recomendaciones tipo fase 2 (sin introducción)
    try:
        summary_full = _summarize_with_llm(domain, year, table_text)
    except Exception as _e_sum:
        logger.error(f"[FREQ_CHANGE_PHASE2_ERROR] {_e_sum}")
        summary_full = "¿Deseas cambiar nuevamente la frecuencia?\n¿Quieres consultar otra serie?\n¿Te interesa comparar con otro año?"

    # limpiar disclaimers residuales
    for pat in [r"no puedo proporcionar cifras", r"no puedo entregar cifras", r"no puedo proporcionar valores"]:
        summary_full = re.sub(pat, "", summary_full, flags=re.IGNORECASE)

    for s in re.split(r"(?<=[.!?])\s+", summary_full.strip()):
        if s:
            yield s + "\n"
    # CSV marker para cambio de frecuencia (tabla únicamente)
    try:
        eff_freq = (data_use.get("meta", {}).get("freq_effective") or target_freq or "").upper()
        filename_base = f"{(domain or 'serie').lower()}_{year}_{eff_freq.lower()}"
        marker = _emit_csv_download_marker(table_text, filename_base, preferred_filename=f"{filename_base}.csv")
        if marker:
            yield "\n" + marker + "\n"
    except Exception as _e_footer:
        logger.error(f"[CSV_MARKER_ERROR] freq_change domain={domain} year={year} tf={target_freq} e={_e_footer}")

def _handle_vector_search_other_series(question: str) -> Optional[List[Dict[str, Any]]]:
    """Si el usuario solicita otra serie, realiza búsqueda vectorial y retorna top-3 coincidencias.

    Retorna lista de matches: [{cod_serie, nkname_esp, similarity}, ...] o None.
    """
    if search_serie_pg_vector is None:
        return None
    if not re.search(r"otra serie|otra.*serie|consultar otra serie", question.lower()):
        return None
    try:
        matches = search_serie_pg_vector(question, top_k=3) or []
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH] Error búsqueda vectorial: {e}")
        return None
    if not matches:
        return None
    return matches

def _get_latest_year_from_data(data: Dict[str, Any]) -> Optional[int]:
    """Obtiene el último año disponible en las observaciones devueltas por BCCh."""
    obs = (data or {}).get("observations") or []
    years = []
    for o in obs:
        d = o.get("date")
        if not d:
            continue
        try:
            y = int(str(d)[:4])
            years.append(y)
        except Exception:
            continue
    return max(years) if years else None


def _format_last_yoy_from_table(data: Dict[str, Any], year: int) -> Optional[str]:
    """Extrae la última variación anual del año indicado y la formatea como texto breve.

    Busca en las observaciones del año `year` el último registro con `yoy_pct` y
    retorna una frase tipo: "La última variación anual disponible es X% (mes AAAA).".
    """
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
    return f"La última variación anual disponible es {last_yoy:.1f}% (período {txt_date})."

def _sanitize_llm_text(text: str) -> str:
    """Elimina artefactos (<bound method ...> y metadatos), normaliza espacios.
    Heurísticas simples para reinsertar espacios donde se pegaron palabras.
    """
    if not text:
        return ""
    # Remover artefactos de métodos bound y metadatos de chunks
    text = re.sub(r"<bound method BaseMessage.text of AIMessageChunk\([^>]+\)>", "", text)
    text = re.sub(r"content='' additional_kwargs=\{\} response_metadata=\{[^}]*\} id='run--[0-9a-f-]+'", "", text)
    # A veces tokens se concatenan sin espacio: insertar espacio entre minúscula + mayúscula consecutivas
    text = re.sub(r"([a-záéíóú])([A-ZÁÉÍÓÚ])", r"\1 \2", text)
    # Colapsar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _build_year_change_only_table(data: Dict[str, Any], year: int) -> str:
    """Construye una tabla Markdown solo con la variación anual (yoy_pct) del año dado.

    Formato:
        Comparación {year-1} vs {year}
        Mes | Variación anual
        ----|-----------------
        2025-01 | 1.2
        ...

    Se basa en las observaciones ya enriquecidas con `yoy_pct` que entrega get_series.
    """
    obs = (data or {}).get("observations") or []
    try:
        logger.info(f"[CALC_TRACE_ENTER] file=orchestrator.py func=_build_year_change_only_table args={{'year': {year}, 'obs_len': {len(obs)}}}")
    except Exception:
        pass
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

    table_text = "\n".join(lines)
    # Loguear algunas filas
    for ln in lines[3:6]:
        logger.info(f"[DATA_TABLE_CONTENT_YOY_ONLY] {ln}")
    try:
        logger.info(f"[CALC_TRACE_EXIT] file=orchestrator.py func=_build_year_change_only_table result={{'type':'str','lines': {len(lines)}}}")
    except Exception:
        pass
    return table_text

def _build_latest_only_table(data: Dict[str, Any]) -> str:
    """Construye una tabla Markdown de UNA sola fila con la última observación disponible.

    Columnas:
        Último período | Variación anual

    Si no existe `yoy_pct`, se intenta usar `value`.
    """
    obs = (data or {}).get("observations") or []
    last = None
    for o in obs:
        # Se toma el último que tenga al menos yoy_pct o value
        if o.get("yoy_pct") is not None or o.get("value") is not None:
            last = o
    if not last:
        try:
            logger.info(f"[CALC_TRACE_ENTER] file=orchestrator.py func=_build_latest_only_table args={{'obs_len': {len(obs)}}}")
            logger.info(f"[CALC_TRACE_EXIT] file=orchestrator.py func=_build_latest_only_table result={'type':'str','lines':1}")
        except Exception:
            pass
        return "No se encontró una observación reciente."
    date = last.get("date") or "(sin fecha)"
    yoy = last.get("yoy_pct")
    val = last.get("value")
    if yoy is not None:
        metric = f"{float(yoy):.1f}%"
        header_metric = "Variación anual"
    elif val is not None:
        try:
            metric = f"{float(val):,.2f}".replace(",", "_").replace("_", ".")  # formato simple
        except Exception:
            metric = str(val)
        header_metric = "Valor"
    else:
        metric = "--"
        header_metric = "Valor"
    lines = [
        f"Último período | {header_metric}",
        "---------------|----------------",
        f"{date} | {metric}",
    ]
    for ln in lines:
        logger.info(f"[DATA_TABLE_CONTENT_LATEST_ONLY] {ln}")
    table = "\n".join(lines)
    try:
        logger.info(f"[CALC_TRACE_ENTER] file=orchestrator.py func=_build_latest_only_table args={{'obs_len': {len(obs)}}}")
        logger.info(f"[CALC_TRACE_EXIT] file=orchestrator.py func=_build_latest_only_table result={{'type':'str','lines': {len(lines)}}}")
    except Exception:
        pass
    return table

def _build_year_yoy_simple_table(data: Dict[str, Any], year: int) -> str:
    """Tabla simplificada SOLO con Periodo | Variación anual para el año dado.

    Diferente de `_build_year_change_only_table` porque NO incluye encabezado de comparación,
    únicamente:
        Mes | Variación anual
        ----|-----------------
        2025-01 | 3.0
        ...
    """
    obs = (data or {}).get("observations") or []
    try:
        logger.info(f"[CALC_TRACE_ENTER] file=orchestrator.py func=_build_year_yoy_simple_table args={{'year': {year}, 'obs_len': {len(obs)}}}")
    except Exception:
        pass
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
    table = "\n".join(lines)
    try:
        logger.info(f"[CALC_TRACE_EXIT] file=orchestrator.py func=_build_year_yoy_simple_table result={{'type':'str','lines': {len(lines)}}}")
    except Exception:
        pass
    return table

def _emit_csv_download_marker(table_text: str, filename_base: str, preferred_filename: Optional[str] = None) -> str:
    """Genera CSV a partir de una tabla Markdown y devuelve un bloque de control para UI.

    Retorna un bloque delimitado por:
        ##CSV_DOWNLOAD_START\n
        path=<abs_path>\n
        filename=<file.csv>\n
        mimetype=text/csv\n
        label=Descargar CSV\n
        ##CSV_DOWNLOAD_END\n

    La app Streamlit detecta este bloque y renderiza un st.download_button nativo.
    """
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

def _export_table_to_csv(table_text: str, filename_base: str) -> str:
    """Exporta una tabla Markdown (formato generado por _build_year_table / variantes) a CSV.

    - Detecta línea de encabezado (contiene '|') y la línea separadora (-----|----...).
    - Ignora líneas vacías y primeras líneas descriptivas tipo 'Comparación X vs Y' si no contienen '|'.
    - Genera archivo en `logs/exports/` con nombre `{filename_base}.csv`.
    - Retorna ruta absoluta del archivo creado.
    """
    lines = [l for l in table_text.split('\n') if l.strip()]
    if not lines:
        raise ValueError("Tabla vacía para exportar")
    header_idx = None
    sep_idx = None
    for i, ln in enumerate(lines):
        if '|' in ln and header_idx is None and not ln.lower().startswith('comparación '):
            header_idx = i
        elif header_idx is not None and re.match(r'^[\-\s|]+$', ln):
            sep_idx = i
            break
    if header_idx is None or sep_idx is None:
        raise ValueError("No se pudo identificar encabezado y separador en la tabla Markdown")
    header_cols = [c.strip() for c in lines[header_idx].split('|')]
    data_rows = []
    for ln in lines[sep_idx+1:]:
        if '|' not in ln:
            continue
        cols = [c.strip() for c in ln.split('|')]
        if len(cols) != len(header_cols):
            continue
        data_rows.append(cols)
    root = os.path.abspath(os.path.dirname(__file__))
    export_dir = os.path.join(root, 'logs', 'exports')
    os.makedirs(export_dir, exist_ok=True)
    filepath = os.path.join(export_dir, f'{filename_base}.csv')
    import csv
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header_cols)
        for row in data_rows:
            w.writerow(row)
    return filepath

def _emit_chart_marker(domain: str, data: Dict[str, Any]) -> Optional[str]:
    """Genera un archivo CSV con columnas básicas para graficar y retorna bloque marker.

    Formato del bloque:
        ##CHART_START\n
        type=line\n
        title=<titulo>\n
        data_path=<abs_path>\n
        columns=date,yoy_pct\n
        domain=<domain>\n
        ##CHART_END\n
    La app detectará este bloque y dibujará st.line_chart sobre la(s) columnas relevantes.
    """
    if not data:
        return None
    obs = (data.get("observations") or [])
    if not obs:
        return None
    metric_type = _last_data_context.get("metric_type", "annual")
    try:
        import csv, time as _t
        root = os.path.abspath(os.path.dirname(__file__))
        export_dir = os.path.join(root, 'logs', 'exports')
        os.makedirs(export_dir, exist_ok=True)
        ts = int(_t.time())
        filename_base = f"{domain.lower()}_chart_{ts}.csv"
        path = os.path.join(export_dir, filename_base)
        # Limitar a año actual + anterior cuando haya contexto de año
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
            # Solo el año de contexto (ej: 2025) y solo variación anual disponible
            if year_ctx and y != int(year_ctx):
                continue
            if metric_type == "monthly":
                val = o.get("pct")
            else:
                val = o.get("yoy_pct")
            if val is None:
                continue
            rows_out.append((d, val))
        with open(path, 'w', newline='', encoding='utf-8') as f:
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
    if metric_type == "monthly":
        title = f"{domain.upper()} - Variación mensual (%) {year_ctx}" if domain else "Serie"
        columns_line = "columns=date,pct"
    else:
        title = f"{domain.upper()} - Variación anual (%) {year_ctx}" if domain else "Serie"
        columns_line = "columns=date,yoy_pct"
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
    logger.info(f"[CHART_MARKER_EMITTED] path={path} domain={domain} rows={len(rows_out)} year_ctx={year_ctx} cols=date,yoy_pct")
    return "\n".join(block)

# ---------------------------------------------------------------------------
# Legacy shim: fase de datos genérica por streaming (fallback)
# ---------------------------------------------------------------------------

def _stream_data_phase(
    classification: ClassificationResult,
    question: str,
    history_text: str,
) -> Iterable[str]:
    """Fallback para la fase de datos cuando no hay fetch real.

    Usa `_data_prompt | _llm_data` para generar un texto guía sin inventar cifras.
    """
    try:
        chain = _data_prompt | _llm_data
        vars_in = {
            "history": history_text,
            "question": question,
            "query_type": classification.query_type,
            "data_domain": classification.data_domain,
            "is_generic": classification.is_generic,
            "default_key": classification.default_key,
            "imacec_tree": asdict(classification.imacec) if classification.imacec else None,
            "pibe_tree": asdict(classification.pibe) if classification.pibe else None,
            "mode_instruction": get_data_first_phase_instruction(),
        }
        for chunk in chain.stream(vars_in):
            raw_content = getattr(chunk, "content", None) or getattr(chunk, "text", None)
            if callable(raw_content):
                try:
                    raw_content = raw_content()
                except Exception:
                    raw_content = str(raw_content)
            content = str(raw_content) if raw_content else ""
            if content.startswith("<bound method"):
                parts = content.split(">", 1)
                content = parts[1].strip() if len(parts) == 2 else ""
            if content:
                yield content
    except Exception as e:
        logger.error(f"[STREAM_DATA_FALLBACK] error: {e}")
        yield "No fue posible generar la fase de datos en este momento."

def _detect_chart_request(question: str) -> bool:
    q = (question or "").lower()
    if re.search(r"grafico|gráfico", q) and ("imacec" in q or "pib" in q or "serie" in q):
        return True
    # Permitir comando directo: 'realiza un grafico' si ya hay contexto de datos
    if re.search(r"realiza un gráfico|realiza un grafico|mostrar gráfico", q):
        return True
    return False

# ---------------------------------------------------------------
# Router de intents desde configuración (catalog/intents.json)
# ---------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_intents_config() -> List[Dict[str, Any]]:
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "catalog", "intents.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        logger.error(f"[INTENTS_CFG_LOAD_ERROR] path={path} e={e}")
        return []
    # Expandir placeholder {MONTHS}
    months = "|".join(map(re.escape, _MONTH_NAME_MAP.keys()))
    for it in cfg:
        pats = []
        for p in it.get("patterns", []) or []:
            pats.append(p.replace("{MONTHS}", months))
        it["_compiled"] = [re.compile(p, flags=re.IGNORECASE) for p in pats]
    # Ordenar por prioridad ascendente (menor = mayor prioridad)
    cfg.sort(key=lambda x: int(x.get("priority", 9999)))
    return cfg

def _intent_requires_domain_ok(intent: Dict[str, Any], question: str) -> bool:
    req = (intent.get("requiresDomain") or "").upper().strip()
    q = (question or "").lower()
    if not req:
        return True
    if req == "IMACEC":
        return "imacec" in q
    if req == "PIB":
        return re.search(r"\bpib\b", q) is not None
    return True

def _dispatch_config_intents(
    classification: ClassificationResult,
    question: str,
    history_text: str,
) -> Optional[Iterable[str]]:
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
            if not handler_name:
                continue
            try:
                handler = globals().get(handler_name)
                if not callable(handler):
                    logger.warning(f"[INTENTS_HANDLER_MISSING] name={handler_name}")
                    continue
                stream = handler(classification, question, history_text, m)
                if stream is None:
                    continue
                it_out = iter(stream)
                try:
                    first = next(it_out)
                except StopIteration:
                    logger.info("[CONFIG_INTENT_EMPTY_OUTPUT] handler produced no chunks; trying next intent")
                    continue
                def _gen():
                    yield first
                    for x in it_out:
                        yield x
                try:
                    intent_name = it.get("name") or "(sin_nombre)"
                    logger.info(f"[CONFIG_INTENT] name={intent_name} handler={handler_name} status=matched")
                    _log_intent(intent_name)
                except Exception:
                    logger.info("[ROUTE] CONFIG_INTENT_MATCHED")
                return _gen()
            except Exception as e:
                logger.error(f"[INTENTS_HANDLER_ERROR] name={handler_name} e={e}")
                continue
    return None

# Handlers que reusan la lógica existente
def _handle_intent_imacec_month_interval(
    classification: ClassificationResult,
    question: str,
    history_text: str,
    match: Any,
) -> Iterable[str]:
    try:
        m1_name = match.group(1).lower()
        m2_name = match.group(2).lower()
        year = int(match.group(3))
    except Exception:
        return iter(["No se pudo interpretar el intervalo de meses."])
    m1 = _MONTH_NAME_MAP.get(m1_name)
    m2 = _MONTH_NAME_MAP.get(m2_name)
    if not (m1 and m2):
        return iter(["Meses no reconocidos en el intervalo solicitado."])
    if m1 > m2:
        m1, m2 = m2, m1
    defaults = _load_defaults_for_domain("IMACEC") or {}
    series_id = defaults.get("cod_serie")
    freq = defaults.get("freq") or None
    if not series_id:
        return iter(["No se encontró la serie por defecto para el IMACEC."])
    firstdate = f"{year-1}-01-01"
    lastdate = f"{year}-12-31"
    data = _call_with_trace(
        "IMACEC_MONTH_INTERVAL",
        _get_series_with_retry,
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=freq,
        agg="avg",
        retries=2,
        backoff=1.0,
    )
    if not data:
        return iter(["No fue posible conectar con la API tras reintentos. Por favor, intenta nuevamente."])
    table_text = _call_with_trace("IMACEC_MONTH_INTERVAL", _build_month_interval_yoy_table, data, year, m1, m2)
    out: List[str] = ["\n" + table_text + "\n\n"]
    _last_data_context.update({
        "series_id": series_id,
        "domain": "IMACEC",
        "year": year,
        "freq": data.get("meta", {}).get("freq_effective"),
        "data_full": data,
        "data_full_original_annual": data,
        "metric_type": "annual",
    })
    md_block = _call_with_trace("IMACEC_MONTH_INTERVAL", _format_series_metadata_block, series_id)
    if md_block.strip():
        out.append(md_block + "\n")
    followups = [
        "¿Deseas ver la variación mensual (mes a mes) en este intervalo?",
        "¿Quieres comparar con el mismo intervalo de otro año?",
        "¿Necesitas un gráfico de la variación anual de estos meses?",
    ]
    out.extend([f + "\n" for f in followups])
    try:
        filename_base = f"imacec_{year}_{m1:02d}_{m2:02d}"
        marker = _call_with_trace("IMACEC_MONTH_INTERVAL", _emit_csv_download_marker, table_text, filename_base, preferred_filename=f"{filename_base}.csv")
        if marker:
            out.append("\n" + marker + "\n")
    except Exception as _e_csv_int:
        logger.error(f"[CSV_MARKER_ERROR] month_interval(INTENTS) year={year} m1={m1} m2={m2} e={_e_csv_int}")
    return iter(out)

def _handle_intent_imacec_month_specific(
    classification: ClassificationResult,
    question: str,
    history_text: str,
    match: Any,
) -> Iterable[str]:
    try:
        m_name = match.group(1).lower()
        year = int(match.group(2))
    except Exception:
        return iter(["No se pudo interpretar el mes y año solicitados."])
    month = _MONTH_NAME_MAP.get(m_name)
    if not month:
        return iter(["No se reconoció el mes solicitado."])
    defaults = _load_defaults_for_domain("IMACEC") or {}
    series_id = defaults.get("cod_serie")
    freq = defaults.get("freq") or None
    if not series_id:
        return iter(["No se encontró la serie por defecto para el IMACEC."])
    firstdate = f"{year-1}-01-01"
    lastdate = f"{year}-12-31"
    data = _call_with_trace(
        "IMACEC_MONTH_SPECIFIC",
        _get_series_with_retry,
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=freq,
        agg="avg",
        retries=2,
        backoff=1.0,
    )
    if not data:
        return iter(["No fue posible conectar con la API tras reintentos. Por favor, intenta nuevamente."])
    table_text = _call_with_trace("IMACEC_MONTH_SPECIFIC", _build_single_month_row, data, year, month)
    out: List[str] = ["\n" + table_text + "\n\n"]
    _last_data_context.update({
        "series_id": series_id,
        "domain": "IMACEC",
        "year": year,
        "freq": data.get("meta", {}).get("freq_effective"),
        "data_full": data,
        "data_full_original_annual": data,
        "metric_type": "annual",
    })
    md_block = _call_with_trace("IMACEC_MONTH_SPECIFIC", _format_series_metadata_block, series_id)
    if md_block.strip():
        out.append(md_block + "\n")
    followups = [
        "¿Deseas ver la variación mensual (mes a mes)?",
        "¿Quieres comparar con otro año?",
        "¿Necesitas un gráfico de la variación anual?",
    ]
    out.extend([f + "\n" for f in followups])
    try:
        filename_base = f"imacec_{year}_{month:02d}"
        marker = _call_with_trace("IMACEC_MONTH_SPECIFIC", _emit_csv_download_marker, table_text, filename_base, preferred_filename=f"{filename_base}.csv")
        if marker:
            out.append("\n" + marker + "\n")
    except Exception as _e_csv_m:
        logger.error(f"[CSV_MARKER_ERROR] month_specific(INTENTS) year={year} month={month} e={_e_csv_m}")
    return iter(out)

def _handle_intent_toggle_monthly(
    classification: ClassificationResult,
    question: str,
    history_text: str,
    match: Any,
) -> Iterable[str]:
    if not (_last_data_context.get("data_full") and _last_data_context.get("metric_type") != "monthly"):
        return iter([])
    year_ctx = _last_data_context.get("year") or _get_latest_year_from_data(_last_data_context.get("data_full") or {})
    data_ctx = _last_data_context.get("data_full")
    if not (year_ctx and data_ctx):
        return iter([])
    table_text = _call_with_trace("TOGGLE_MONTHLY", _build_year_mom_table, data_ctx, int(year_ctx))
    out: List[str] = ["\n" + table_text + "\n\n"]
    _last_data_context["metric_type"] = "monthly"
    try:
        filename_base = f"{(_last_data_context.get('domain') or 'serie').lower()}_{year_ctx}_mom"
        marker = _call_with_trace("TOGGLE_MONTHLY", _emit_csv_download_marker, table_text, filename_base, preferred_filename=f"{filename_base}.csv")
        if marker:
            out.append("\n" + marker + "\n")
    except Exception as _e_csv_mom:
        logger.error(f"[CSV_MARKER_ERROR] toggle_monthly(INTENTS) year={year_ctx} e={_e_csv_mom}")
    out.extend([
        "¿Quieres volver a la variación anual (interanual)?\n",
        "¿Deseas generar un gráfico de esta variación mensual?\n",
        "¿Necesitas cambiar la frecuencia (trimestral/anual)?\n",
    ])
    return iter(out)

def _handle_intent_toggle_annual(
    classification: ClassificationResult,
    question: str,
    history_text: str,
    match: Any,
) -> Iterable[str]:
    if not (_last_data_context.get("data_full") and _last_data_context.get("metric_type") == "monthly"):
        return iter([])
    year_ctx = _last_data_context.get("year") or _get_latest_year_from_data(_last_data_context.get("data_full") or {})
    data_ctx = _last_data_context.get("data_full_original_annual") or _last_data_context.get("data_full")
    if not (year_ctx and data_ctx):
        return iter([])
    table_text = _call_with_trace("TOGGLE_ANNUAL", _build_year_yoy_simple_table, data_ctx, int(year_ctx))
    out: List[str] = ["\n" + table_text + "\n\n"]
    _last_data_context["metric_type"] = "annual"
    try:
        filename_base = f"{(_last_data_context.get('domain') or 'serie').lower()}_{year_ctx}_yoy"
        marker = _call_with_trace("TOGGLE_ANNUAL", _emit_csv_download_marker, table_text, filename_base, preferred_filename=f"{filename_base}.csv")
        if marker:
            out.append("\n" + marker + "\n")
    except Exception as _e_csv_yoy:
        logger.error(f"[CSV_MARKER_ERROR] toggle_annual(INTENTS) year={year_ctx} e={_e_csv_yoy}")
    out.extend([
        "¿Quieres ver nuevamente la variación mensual?\n",
        "¿Deseas generar un gráfico de la variación anual?\n",
        "¿Buscas consultar otro indicador económico?\n",
    ])
    return iter(out)

def _handle_intent_chart_request(
    classification: ClassificationResult,
    question: str,
    history_text: str,
    match: Any,
) -> Iterable[str]:
    requested_domain = _extract_chart_domain(question) or (_last_data_context.get("domain") or classification.data_domain)
    prev_domain = _last_data_context.get("domain")
    prev_data = _last_data_context.get("data_full")
    use_data: Optional[Dict[str, Any]] = None
    use_domain = (requested_domain or (prev_domain or "")).upper() if requested_domain or prev_domain else None
    if prev_data and prev_domain and use_domain == prev_domain:
        use_data = prev_data
    else:
        year_sel = _last_data_context.get("year") or _dt.datetime.now().year
        if use_domain in ("IMACEC", "PIB"):
            use_data = _call_with_trace("CHART_REQUEST", _fetch_series_for_year, use_domain, int(year_sel))
            if use_data:
                _last_data_context.update({
                    "domain": use_domain,
                    "year": int(year_sel),
                    "freq": (use_data.get("meta", {}) or {}).get("freq_effective"),
                    "data_full": use_data,
                })
    if use_data and use_domain:
        marker = _call_with_trace("CHART_REQUEST", _emit_chart_marker, use_domain, use_data)
        if marker:
            return iter(["Se genera un gráfico a partir de la serie solicitada.\n" + marker])
        return iter(["No se pudo construir el gráfico por falta de datos disponibles."])
    return iter(["No fue posible recuperar datos para generar el gráfico. Intenta consultar primero los datos del indicador."])

def _handle_intent_imacec_index_yoy(
    classification: ClassificationResult,
    question: str,
    history_text: str,
    match: Any,
) -> Iterable[str]:
    domain = "IMACEC"
    year_sel = _last_data_context.get("year") or _dt.datetime.now().year
    data = _last_data_context.get("data_full")
    if not data or (_last_data_context.get("domain") or "").upper() != domain:
        data = _call_with_trace("IMACEC_INDEX_YOY", _fetch_series_for_year, domain, int(year_sel))
        if not data:
            return iter(["No fue posible conectar con la API tras reintentos. Por favor, intenta nuevamente."])
        y_auto = _call_with_trace("IMACEC_INDEX_YOY", _get_latest_year_from_data, data) or int(year_sel)
        year_sel = y_auto
        _last_data_context.update({
            "series_id": (_load_defaults_for_domain(domain) or {}).get("cod_serie"),
            "domain": domain,
            "year": int(year_sel),
            "freq": (data.get("meta", {}) or {}).get("freq_effective"),
            "data_full": data,
            "data_full_original_annual": data,
            "metric_type": "annual",
        })
    table_text = _call_with_trace("IMACEC_INDEX_YOY", _build_year_index_yoy_table, data, int(year_sel))
    out: List[str] = ["\n" + table_text + "\n\n"]
    md_block = _call_with_trace("IMACEC_INDEX_YOY", _format_series_metadata_block, _last_data_context.get("series_id") or "")
    if md_block.strip():
        out.append(md_block + "\n")
    followups = [
        "¿Quieres ver solo la variación anual (interanual) del año actual?",
        "¿Deseas alternar a variación mensual (mes a mes)?",
        "¿Necesitas un gráfico con estos datos?",
    ]
    out.extend([f + "\n" for f in followups])
    try:
        filename_base = f"imacec_{int(year_sel)}_index_yoy"
        marker = _call_with_trace("IMACEC_INDEX_YOY", _emit_csv_download_marker, table_text, filename_base, preferred_filename=f"{filename_base}.csv")
        if marker:
            out.append("\n" + marker + "\n")
    except Exception as _e_csv_idx:
        logger.error(f"[CSV_MARKER_ERROR] index_yoy(INTENTS) year={year_sel} e={_e_csv_idx}")
    return iter(out)

# ---------------------------------------------------------------
# Detección: Contribución sectores IMACEC
# ---------------------------------------------------------------

def _detect_imacec_sector_contribution(question: str) -> bool:
    """Detecta intención de consulta sobre contribución/aporte de sectores a la variación del IMACEC.

    Palabras clave:
      - 'contribución' + 'variación' + 'imacec'
      - 'aportes' + 'imacec'
      - 'aporte de los sectores al imacec'
    """
    if not question:
        return False
    q = question.lower()
    if "imacec" not in q:
        return False
    pat_list = [
        r"contribuci[óo]n.*variaci[óo]n.*imacec",
        r"aportes?.*imacec",
        r"aporte de los sectores.*imacec",
    ]
    return any(re.search(pat, q) for pat in pat_list)

def _stream_imacec_sector_contribution(question: str) -> Iterable[str]:
    """Emite respuesta metodológica sobre contribución de sectores al IMACEC.

    Incluye listado de sectores principales y referencia oficial. Luego tres preguntas de seguimiento.
    """
    sectors = [
        "Minería",
        "Industria manufacturera",
        "Comercio",
        "Servicios empresariales y financieros",
        "Transporte y almacenamiento",
        "Restaurantes y hoteles",
        "Construcción",
        "Agropecuario-silvícola",
        "Electricidad, gas y agua",
    ]
    ref_link = "https://www.bcentral.cl/areas/estadisticas/imacec"
    intro = (
        "La contribución de los distintos sectores a la variación del IMACEC se estima ponderando la evolución de cada actividad "
        "según su participación en el valor agregado. Esto permite identificar qué sectores explican en mayor medida las variaciones mensuales e interanuales del índice. "
        "A continuación se listan sectores agregados que suelen analizarse en descomposiciones del IMACEC:"
    )
    yield intro + "\n\n"
    for s in sectors:
        yield f"- {s}\n"
    yield "\nReferencia oficial de metodología y series: " + ref_link + "\n\n"
    # Preguntas de seguimiento
    followups = [
        "¿Quieres consultar la variación anual del IMACEC en el año actual?",
        "¿Deseas ver la variación mes a mes (mensual) del IMACEC?",
        "¿Te interesa generar un gráfico de la variación anual?",
    ]
    for f in followups:
        yield f + "\n"

# ---------------------------------------------------------------
# Detección: variación mensual / mes a mes (toggle de métrica)
# ---------------------------------------------------------------

def _detect_monthly_variation_request(question: str) -> bool:
    if not question:
        return False
    q = question.lower()
    return bool(re.search(r"variaci[óo]n (mensual|mes a mes)|mes a mes", q))

def _detect_annual_variation_request(question: str) -> bool:
    if not question:
        return False
    q = question.lower()
    # Revertir a anual si menciona explicitamente variación anual / interanual
    return bool(re.search(r"variaci[óo]n anual|interanual", q))

# ---------------------------------------------------------------
# Tabla: variación mensual (pct) para un año
# ---------------------------------------------------------------

def _build_year_mom_table(data: Dict[str, Any], year: int) -> str:
    """Construye tabla 'Periodo | Variación mensual' usando pct para el año dado."""
    obs = (data or {}).get("observations") or []
    try:
        logger.info(f"[CALC_TRACE_ENTER] file=orchestrator.py func=_build_year_mom_table args={{'year': {year}, 'obs_len': {len(obs)}}}")
    except Exception:
        pass
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
        pct_val = o.get("pct")
        if pct_val is None:
            continue
        rows.append((str(d), float(pct_val)))
    lines: List[str] = []
    lines.append("Periodo | Variación mensual")
    lines.append("----|-----------------")
    for d, pct_val in sorted(rows, key=lambda r: r[0]):
        lines.append(f"{d} | {pct_val:.1f}")
    for ln in lines[2:5]:
        logger.info(f"[DATA_TABLE_CONTENT_YEAR_MOM] {ln}")
    table = "\n".join(lines)
    try:
        logger.info(f"[CALC_TRACE_EXIT] file=orchestrator.py func=_build_year_mom_table result={{'type':'str','lines': {len(lines)}}}")
    except Exception:
        pass
    return table

def _build_year_index_yoy_table(data: Dict[str, Any], year: int) -> str:
    """Construye tabla 'Periodo | Índice | Variación anual' para el año dado."""
    obs = (data or {}).get("observations") or []
    try:
        logger.info(f"[CALC_TRACE_ENTER] file=orchestrator.py func=_build_year_index_yoy_table args={{'year': {year}, 'obs_len': {len(obs)}}}")
    except Exception:
        pass
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
        val = o.get("value")
        yoy = o.get("yoy_pct")
        if val is None and yoy is None:
            continue
        try:
            val_fmt = f"{float(val):,.2f}".replace(",","_").replace("_", ".") if val is not None else "--"
        except Exception:
            val_fmt = str(val)
        yoy_fmt = f"{float(yoy):.1f}%" if yoy is not None else "--"
        rows.append((str(d), val_fmt, yoy_fmt))
    lines: List[str] = []
    lines.append("Periodo | Índice | Variación anual")
    lines.append("--------|--------|----------------")
    for d, val_fmt, yoy_fmt in sorted(rows, key=lambda r: r[0]):
        lines.append(f"{d} | {val_fmt} | {yoy_fmt}")
    if len(lines) <= 2:
        lines.append("(Sin observaciones en el año solicitado)")
    for ln in lines[2:6]:
        logger.info(f"[DATA_TABLE_CONTENT_INDEX_YOY] {ln}")
    table = "\n".join(lines)
    try:
        logger.info(f"[CALC_TRACE_EXIT] file=orchestrator.py func=_build_year_index_yoy_table result={{'type':'str','lines': {len(lines)}}}")
    except Exception:
        pass
    return table

# ---------------------------------------------------------------
# Detección: mes específico + año para IMACEC (enero 2025, etc.)
# ---------------------------------------------------------------

_MONTH_NAME_MAP = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,  # variante
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

def _detect_specific_month_year(question: str) -> Optional[Tuple[int, int]]:
    """Retorna (year, month) si se menciona un mes en español + año (solo IMACEC)."""
    if not question:
        return None
    q = question.lower()
    if "imacec" not in q:
        return None
    # Buscar patrón 'mes <de?> año'
    for mname, mnum in _MONTH_NAME_MAP.items():
        pat = rf"{mname}\s+(?:de\s+)?((?:19|20)\d{{2}})"
        mt = re.search(pat, q)
        if mt:
            try:
                year = int(mt.group(1))
                return (year, mnum)
            except Exception:
                continue
    return None

def _build_single_month_row(data: Dict[str, Any], year: int, month: int) -> str:
    """Tabla de una fila para mes específico: Periodo | Valor | Variación anual."""
    obs = (data or {}).get("observations") or []
    try:
        logger.info(f"[CALC_TRACE_ENTER] file=orchestrator.py func=_build_single_month_row args={{'year': {year}, 'month': {month}, 'obs_len': {len(obs)}}}")
    except Exception:
        pass
    target_date_prefix = f"{year:04d}-{month:02d}"
    row_sel = None
    for o in obs:
        d = o.get("date")
        if not d:
            continue
        if str(d).startswith(target_date_prefix):
            row_sel = o
    if not row_sel:
        return "No se encontró dato para el mes solicitado."
    val = row_sel.get("value")
    yoy = row_sel.get("yoy_pct")
    try:
        val_fmt = f"{float(val):,.2f}".replace(",","_").replace("_", ".") if val is not None else "--"
    except Exception:
        val_fmt = str(val)
    yoy_fmt = f"{float(yoy):.1f}%" if yoy is not None else "--"
    lines = [
        "Periodo | Valor | Variación anual",
        "--------|-------|----------------",
        f"{year:04d}-{month:02d} | {val_fmt} | {yoy_fmt}",
    ]
    for ln in lines:
        logger.info(f"[DATA_TABLE_CONTENT_MONTH_SPECIFIC] {ln}")
    try:
        logger.info(f"[CALC_TRACE_EXIT] file=orchestrator.py func=_build_single_month_row result={{'type':'str','lines': {len(lines)}}}")
    except Exception:
        pass
    return "\n".join(lines)

# ---------------------------------------------------------------
# Detección: intervalo de meses dentro de un año para IMACEC
# Ejemplos: "desde enero a junio de 2025", "entre marzo y agosto 2024"
# ---------------------------------------------------------------

def _detect_month_interval(question: str) -> Optional[Tuple[int, int, int]]:
    """Retorna (year, month_start, month_end) si detecta intervalo de meses IMACEC.

    Patrones soportados (en minúsculas, tildes opcionales):
      - desde <mes> a|hasta <mes> de <año>
      - entre <mes> y <mes> de <año>
      - ... <mes> a|hasta <mes> <año> (sin 'de')
    Requiere que se mencione 'imacec' en la consulta.
    """
    if not question:
        return None
    q = question.lower()
    if "imacec" not in q:
        return None
    # Construir opciones de meses en regex
    mnames = "|".join(map(re.escape, _MONTH_NAME_MAP.keys()))
    # Patrones: desde ... a/hasta ..., entre ... y ...
    pats = [
        rf"desde\s+({mnames})\s+(?:a|hasta)\s+({mnames})\s+(?:de\s+)?((?:19|20)\d{{2}})",
        rf"entre\s+({mnames})\s+y\s+({mnames})\s+(?:de\s+)?((?:19|20)\d{{2}})",
    ]
    for pat in pats:
        mt = re.search(pat, q)
        if mt:
            m1_name, m2_name, ytxt = mt.group(1), mt.group(2), mt.group(3)
            y = None
            try:
                y = int(ytxt)
            except Exception:
                y = None
            if y is None:
                continue
            m1 = _MONTH_NAME_MAP.get(m1_name, None)
            m2 = _MONTH_NAME_MAP.get(m2_name, None)
            if not m1 or not m2:
                continue
            if m1 > m2:
                # Normalizar: si el usuario invierte el orden, intercambiamos
                m1, m2 = m2, m1
            return (y, m1, m2)
    return None

def _build_month_interval_yoy_table(data: Dict[str, Any], year: int, m1: int, m2: int) -> str:
    """Tabla 'Periodo | Variación anual' para meses [m1..m2] del año dado."""
    obs = (data or {}).get("observations") or []
    try:
        logger.info(f"[CALC_TRACE_ENTER] file=orchestrator.py func=_build_month_interval_yoy_table args={{'year': {year}, 'm1': {m1}, 'm2': {m2}, 'obs_len': {len(obs)}}}")
    except Exception:
        pass
    rows: List[Tuple[str, float]] = []
    for o in obs:
        d = o.get("date")
        if not d:
            continue
        try:
            y = int(str(d)[:4])
            m = int(str(d)[5:7])
        except Exception:
            continue
        if y != year or m < m1 or m > m2:
            continue
        yoy = o.get("yoy_pct")
        if yoy is None:
            continue
        rows.append((str(d), float(yoy)))
    lines: List[str] = []
    lines.append(f"Intervalo {year}-{m1:02d} a {year}-{m2:02d}")
    lines.append("Periodo | Variación anual")
    lines.append("----|-----------------")
    for d, yoy in sorted(rows, key=lambda r: r[0]):
        lines.append(f"{d} | {yoy:.1f}")
    if len(lines) <= 3:
        lines.append("(Sin observaciones en el intervalo solicitado)")
    for ln in lines[3:6]:
        logger.info(f"[DATA_TABLE_CONTENT_YOY_INTERVAL] {ln}")
    table = "\n".join(lines)
    try:
        logger.info(f"[CALC_TRACE_EXIT] file=orchestrator.py func=_build_month_interval_yoy_table result={{'type':'str','lines': {len(lines)}}}")
    except Exception:
        pass
    return table

def _extract_chart_domain(question: str) -> Optional[str]:
    """Extrae dominio explícito para gráficos desde el texto (IMACEC/PIB)."""
    q = (question or "").lower()
    if "imacec" in q:
        return "IMACEC"
    if re.search(r"\bpib\b", q):
        return "PIB"
    return None

# ---------------------------------------------------------------
# Calendario de publicaciones IMACEC / PIB
# ---------------------------------------------------------------

_CALENDAR_IMACEC_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "catalog", "imacec_calendar.json")
_CALENDAR_PIB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "catalog", "pib_calendar.json")

def _detect_calendar_request(question: str) -> Optional[str]:
    """Detecta si la consulta es de calendario y retorna dominio ('IMACEC'|'PIB')."""
    q = (question or "").lower()
    if re.search(r"calendario|cuando se publica|cuándo se publica|fecha de publicaci", q):
        if "imacec" in q:
            return "IMACEC"
        if re.search(r"\bpib\b", q):
            return "PIB"
    return None

def _load_calendar(domain: str) -> List[Dict[str, Any]]:
    path = _CALENDAR_IMACEC_PATH if domain == "IMACEC" else _CALENDAR_PIB_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[CALENDAR_LOAD_ERROR] domain={domain} e={e}")
        return []

def _build_calendar_table(domain: str, entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "No hay entradas de calendario disponibles."
    if domain == "IMACEC":
        header = "Fecha | Publicación\n------|------------"
        lines = [header]
        for e in entries:
            d = e.get("release_date")
            pub = e.get("publication")
            lines.append(f"{d} | {pub}")
    else:  # PIB
        header = "Fecha | Publicación\n------|------------"
        lines = [header]
        for e in entries:
            lines.append(f"{e.get('release_date')} | {e.get('publication')}")
    return "\n".join(lines)

def _calendar_recommendations(domain: str) -> str:
    if domain == "IMACEC":
        return (
            "Recomendación: puedes consultar la variación mensual más reciente del IMACEC, "
            "solicitar un gráfico del año actual o pedir cambiar la frecuencia a trimestral."
        )
    return (
        "Recomendación: puedes consultar el último crecimiento trimestral del PIB, "
        "pedir el calendario regional completo o comparar con el año anterior."
    )

def _stream_imacec_latest_flow(
    classification: ClassificationResult,
    question: str,
    history_text: str,
) -> Iterable[str]:
    """Flujo especial para consultas genéricas tipo "¿Cuál es el valor del IMACEC?".

    Comportamiento:
    - Detecta y usa la serie IMACEC por defecto.
    - Obtiene datos del rango amplio reciente (últimos 2 años).
    - Identifica el último año completo disponible.
    - Emite:
      1) Un párrafo introductorio (fase 1) usando get_data_first_phase_instruction.
      2) Una tabla reducida solo con la variación anual del último año.
      3) El bloque de metadatos de la serie.
      4) Frase con el último valor de variación anual + sugerencias (fase 2).
    """
    domain = "IMACEC"
    # Resolver vía catálogo JSON sin vector aún (para poder emitir descripción primero si falla)
    series_id, freq, agg = resolve_series_for_key(question, domain, vector_fallback=False)
    logger.info(f"[IMACEC_LATEST] question_without_year series_id={series_id} freq={freq}")

    # Si no se resolvió por JSON, emitir introducción metodológica y luego intentar vector
    if not series_id:
        mode_instruction = get_data_first_phase_instruction()
        human_text = (
            "Historial de la conversación (puede estar vacío):\n"
            f"{history_text}\n\n"
            "Consulta actual del usuario:\n"
            f"{question}\n\n"
            "Instrucción de modo (fase 1 de datos):\n"
            f"{mode_instruction}\n"
        )
        try:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
            except ImportError:
                from langchain.schema import SystemMessage, HumanMessage  # type: ignore
            msg_intro = _llm_data.invoke([
                SystemMessage(content=_DATA_SYSTEM),
                HumanMessage(content=human_text),
            ])
            raw_intro = getattr(msg_intro, "content", None) or getattr(msg_intro, "text", None) or ""
            intro_text = _sanitize_llm_text(str(raw_intro))
            yield intro_text + "\n"
        except Exception as e:
            logger.error(f"[IMACEC_LATEST_INTRO_FALLBACK] {e}")
            yield "Descripción metodológica no disponible temporalmente."

        # Intentar vector ahora
        v_sid, v_freq, v_agg = resolve_series_for_key(question, domain, vector_fallback=True)
        if not v_sid:
            # Log ya habrá marcado ausencia de vector o sin resultados
            yield (
                "No encontré una serie IMACEC específica en el catálogo y la búsqueda vectorial no está disponible o no devolvió coincidencias. "
                "Para habilitarla revisa la función 'search_serie_pg_vector' en 'search.py' o agrega sinónimos/códigos al JSON 'series_default'."
            )
            return
        series_id, freq, agg = v_sid, v_freq, v_agg
        logger.info(f"[IMACEC_LATEST_VECTOR_USED] sid={series_id} freq={freq} agg={agg}")

    # Excepción: consulta "indices y variacion anual" sin año -> tabla comparativa completa
    q_low = (question or "").lower()
    want_full_comparison = (
        ("indices" in q_low or "índices" in q_low) and ("variacion anual" in q_low or "variación anual" in q_low) and _extract_year(question) is None
    )

    # Fase 1: párrafo introductorio breve sobre el IMACEC (no streaming granular)
    mode_instruction = get_data_first_phase_instruction()
    human_text = (
        "Historial de la conversación (puede estar vacío):\n"
        f"{history_text}\n\n"
        "Consulta actual del usuario:\n"
        f"{question}\n\n"
        "Instrucción de modo (fase 1 de datos):\n"
        f"{mode_instruction}\n"
    )
    try:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
        except ImportError:
            from langchain.schema import SystemMessage, HumanMessage  # type: ignore
        msg_intro = _llm_data.invoke([
            SystemMessage(content=_DATA_SYSTEM),
            HumanMessage(content=human_text),
        ])
        raw_intro = getattr(msg_intro, "content", None) or getattr(msg_intro, "text", None) or ""
        intro_text = _sanitize_llm_text(str(raw_intro))
        yield intro_text + "\n"
    except Exception as e:
        logger.error(f"[IMACEC_LATEST_STREAM_INTRO_ERROR] {e}")
        intro_text = ""
        yield "No fue posible generar la introducción del IMACEC.\n"
    logger.info(f"[IMACEC_LATEST] intro_len={len(intro_text)}")

    # Fetch de datos: tomamos un rango de últimos 2 años para encontrar el último año completo
    today = _dt.date.today()
    start_year = today.year - 2
    firstdate = f"{start_year}-01-01"
    lastdate = f"{today.year}-12-31"
    try:
        from get_series import build_year_comparison_table_text
    except Exception as e:
        logger.error(f"[IMACEC_LATEST] import get_series falló: {e}")
        yield "No fue posible obtener los datos más recientes del IMACEC en este momento."
        return

    logger.info(f"[IMACEC_LATEST] fetching series_id={series_id} range={firstdate}->{lastdate} freq={freq}")
    data = _get_series_with_retry(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=freq,
        agg="avg",
        retries=2,
        backoff=1.0,
    )
    if not data:
        yield "No fue posible conectar con la API tras reintentos. Por favor, intenta nuevamente."
        return

    latest_year = _get_latest_year_from_data(data)
    if not latest_year:
        logger.warning("[IMACEC_LATEST] no se pudo determinar el último año disponible")
        yield "No se pudo determinar el último año disponible del IMACEC en la base de datos."
        return

    if want_full_comparison:
        table_text = _build_year_table(data, latest_year)
        # Reemplazar encabezado 'Mes |' o 'Trimestre |' por 'Periodo |'
        _lines = table_text.split("\n")
        for _i, _ln in enumerate(_lines):
            if _ln.startswith("Mes | ") or _ln.startswith("Mes | Año"):
                _lines[_i] = _ln.replace("Mes |", "Periodo |")
            elif _ln.startswith("Trimestre |"):
                _lines[_i] = _ln.replace("Trimestre |", "Periodo |")
        table_text = "\n".join(_lines)
        _imacec_full_lines = table_text.count("\n") + 1
        logger.info(f"[IMACEC_LATEST_TABLE_FULL_COMPARISON] year={latest_year} lines={_imacec_full_lines}")
        yield "\n" + table_text + "\n\n"
    else:
        table_text = _build_latest_only_table(data)
        _imacec_one_row_lines = table_text.count("\n") + 1
        logger.info(f"[IMACEC_LATEST_TABLE_ONE_ROW] lines={_imacec_one_row_lines}")
        yield "\n" + table_text + "\n\n"
    # Guardar contexto completo para soportar cambio de frecuencia posterior
    _last_data_context.update({
        "series_id": series_id,
        "domain": domain,
        "year": latest_year,
        "freq": data.get("meta", {}).get("freq_effective"),
        "data_full": data,
    })

    # Metadatos de la serie
    md_block = _format_series_metadata_block(series_id)
    if md_block.strip():
        logger.info(f"[IMACEC_LATEST_META] series_id={series_id}")
        yield md_block + "\n"

    # Frase con última variación anual disponible: se usará como input explícito para fase 2
    last_yoy_text = _format_last_yoy_from_table(data, latest_year)
    if last_yoy_text:
        logger.info(f"[IMACEC_LATEST] last_yoy year={latest_year} text={last_yoy_text}")

    # Fase 2: usar el mismo mecanismo que _summarize_with_llm, pero con contexto adicional
    instruction = get_data_second_phase_instruction()
    system_msg = (
        "Eres el asistente económico del Banco Central de Chile (PIBot). Responde SIEMPRE en español. "
        "Estás en la FASE 2 de una respuesta orientada a DATOS para el IMACEC. "
        "La tabla de comparación ya fue mostrada al usuario. Ahora SOLO debes: primero mencionar explícitamente "
        "el último valor de variación anual del IMACEC que te doy, en una frase muy breve y neutra, y luego presentar "
        "tres preguntas de seguimiento, sin análisis ni interpretaciones adicionales. "
        "NO repitas la tabla, NO inventes números nuevos, NO menciones años futuros al {year} ni proyectes resultados. "
        "La salida debe ser SOLO texto plano: una breve frase con el último valor y luego exactamente tres preguntas en líneas separadas."
    ).format(year=latest_year)

    last_yoy_info = last_yoy_text or "No se dispone de una variación anual identificable para el último año."  # fallback seguro
    human_msg = (
        f"Dominio: IMACEC\nAño consultado (último disponible): {latest_year}\n"
        f"Último valor de variación anual (para mencionar tal cual, sin cambiarlo): {last_yoy_info}\n\n"
        f"Instrucción fase 2 desde answer.py:\n{instruction}"
    )

    try:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
        except ImportError:
            from langchain.schema import SystemMessage, HumanMessage  # type: ignore
        msg_phase2 = _llm_data.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=human_msg),
        ])
        raw_p2 = getattr(msg_phase2, "content", None) or getattr(msg_phase2, "text", None) or ""
        phase2_text = _sanitize_llm_text(str(raw_p2))
    except Exception as e:
        logger.error(f"[IMACEC_LATEST_PHASE2_STREAM_ERROR] {e}")
        phase2_text = "No fue posible generar la segunda fase del IMACEC."
    logger.info(f"[IMACEC_LATEST_PHASE2] chars={len(phase2_text)}")

    # Limpiar disclaimers residuales y stream por oración
    disclaimer_patterns = [
        r"no puedo proporcionar cifras",
        r"no puedo entregar cifras",
        r"no puedo proporcionar valores",
    ]
    for pat in disclaimer_patterns:
        phase2_text = re.sub(pat, "", phase2_text, flags=re.IGNORECASE)

    sentences = re.split(r"(?<=[.!?])\s+", phase2_text.strip())
    for s in sentences:
        if s:
            yield s + "\n"

    # CSV marker final para IMACEC latest
    try:
        filename_base = f"imacec_{latest_year or 'latest'}"
        marker = _emit_csv_download_marker(table_text, filename_base, preferred_filename=f"{filename_base}.csv")
        if marker:
            yield "\n" + marker + "\n"
    except Exception as _e_footer:
        logger.error(f"[CSV_MARKER_ERROR] imacec_latest year={latest_year} e={_e_footer}")

def _stream_pib_latest_flow(
    classification: ClassificationResult,
    question: str,
    history_text: str,
) -> Iterable[str]:
    """Flujo especial para consultas genéricas tipo "¿Cuál es el valor del PIB?".

    Estructura emitida:
      1) Párrafo introductorio (fase 1 datos) neutro.
      2) Tabla: a) SI la consulta incluye 'índices' y 'variación anual' (sin año explícito) -> tabla comparativa año anterior vs año actual (como IMACEC). b) En caso contrario -> una sola fila (último período y variación anual o valor).
      3) Metadatos de la serie default PIB.
      4) Fase 2: a) En modo comparativo -> frase con último valor de variación anual + tres preguntas. b) En modo una-fila -> frase último valor + tres preguntas.
    """
    domain = "PIB"
    # Resolver serie por defecto con catálogo modular
    try:
        from series_default import resolve_series_from_text
        _sd = resolve_series_from_text(question, default_key=domain)
        if _sd:
            series_id = _sd.series_id
            freq = _sd.frequency
            logger.info(f"[PIB_DEFAULT_RESOLVE] key={domain} variant={_sd.variant} sid={series_id} freq={freq}")
        else:
            series_id = None
            freq = None
            logger.warning(f"[PIB_DEFAULT_RESOLVE] sin resolución por catálogo para key={domain}")
    except Exception as _e_sd:
        defaults = _load_defaults_for_domain(domain) or {}
        series_id = defaults.get("cod_serie")
        freq = defaults.get("freq") or None
        logger.warning(f"[PIB_DEFAULT_RESOLVE] fallback por excepción: {_e_sd}")
    logger.info(f"[PIB_LATEST] question_without_year series_id={series_id} freq={freq}")
    if not series_id:
        yield "No se encontró la serie por defecto para el PIB."
        return

    # Detectar intención de comparación completa ("índices" + "variación anual" sin año explícito)
    q_low = (question or "").lower()
    want_full_comparison = (
        ("indices" in q_low or "índices" in q_low)
        and ("variacion anual" in q_low or "variación anual" in q_low)
        and _extract_year(question) is None
    )

    # Intro (fase 1)
    mode_instruction = get_data_first_phase_instruction()
    human_text = (
        "Historial de la conversación (puede estar vacío):\n"
        f"{history_text}\n\n"
        "Consulta actual del usuario:\n"
        f"{question}\n\n"
        "Instrucción de modo (fase 1 de datos):\n"
        f"{mode_instruction}\n"
    )
    try:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
        except ImportError:
            from langchain.schema import SystemMessage, HumanMessage  # type: ignore
        msg_intro = _llm_data.invoke([
            SystemMessage(content=_DATA_SYSTEM),
            HumanMessage(content=human_text),
        ])
        raw_intro = getattr(msg_intro, "content", None) or getattr(msg_intro, "text", None) or ""
        intro_text = _sanitize_llm_text(str(raw_intro))
        yield intro_text + "\n"
    except Exception as e:
        logger.error(f"[PIB_LATEST_STREAM_INTRO_ERROR] {e}")
        intro_text = ""
        yield "No fue posible generar la introducción del PIB.\n"
    logger.info(f"[PIB_LATEST] intro_len={len(intro_text)}")

    # Fetch datos: últimos 3 años para asegurar último período y posibilitar comparación año-1 vs año
    today = _dt.date.today()
    start_year = today.year - 3
    firstdate = f"{start_year}-01-01"
    lastdate = f"{today.year}-12-31"
    logger.info(f"[PIB_LATEST] fetching series_id={series_id} range={firstdate}->{lastdate} freq={freq}")
    data = _get_series_with_retry(
        series_id=series_id,
        firstdate=firstdate,
        lastdate=lastdate,
        target_frequency=freq,
        agg="avg",
        retries=2,
        backoff=1.0,
    )
    if not data:
        yield "No fue posible conectar con la API tras reintentos. Por favor, intenta nuevamente."
        return

    latest_year = _get_latest_year_from_data(data)
    if want_full_comparison and latest_year:
        table_text = _build_year_table(data, latest_year)
        # Reemplazar encabezado 'Mes |' o 'Trimestre |' por 'Periodo |'
        _p_lines = table_text.split("\n")
        for _pi, _pl in enumerate(_p_lines):
            if _pl.startswith("Mes | ") or _pl.startswith("Mes | Año"):
                _p_lines[_pi] = _pl.replace("Mes |", "Periodo |")
            elif _pl.startswith("Trimestre |"):
                _p_lines[_pi] = _pl.replace("Trimestre |", "Periodo |")
        table_text = "\n".join(_p_lines)
        _pib_full_lines = table_text.count("\n") + 1
        logger.info(f"[PIB_LATEST_TABLE_FULL_COMPARISON] year={latest_year} lines={_pib_full_lines}")
        yield "\n" + table_text + "\n\n"
    else:
        table_text = _build_latest_only_table(data)
        _pib_one_row_lines = table_text.count("\n") + 1
        logger.info(f"[PIB_LATEST_TABLE_ONE_ROW] lines={_pib_one_row_lines}")
        yield "\n" + table_text + "\n\n"
    _last_data_context.update({
        "series_id": series_id,
        "domain": domain,
        "year": latest_year,
        "freq": data.get("meta", {}).get("freq_effective"),
        "data_full": data,
    })

    md_block = _format_series_metadata_block(series_id)
    if md_block.strip():
        logger.info(f"[PIB_LATEST_META] series_id={series_id}")
        yield md_block + "\n"

    # Fase 2: frase + 3 preguntas (modo comparativo o modo una-fila)
    obs = (data or {}).get("observations") or []
    last_obs = None
    for o in obs:
        if o.get("yoy_pct") is not None or o.get("value") is not None:
            last_obs = o
    if last_obs:
        date_last = last_obs.get("date") or "(sin fecha)"
        if last_obs.get("yoy_pct") is not None:
            last_val_text = f"La última variación anual disponible es {float(last_obs.get('yoy_pct')):.1f}% (período {date_last})."
        else:
            try:
                v_fmt = f"{float(last_obs.get('value')):,.2f}".replace(",","_").replace("_",".")
            except Exception:
                v_fmt = str(last_obs.get("value"))
            last_val_text = f"El último valor disponible es {v_fmt} (período {date_last})."
    else:
        last_val_text = "No se identificó un último valor para el PIB."

    # Mensaje de sistema adaptado si hubo comparación completa
    if want_full_comparison and latest_year:
        system_phase2 = (
            "Eres el asistente económico del Banco Central de Chile (PIBot). Responde SIEMPRE en español. "
            "La tabla comparativa año anterior vs año actual ya fue mostrada al usuario para el PIB. "
            "Ahora SOLO debes: primero mencionar explícitamente el último valor de variación anual proporcionado en una frase MUY breve y neutra, "
            "y luego presentar EXACTAMENTE tres preguntas de seguimiento neutras. No repitas la tabla ni describas sus celdas."
        )
    else:
        system_phase2 = (
            "Eres el asistente económico del Banco Central de Chile (PIBot). Genera una frase neutra con el último valor dado y luego EXACTAMENTE tres preguntas de seguimiento."  # mismo comportamiento que antes
        )
    try:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
        except ImportError:
            from langchain.schema import SystemMessage, HumanMessage  # type: ignore
        msg_phase2 = _llm_data.invoke([
            SystemMessage(content=system_phase2),
            HumanMessage(content=f"Último valor: {last_val_text}"),
        ])
        raw_p2 = getattr(msg_phase2, "content", None) or getattr(msg_phase2, "text", None) or ""
        phase2_text = _sanitize_llm_text(str(raw_p2))
    except Exception as e:
        logger.error(f"[PIB_LATEST_PHASE2_STREAM_ERROR] {e}")
        phase2_text = last_val_text + "\n¿Deseas cambiar la frecuencia?\n¿Necesitas otra serie?\n¿Quieres comparar con otro año?"

    # Normalización similar al flujo IMACEC
    lines = [l.strip() for l in phase2_text.splitlines() if l.strip()]
    if not lines:
        lines = [last_val_text]
    phrase = lines[0]
    questions = lines[1:]
    fallback_q = [
        "¿Deseas cambiar la frecuencia?",
        "¿Quieres consultar otra serie económica?",
        "¿Te interesa comparar con otro año?",
    ]
    while len(questions) < 3:
        questions.append(fallback_q[len(questions)])
    for out_line in [phrase] + questions[:3]:
        yield out_line + "\n"

    # CSV marker final para PIB latest
    try:
        if want_full_comparison and latest_year:
            filename_base = f"pib_indices_var_anual_{latest_year}"
        else:
            filename_base = "pib_latest"
        marker = _emit_csv_download_marker(table_text, filename_base, preferred_filename=f"{filename_base}.csv")
        if marker:
            yield "\n" + marker + "\n"
    except Exception as _e_footer:
        logger.error(f"[CSV_MARKER_ERROR] pib_latest e={_e_footer}")

# ---------------------------------------------------------------------------
# API pública: stream / invoke
# ---------------------------------------------------------------------------


def stream(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Iterable[str]:
    """
    Versión con STREAMING real para integrar con `st.write_stream`.
    """
    t_orch_start = time.perf_counter()
    logger.info(
        f"[FASE] start: Fase C0: Orquestador - Turno de chat | question='{question}'"
    )

    # 1) Clasificación vía función explícita
    classification, history_text = response_api_openai_type(question, history or [])
    q_type = _normalize(getattr(classification, "query_type", None))
    domain = _normalize(getattr(classification, "data_domain", None))

    # 2) Ruteo determinista modular: intenta manejar por intents/handlers
    intent_iter = intent_response(classification, question, history_text)
    if intent_iter is not None:
        for chunk in intent_iter:
            yield chunk
        t_orch_end = time.perf_counter()
        logger.info(
            f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='CONFIG_INTENT'"
        )
        return

    # Ruta temprana: intervalo de meses dentro de un año (IMACEC)
    month_interval = _detect_month_interval(question)
    if month_interval:
        year_sel, m_start, m_end = month_interval
        logger.info(f"[ROUTE] IMACEC_MONTH_INTERVAL year={year_sel} from={m_start} to={m_end}")
        _log_intent("IMACEC_MONTH_INTERVAL")
        defaults = _load_defaults_for_domain("IMACEC") or {}
        series_id = defaults.get("cod_serie")
        freq = defaults.get("freq") or None
        if not series_id:
            yield "No se encontró la serie por defecto para el IMACEC."
            return
        firstdate = f"{year_sel-1}-01-01"
        lastdate = f"{year_sel}-12-31"
        data = _get_series_with_retry(
            series_id=series_id,
            firstdate=firstdate,
            lastdate=lastdate,
            target_frequency=freq,
            agg="avg",
            retries=2,
            backoff=1.0,
        )
        if not data:
            yield "No fue posible conectar con la API tras reintentos. Por favor, intenta nuevamente."
            return
        table_text = _build_month_interval_yoy_table(data, year_sel, m_start, m_end)
        yield "\n" + table_text + "\n\n"
        _last_data_context.update({
            "series_id": series_id,
            "domain": "IMACEC",
            "year": year_sel,
            "freq": data.get("meta", {}).get("freq_effective"),
            "data_full": data,
            "data_full_original_annual": data,
            "metric_type": "annual",
        })
        md_block = _format_series_metadata_block(series_id)
        if md_block.strip():
            yield md_block + "\n"
        followups = [
            "¿Deseas ver la variación mensual (mes a mes) en este intervalo?",
            "¿Quieres comparar con el mismo intervalo de otro año?",
            "¿Necesitas un gráfico de la variación anual de estos meses?",
        ]
        for f in followups:
            yield f + "\n"
        # CSV marker para el intervalo
        try:
            filename_base = f"imacec_{year_sel}_{m_start:02d}_{m_end:02d}"
            marker = _emit_csv_download_marker(table_text, filename_base, preferred_filename=f"{filename_base}.csv")
            if marker:
                yield "\n" + marker + "\n"
        except Exception as _e_csv_int:
            logger.error(f"[CSV_MARKER_ERROR] month_interval year={year_sel} m1={m_start} m2={m_end} e={_e_csv_int}")
        t_orch_end = time.perf_counter()
        logger.info(
            f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='IMACEC/MONTH_INTERVAL'"
        )
        return

    # Ruta temprana: mes específico + año (IMACEC) — evaluar después de intervalos
    month_spec = _detect_specific_month_year(question)
    if month_spec:
        year_sel, month_sel = month_spec
        logger.info(f"[ROUTE] IMACEC_MONTH_SPECIFIC year={year_sel} month={month_sel}")
        _log_intent("IMACEC_MONTH_SPECIFIC")
        defaults = _load_defaults_for_domain("IMACEC") or {}
        series_id = defaults.get("cod_serie")
        freq = defaults.get("freq") or None
        if not series_id:
            yield "No se encontró la serie por defecto para el IMACEC."
            return
        firstdate = f"{year_sel-1}-01-01"
        lastdate = f"{year_sel}-12-31"
        data = _get_series_with_retry(
            series_id=series_id,
            firstdate=firstdate,
            lastdate=lastdate,
            target_frequency=freq,
            agg="avg",
            retries=2,
            backoff=1.0,
        )
        if not data:
            yield "No fue posible conectar con la API tras reintentos. Por favor, intenta nuevamente."
            return
        table_text = _build_single_month_row(data, year_sel, month_sel)
        yield "\n" + table_text + "\n\n"
        _last_data_context.update({
            "series_id": series_id,
            "domain": "IMACEC",
            "year": year_sel,
            "freq": data.get("meta", {}).get("freq_effective"),
            "data_full": data,
            "data_full_original_annual": data,
            "metric_type": "annual",
        })
        md_block = _format_series_metadata_block(series_id)
        if md_block.strip():
            yield md_block + "\n"
        followups = [
            "¿Deseas ver la variación mensual (mes a mes)?",
            "¿Quieres comparar con otro año?",
            "¿Necesitas un gráfico de la variación anual?",
        ]
        for f in followups:
            yield f + "\n"
        # CSV marker
        try:
            filename_base = f"imacec_{year_sel}_{month_sel:02d}"
            marker = _emit_csv_download_marker(table_text, filename_base, preferred_filename=f"{filename_base}.csv")
            if marker:
                yield "\n" + marker + "\n"
        except Exception as _e_csv_m:
            logger.error(f"[CSV_MARKER_ERROR] month_specific year={year_sel} month={month_sel} e={_e_csv_m}")
        t_orch_end = time.perf_counter()
        logger.info(
            f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='IMACEC/MONTH_SPEC'"
        )
        return

    # Toggle variación mensual -> construir tabla desde contexto
    if _detect_monthly_variation_request(question) and _last_data_context.get("data_full") and _last_data_context.get("metric_type") != "monthly":
        year_ctx = _last_data_context.get("year") or _get_latest_year_from_data(_last_data_context.get("data_full") or {})
        data_ctx = _last_data_context.get("data_full")
        if year_ctx and data_ctx:
            logger.info(f"[ROUTE] TOGGLE_TO_MONTHLY year={year_ctx}")
            _log_intent("TOGGLE_MONTHLY")
            table_text = _build_year_mom_table(data_ctx, int(year_ctx))
            yield "\n" + table_text + "\n\n"
            _last_data_context["metric_type"] = "monthly"
            # CSV marker mensual
            try:
                filename_base = f"{(_last_data_context.get('domain') or 'serie').lower()}_{year_ctx}_mom"
                marker = _emit_csv_download_marker(table_text, filename_base, preferred_filename=f"{filename_base}.csv")
                if marker:
                    yield "\n" + marker + "\n"
            except Exception as _e_csv_mom:
                logger.error(f"[CSV_MARKER_ERROR] toggle_monthly year={year_ctx} e={_e_csv_mom}")
            followups = [
                "¿Quieres volver a la variación anual (interanual)?",
                "¿Deseas generar un gráfico de esta variación mensual?",
                "¿Necesitas cambiar la frecuencia (trimestral/anual)?",
            ]
            for f in followups:
                yield f + "\n"
            t_orch_end = time.perf_counter()
            logger.info(
                f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='TOGGLE/MONTHLY'"
            )
            return

    # Toggle revertir a variación anual (solo si venimos de modo mensual)
    if _detect_annual_variation_request(question) and _last_data_context.get("data_full") and _last_data_context.get("metric_type") == "monthly":
        year_ctx = _last_data_context.get("year") or _get_latest_year_from_data(_last_data_context.get("data_full") or {})
        data_ctx = _last_data_context.get("data_full_original_annual") or _last_data_context.get("data_full")
        if year_ctx and data_ctx:
            logger.info(f"[ROUTE] TOGGLE_TO_ANNUAL year={year_ctx}")
            _log_intent("TOGGLE_ANNUAL")
            table_text = _build_year_yoy_simple_table(data_ctx, int(year_ctx))
            yield "\n" + table_text + "\n\n"
            _last_data_context["metric_type"] = "annual"
            try:
                filename_base = f"{(_last_data_context.get('domain') or 'serie').lower()}_{year_ctx}_yoy"
                marker = _emit_csv_download_marker(table_text, filename_base, preferred_filename=f"{filename_base}.csv")
                if marker:
                    yield "\n" + marker + "\n"
            except Exception as _e_csv_yoy:
                logger.error(f"[CSV_MARKER_ERROR] toggle_annual year={year_ctx} e={_e_csv_yoy}")
            followups = [
                "¿Quieres ver nuevamente la variación mensual?",
                "¿Deseas generar un gráfico de la variación anual?",
                "¿Buscas consultar otro indicador económico?",
            ]
            for f in followups:
                yield f + "\n"
            t_orch_end = time.perf_counter()
            logger.info(
                f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='TOGGLE/ANNUAL'"
            )
            return

    # Solicitud de gráfico basada en datos previos
    if _detect_chart_request(question):
        requested_domain = _extract_chart_domain(question) or (_last_data_context.get("domain") or domain or classification.data_domain)
        prev_domain = _last_data_context.get("domain")
        prev_data = _last_data_context.get("data_full")
        use_data: Optional[Dict[str, Any]] = None
        use_domain = (requested_domain or (prev_domain or "")).upper() if requested_domain or prev_domain else None
        # Si el dominio solicitado coincide con el previo y hay datos, reutilizamos
        if prev_data and prev_domain and use_domain == prev_domain:
            use_data = prev_data
        else:
            # Intentar obtener datos reales para el dominio solicitado y el año en contexto
            year_sel = _last_data_context.get("year") or _dt.datetime.now().year
            if use_domain in ("IMACEC", "PIB"):
                logger.info(f"[CHART_FETCH] domain={use_domain} year={year_sel}")
                use_data = _fetch_series_for_year(use_domain, int(year_sel))
                if use_data:
                    _last_data_context.update({
                        "domain": use_domain,
                        "year": int(year_sel),
                        "freq": (use_data.get("meta", {}) or {}).get("freq_effective"),
                        "data_full": use_data,
                    })
        if use_data and use_domain:
            logger.info(f"[ROUTE] CHART_REQUEST domain={use_domain} source={'reuse' if use_data is prev_data else 'fetched'}")
            _log_intent("CHART_REQUEST")
            marker = _emit_chart_marker(use_domain, use_data)
            if marker:
                yield "Se genera un gráfico a partir de la serie solicitada.\n" + marker
            else:
                yield "No se pudo construir el gráfico por falta de datos disponibles."
            t_orch_end = time.perf_counter()
            logger.info(
                f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='CHART/{use_domain}'"
            )
            return
        else:
            yield "No fue posible recuperar datos para generar el gráfico. Intenta consultar primero los datos del indicador."
            t_orch_end = time.perf_counter()
            logger.info(
                f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='CHART/NO_DATA'"
            )
            return

    # Cambio de frecuencia: interceptar ANTES de los flujos especiales
    freq_req = _detect_frequency_change(question)
    if freq_req and _last_data_context.get("series_id") and _last_data_context.get("data_full"):
        target_freq = freq_req["target_freq"]
        same_series = freq_req["same_series"]
        # Si el usuario menciona el dominio y coincide con el contexto, también lo tratamos como misma serie
        if same_series:
            logger.info(
                f"[ROUTE] FREQ_CHANGE intercept | target={target_freq} | last_series={_last_data_context.get('series_id')} | last_freq={_last_data_context.get('freq')}"
            )
            original_data = _last_data_context.get("data_full")
            year_used = _last_data_context.get("year")
            for chunk in _stream_frequency_change_table_only(
                original_data=original_data,
                target_freq=target_freq,
                domain=_last_data_context.get("domain") or domain,
                year=year_used,
            ):
                yield chunk
            t_orch_end = time.perf_counter()
            logger.info(
                f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='DATA/FREQ_CHANGE_TABLE_ONLY'"
            )
            return

    # Caso especial: DATA sobre IMACEC sin año explícito (p.ej. "¿Cuál es el valor del IMACEC?")
    if q_type == "DATA" and domain == "IMACEC" and _extract_year(question) is None:
        logger.info("[ROUTE] IMACEC_LATEST_FLOW question_without_year (stream)")
        _log_intent("IMACEC_LATEST")
        for chunk in _stream_imacec_latest_flow(
            classification=classification,
            question=question,
            history_text=history_text,
        ):
            yield chunk
        t_orch_end = time.perf_counter()
        logger.info(
            f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='DATA/IMACEC_LATEST'"
        )
        return

    # Consulta de calendario (intercepta antes de flujos de datos/metodológicos)
    cal_domain = _detect_calendar_request(question)
    if cal_domain:
        logger.info(f"[ROUTE] CALENDAR_REQUEST domain={cal_domain}")
        _log_intent("CALENDAR_REQUEST")
        entries = _load_calendar(cal_domain)
        table = _build_calendar_table(cal_domain, entries)
        yield f"Calendario de publicaciones {cal_domain} (2025)\n\n" + table + "\n\n" + _calendar_recommendations(cal_domain) + "\n"
        t_orch_end = time.perf_counter()
        logger.info(f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='CALENDAR/{cal_domain}'")
        return

    # Caso especial: DATA sobre PIB sin año explícito (p.ej. "¿Cuál es el valor del PIB?")
    if q_type == "DATA" and domain == "PIB" and _extract_year(question) is None:
        logger.info("[ROUTE] PIB_LATEST_FLOW question_without_year (stream)")
        _log_intent("PIB_LATEST")
        for chunk in _stream_pib_latest_flow(
            classification=classification,
            question=question,
            history_text=history_text,
        ):
            yield chunk
        t_orch_end = time.perf_counter()
        logger.info(
            f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='DATA/PIB_LATEST'"
        )
        return

    # Caso especial: año específico IMACEC / PIB (tabla simple Mes | Variación anual)
    year_specific = _extract_year(question)
    if (
        q_type == "DATA" and year_specific is not None and domain in ("IMACEC", "PIB")
        and classification.is_generic
    ):
        # Excepción: si el usuario pide explícitamente índices y variación anual, usar flujo estándar
        q_low = question.lower()
        if not ("indices" in q_low or "índices" in q_low) or ("variacion anual" not in q_low and "variación anual" not in q_low):
            logger.info(f"[ROUTE] YEAR_SIMPLE_FLOW domain={domain} year={year_specific}")
            _log_intent("YEAR_SIMPLE_FLOW")
            # Fase 1 metodológica breve (restaurada)
            mode_instruction = get_data_first_phase_instruction()
            wrapped_method = _wrap_phase_stream(
                phase_name="Fase C2: Respuesta metodológica (stream)",
                description="modo='primera_fase_datos/year_simple'",
                inner_iter=_stream_methodological_phase(
                    classification=classification,
                    question=question,
                    history_text=history_text,
                    mode_instruction=mode_instruction,
                ),
            )
            for chunk in wrapped_method:
                yield str(chunk)
            # Banner
            yield get_processing_banner()
            # Resolver serie específica por JSON (variant-aware) y luego fetch año-1/año
            try:
                series_id, freq, agg = resolve_series_for_key(question, domain)
                if not series_id:
                    # Fallback legacy por dominio
                    defaults = _load_defaults_for_domain(domain) or {}
                    series_id = defaults.get("cod_serie")
                    freq = defaults.get("freq") or None
                    agg = "avg"
                if not series_id:
                    raise RuntimeError("no series_id for YEAR_SIMPLE_FLOW")
                data_year = _fetch_series_for_year_by_series_id(series_id, year_specific, freq)
            except Exception as e:
                logger.error(f"[YEAR_SIMPLE_FLOW_FETCH] error domain={domain} year={year_specific} e={e}")
                yield f"No fue posible obtener datos para {domain} año {year_specific}."
                t_orch_end = time.perf_counter()
                logger.info(
                    f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='DATA/YEAR_SIMPLE_FAIL'"
                )
                return
            if not data_year:
                yield f"No se encontraron datos del {domain} para el año {year_specific}."
                t_orch_end = time.perf_counter()
                logger.info(
                    f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='DATA/YEAR_SIMPLE_EMPTY'"
                )
                return
            table_text = _build_year_yoy_simple_table(data_year, year_specific)
            yield "\n" + table_text + "\n\n"
            # Metadatos según la serie efectiva usada
            meta_sid = (data_year.get("meta") or {}).get("series_id")
            if meta_sid:
                md_block = _format_series_metadata_block(meta_sid)
                if md_block.strip():
                    yield md_block + "\n"
            # Último yoy para frase
            last_yoy_text = _format_last_yoy_from_table(data_year, year_specific) or "No se identificó una variación anual final."
            instruction2 = get_data_second_phase_instruction()
            system_msg = (
                "Eres el asistente económico del Banco Central de Chile (PIBot). Responde SIEMPRE en español. "
                "Estás en la FASE 2 para una consulta de datos anual simplificada. "
                "Debes mencionar brevemente el último valor de variación anual del año consultado y luego ofrecer tres preguntas de seguimiento. "
                "No repitas la tabla ni agregues análisis.")
            human_msg = (
                f"Dominio: {domain}\nAño consultado: {year_specific}\nÚltimo valor a mencionar tal cual: {last_yoy_text}\n\nInstrucción fase 2:\n{instruction2}" )
            # Fase 2 usando invoke + sanitización para evitar artefactos
            try:
                try:
                    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
                except ImportError:
                    from langchain.schema import SystemMessage, HumanMessage  # type: ignore
                msg_p2 = _llm_data.invoke([
                    SystemMessage(content=system_msg),
                    HumanMessage(content=human_msg),
                ])
                raw_p2 = getattr(msg_p2, "content", None) or getattr(msg_p2, "text", None) or ""
                phase2 = _sanitize_llm_text(str(raw_p2))
            except Exception as e:
                logger.error(f"[YEAR_SIMPLE_PHASE2_ERROR] {e}")
                phase2 = "No fue posible generar la segunda fase."
            sentences = re.split(r"(?<=[.!?])\s+", phase2.strip())
            for s in sentences:
                if s:
                    yield s + "\n"
            # CSV marker para YEAR_SIMPLE_FLOW
            try:
                marker = _emit_csv_download_marker(table_text, f"{domain.lower()}_{year_specific}_yoy", preferred_filename=f"{domain.lower()}_{year_specific}_yoy.csv")
                if marker:
                    yield "\n" + marker + "\n"
            except Exception as _e_footer:
                logger.error(f"[CSV_MARKER_ERROR] year_simple domain={domain} year={year_specific} e={_e_footer}")
            t_orch_end = time.perf_counter()
            logger.info(
                f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='DATA/YEAR_SIMPLE'"
            )
            return

    # Manejo especial: cambio de frecuencia solicitado
    # Si no es misma serie continúa flujo normal

    # Manejo especial: selección directa de una serie después de vector search
    m_use = re.search(r"usar serie\s+([A-Z0-9_.]+)", question.upper())
    if m_use:
        code_sel = m_use.group(1)
        year_extracted = _extract_year(question)
        year_source = "question" if year_extracted else ("vector_last" if _last_vector_year else "current_year_auto")
        year_sel = year_extracted or _last_vector_year or _dt.datetime.now().year
        logger.info(f"[VECTOR_SELECT_UI] recibido | raw_question='{question}' | code_sel='{code_sel}' | year_sel={year_sel} | year_source={year_source}")
        if not code_sel:
            logger.warning(f"[VECTOR_SELECT] code vacío tras parseo | raw_question='{question}'")
        inferred_domain = _infer_domain_from_series_id(code_sel)
        if inferred_domain and inferred_domain != domain:
            logger.info(f"[VECTOR_SELECT] domain override | old_domain={domain} | inferred={inferred_domain}")
            domain = inferred_domain
        # Forzar clasificación en modo DATA
        try:
            classification.query_type = "DATA"  # type: ignore[attr-defined]
            classification.data_domain = domain  # type: ignore[attr-defined]
        except Exception:
            logger.warning("[VECTOR_SELECT] No se pudo ajustar classification dataclass.")
        # Intro serie seleccionada
        sel_title = None
        try:
            if get_series_metadata:
                _md = get_series_metadata(code_sel)
                if _md:
                    sel_title = _md.get("title")
        except Exception:
            sel_title = None
        if sel_title:
            yield f"Has seleccionado la serie: {sel_title} ({code_sel}).\n\n"
        else:
            yield f"Has seleccionado la serie: {code_sel}.\n\n"
        # Fase metodológica breve
        mode_instruction = get_data_first_phase_instruction()
        wrapped_method = _wrap_phase_stream(
            phase_name="Fase C2: Respuesta metodológica (stream)",
            description="modo='primera_fase_datos/seleccion_vector'",
            inner_iter=_stream_methodological_phase(
                classification=classification,
                question=question,
                history_text=history_text,
                mode_instruction=mode_instruction,
            ),
        )
        for chunk in wrapped_method:
            yield str(chunk)
        # Banner
        yield get_processing_banner()
        # Fetch datos reales
        logger.info(f"[VECTOR_SELECT_FETCH] intento | series_id={code_sel} | year_range={year_sel-1}-{year_sel}")
        try:
            data_sel = _fetch_series_for_year_by_series_id(code_sel, year_sel, None)
        except Exception as e:
            logger.error(f"[VECTOR_SELECT_FETCH] excepción | series_id={code_sel} | error={e}")
            data_sel = None
        if not data_sel:
            logger.error(f"[VECTOR_SELECT_FETCH] fail | series_id={code_sel} | year_sel={year_sel} | year_source={year_source} | data_sel=None")
            yield f"No fue posible recuperar datos para la serie seleccionada ({code_sel}) en el rango {year_sel-1}-{year_sel}."
            t_orch_end = time.perf_counter()
            logger.info(f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='VECTOR_SELECT/FAIL'")
            return
        # Log éxito con tamaño observaciones
        obs_count = len(data_sel.get('observations', []) or [])
        logger.info(f"[VECTOR_SELECT_FETCH] ok | series_id={code_sel} | year_sel={year_sel} | obs={obs_count}")
        _last_data_context.update({"series_id": code_sel, "domain": domain, "year": year_sel, "freq": data_sel.get("meta", {}).get("freq_effective"), "data_full": data_sel})
        for chunk in _stream_data_phase_with_table(
            classification=classification,
            question=question,
            history_text=history_text,
            domain=domain,
            year=year_sel,
            data=data_sel,
        ):
            yield chunk
        t_orch_end = time.perf_counter()
        logger.info(f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='DATA/VECTOR_SELECT'")
        return

    # Manejo especial: vector search otra serie → solo mostrar opciones
    matches_vs = _handle_vector_search_other_series(question)
    if matches_vs:
        _last_vector_matches.clear()
        _last_vector_matches.extend(matches_vs)
        _last_vector_year = _extract_year(question) or _last_data_context.get("year") or _dt.datetime.now().year
        try:
            logger.info(f"[VECTOR_MATCHES] { [m.get('cod_serie') for m in matches_vs] }")
            yield "\n##VECTOR_MATCHES_START\n"
            yield "Coincidencias encontradas (ordenadas por similitud):\n"
            rank = 1
            for m in matches_vs:
                yield f"{rank}. {m.get('nkname_esp','')} ({m.get('cod_serie')})\n"
                rank += 1
            yield "##VECTOR_MATCHES_END\n"
            yield "Selecciona una de las series en el panel para ver sus datos.\n"
        except Exception:
            pass
        t_orch_end = time.perf_counter()
        logger.info(f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='VECTOR_MATCHES'" )
        return

    # Caso A: metodológicas
    if q_type == "METHODOLOGICAL":
        mode_instruction = get_methodological_instruction()
        _log_intent("METHODOLOGICAL")
        # 
        # Para usar RAG en respuestas metodológicas, sustituye `inner_iter` por
     
        # )
        wrapped = _wrap_phase_stream(
            phase_name="Fase C2: Respuesta metodológica (stream)",
            description="modo='metodologico'",
            inner_iter=_stream_methodological_phase(
                classification=classification,
                question=question,
                history_text=history_text,
                mode_instruction=mode_instruction,
            ),
        )
        for chunk in wrapped:
            yield chunk
        # Recomendación final para sostener flujo
        yield "\n¿Quieres saber sobre datos del IMACEC 2025?\n"

        t_orch_end = time.perf_counter()
        logger.info(
            f"[FASE] end: Fase C0: Orquestador - Turno de chat "
            f"({t_orch_end - t_orch_start:.3f}s) | resumen='METHODOLOGICAL/{domain}'"
        )
        return

    # Caso B: DATA IMACEC / PIB / PIB_REGIONAL
    if q_type == "DATA" and domain in {"IMACEC", "PIB", "PIB_REGIONAL"}:
        mode_instruction = get_data_first_phase_instruction()
        _log_intent(f"DATA/{domain}")
        year_detected = _extract_year(question)
        if year_detected:
            mode_instruction = mode_instruction + "\n(Pista interna: se detectó año y habrá datos reales; evita frases de incapacidad de proveer cifras.)"
        # Asegurar defaults para PIB y PIB_REGIONAL si el clasificador no provee árbol
        if domain in {"PIB", "PIB_REGIONAL"}:
            dflt = _load_defaults_for_domain(domain)
            if not dflt or not dflt.get("cod_serie"):
                logger.warning(f"[DEFAULTS] No se encontraron defaults para domain={domain}")
            else:
                logger.info(f"[DEFAULTS] domain={domain} series_id={dflt['cod_serie']} freq={dflt.get('freq')}")
        # Insertar referencia BDE en la intro para IMACEC
        if domain == "IMACEC":
            # Ya no se muestra referencia BDE en fase de datos; metadatos la contendrán
            pass
        wrapped_method = _wrap_phase_stream(
            phase_name="Fase C2: Respuesta metodológica (stream)",
            description="modo='primera_fase_datos'",
            inner_iter=_stream_methodological_phase(
                classification=classification,
                question=question,
                history_text=history_text,
                mode_instruction=mode_instruction,
            ),
        )
        for chunk in wrapped_method:
            if year_detected:
                # Convertir chunk a str seguro antes de aplicar regex
                chunk_str = str(chunk)
                if re.search(r"no puedo proporcionar cifras|no puedo entregar cifras", chunk_str, re.IGNORECASE):
                    continue
                yield chunk_str
            else:
                yield chunk

        logger.info(
            "[FASE] info: Fase C2.5: Transición a fase de datos | "
            "mensaje='Procesando los datos solicitados...'"
        )
        yield get_processing_banner()

        year = year_detected
        if year:
            data = _fetch_series_for_year(domain, year)
            if data:
                # Nuevo streaming tabla + resumen
                for chunk in _stream_data_phase_with_table(
                    classification=classification,
                    question=question,
                    history_text=history_text,
                    domain=domain,
                    year=year,
                    data=data,
                ):
                    yield chunk
                _last_data_context["data_full"] = data
                t_orch_end = time.perf_counter()
                logger.info(
                    f"[FASE] end: Fase C0: Orquestador - Turno de chat ({t_orch_end - t_orch_start:.3f}s) | resumen='DATA/{domain}/YEAR'"
                )
                return
            else:
                logger.warning(f"[DATA_FETCH] sin datos reales para domain={domain} year={year}; se usa fase datos genérica.")
        # Sin año o fallo fetch → prompt genérico de datos
        wrapped_data = _wrap_phase_stream(
            phase_name="Fase C3: Respuesta orientada a datos (stream)",
            description=f"data_domain='{domain}'",
            inner_iter=_stream_data_phase(
                classification=classification,
                question=question,
                history_text=history_text,
            ),
        )
        for chunk in wrapped_data:
            yield chunk

        t_orch_end = time.perf_counter()
        logger.info(
            f"[FASE] end: Fase C0: Orquestador - Turno de chat "
            f"({t_orch_end - t_orch_start:.3f}s) | resumen='DATA/{domain}'"
        )
        return

    # Caso C: otros
    mode_instruction = get_generic_instruction()
    wrapped_generic = _wrap_phase_stream(
        phase_name="Fase C2: Respuesta metodológica (stream)",
        description="modo='generico'",
        inner_iter=_stream_methodological_phase(
            classification=classification,
            question=question,
            history_text=history_text,
            mode_instruction=mode_instruction,
        ),
    )
    for chunk in wrapped_generic:
        yield chunk
    _log_intent(f"OTHER/{domain}")

    t_orch_end = time.perf_counter()
    logger.info(
        f"[FASE] end: Fase C0: Orquestador - Turno de chat "
        f"({t_orch_end - t_orch_start:.3f}s) | resumen='OTRO/{domain}'"
    )


def invoke(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    return "".join(chunk for chunk in stream(question, history=history))


def stream_answer(question: str, history: Optional[List[Dict[str, str]]] = None) -> Iterable[str]:
    """Orquestador principal expuesto a la app (wrapper simplificado).

    Aquí se enruta la consulta según su clasificación y se aplica el flujo
    especial para preguntas generales sobre IMACEC sin año explícito.
    """
    history = history or []
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    # Clasificación de la consulta
    # La firma actual de classify_query solo acepta `question`.
    cls = classify_query(question)

    # Caso especial: DATA sobre IMACEC sin año explícito (p.ej. "¿Cuál es el valor del IMACEC?")
    if (
        getattr(cls, "query_type", "") == "DATA"
        and getattr(cls, "data_domain", "") == "IMACEC"
        and _extract_year(question) is None
    ):
        logger.info("[ROUTE] IMACEC_LATEST_FLOW question_without_year")
        for chunk in _stream_imacec_latest_flow(cls, question, history_text):
            yield chunk
        return

    # ...existing routing logic for other cases (DATA con año, METHODOLOGICAL, etc.)...
