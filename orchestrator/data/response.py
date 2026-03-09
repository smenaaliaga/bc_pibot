"""Generación de respuesta en streaming para el nodo DATA usando LLM.

Recibe un payload con la pregunta del usuario, la clasificación de entidades
y las observaciones obtenidas de las series económicas, y genera una respuesta
en lenguaje natural usando el LLM (GPT) vía LangChain con streaming.

El LLM actúa como experto económico del Banco Central de Chile especializado
en PIB e IMACEC de la Base de Datos Estadística.

También incluye utilidades de formateo de períodos y mensajes cuando no se
encuentra la serie solicitada.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langgraph.types import StreamWriter

logger = logging.getLogger(__name__)


# ===========================================================================
# Formateo de períodos y mensajes informativos
# ===========================================================================

_MONTH_NAMES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}

_QUARTER_LABELS = {1: "1er trimestre", 2: "2do trimestre", 3: "3er trimestre", 4: "4to trimestre"}


def format_period_labels(period_str: str, freq: str) -> Tuple[str, ...]:
    """Convierte una fecha ISO (YYYY-MM-DD) en una etiqueta de período legible.

    Args:
        period_str: Fecha en formato ISO o año (e.g. ``"2025-03-31"``, ``"2025"``).
        freq: Frecuencia — ``"m"`` (mensual), ``"q"`` (trimestral), ``"a"`` (anual).

    Returns:
        Tupla con al menos un elemento: la etiqueta formateada.
        Retorna ``("--",)`` si no se puede parsear.
    """
    text = str(period_str or "").strip()
    if not text:
        return ("--",)

    freq_lower = str(freq or "").strip().lower()
    parts = text[:10].split("-")

    if len(parts) == 1:
        match = re.match(r"^(19|20)\d{2}$", parts[0])
        if match:
            return (f"el año {parts[0]}",)
        return ("--",)

    if len(parts) < 3:
        return ("--",)

    try:
        year = int(parts[0])
        month = int(parts[1])
    except (ValueError, IndexError):
        return ("--",)

    if month < 1 or month > 12:
        return ("--",)

    if freq_lower == "m":
        return (f"{_MONTH_NAMES.get(month, '--')} {year}",)
    if freq_lower == "q":
        quarter = ((month - 1) // 3) + 1
        return (f"{_QUARTER_LABELS.get(quarter, '--')} {year}",)
    if freq_lower == "a":
        return (f"el año {year}",)

    return (f"{_MONTH_NAMES.get(month, '--')} {year}",)


def build_no_series_message(
    *,
    question: str,
    requested_activity: Optional[str] = None,
    normalized_activity: Optional[str] = None,
    indicator_label: Optional[str] = None,
) -> str:
    """Genera un mensaje amigable cuando no se encuentra la serie solicitada."""
    parts: List[str] = []

    if indicator_label and requested_activity:
        parts.append(
            f"No encontré datos de **{indicator_label}** para la actividad "
            f"**{requested_activity}** en la Base de Datos Estadística del Banco Central."
        )
    elif indicator_label:
        parts.append(
            f"No encontré la serie de **{indicator_label}** solicitada "
            f"en la Base de Datos Estadística del Banco Central."
        )
    elif requested_activity:
        parts.append(
            f"No encontré datos para la actividad **{requested_activity}** "
            f"en la Base de Datos Estadística del Banco Central."
        )
    else:
        parts.append(
            "No pude identificar la serie solicitada en la Base de Datos Estadística "
            "del Banco Central de Chile."
        )

    if normalized_activity and normalized_activity != str(requested_activity or ""):
        parts.append(
            f"La actividad fue normalizada como «{normalized_activity}»."
        )

    parts.append(
        "Puedes reformular tu pregunta o consultar el catálogo de series disponibles."
    )

    return " ".join(parts)


def handle_no_series(
    *,
    question: str,
    entities: List[Dict[str, Any]],
    ent,
    writer: Optional[StreamWriter],
    emit_fn,
    first_non_empty_fn,
) -> Dict[str, Any]:
    """Construye la respuesta cuando no se encontró familia o serie en catálogo."""
    primary_entity = entities[0] if isinstance(entities, list) and entities else {}
    requested_activity = None
    if isinstance(primary_entity, dict):
        requested_activity = first_non_empty_fn(primary_entity.get("activity"))
        if requested_activity is not None:
            requested_activity = str(requested_activity).strip()

    if requested_activity in {None, "", "[]", "none", "null"}:
        requested_activity = None

    indicator_candidate = ent.indicator_ent
    if indicator_candidate in (None, "", [], {}, ()):
        indicator_candidate = (
            first_non_empty_fn(primary_entity.get("indicator"))
            if isinstance(primary_entity, dict)
            else None
        )
    indicator_label = str(indicator_candidate or "").strip().upper()
    if indicator_label in {"", "[]", "NONE", "NULL"}:
        indicator_label = None

    normalized_activity = str(ent.activity_ent or "").strip()
    text = build_no_series_message(
        question=question,
        requested_activity=requested_activity,
        normalized_activity=normalized_activity,
        indicator_label=indicator_label,
    )

    logger.warning("[DATA_NODE] %s", text)
    emit_fn(text, writer)
    return {
        "output": text,
        "entities": entities,
        "parsed_point": None,
        "parsed_range": None,
        "series": None,
        "data_classification": asdict(ent),
    }


# ---------------------------------------------------------------------------
# LangChain / OpenAI
# ---------------------------------------------------------------------------
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore[assignment]

try:
    from langchain.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:
    try:
        from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
    except Exception:
        SystemMessage = None  # type: ignore[assignment,misc]
        HumanMessage = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Eres un experto económico del Banco Central de Chile, especializado en PIB e IMACEC \
de la Base de Datos Estadística (BDE).

Reglas estrictas:
- Responde SIEMPRE en español.
- Usa EXCLUSIVAMENTE los datos proporcionados en el contexto de observaciones. \
No inventes ni supongas datos que no estén presentes.
- Presenta los valores numéricos con formato claro (separador de miles, decimales \
según corresponda).
- Si los datos incluyen variaciones (yoy_pct, pct), incorpóralas en tu análisis.
- Sé conciso y directo. Prioriza claridad sobre extensión.
- Si la información disponible no es suficiente para responder la pregunta, \
indícalo explícitamente.
- No hagas proyecciones ni predicciones a futuro.
- Cita la fuente como "Banco Central de Chile — Base de Datos Estadística" \
cuando presentes datos.
"""


_CONTRIBUTION_PROMPT = """\

Reglas adicionales para consultas de contribución:
- Cuando se presentan múltiples series de una familia de contribuciones, analiza TODAS \
las series para identificar cuál tiene mayor o menor contribución al crecimiento.
- Ordena las actividades económicas de mayor a menor contribución según su valor.
- Indica el valor de contribución de cada actividad relevante.
- El valor de cada serie YA está expresado en porcentaje (puntos porcentuales). \
Por ejemplo, un valor de 1.33 significa una contribución de 1,33 puntos porcentuales. \
NO multipliques ni dividas el valor; preséntalo tal cual, redondeado a 2 decimales.

REGLAS ESTRICTAS sobre el lenguaje de contribución (OBLIGATORIO):
1. "Mayor contribución" / "más aportó" / "mayor aporte" = la actividad/región con el valor POSITIVO más alto.
2. "Menor contribución" / "menos aportó" / "menor aporte" / "menos contribuyó" = la actividad/región \
con el valor POSITIVO más bajo (pero aún > 0). \
JAMÁS uses una actividad con valor negativo como respuesta a estas preguntas. \
"Menos aportar" sigue siendo aportar: implica un valor positivo, solo que pequeño.
3. Las actividades/regiones con valores NEGATIVOS no "aportan" ni "contribuyen" al crecimiento: son \
"detractoras" o "restaron" al crecimiento. NUNCA las llames "menor contribución" ni "menos aportó".
4. Si la pregunta pide "menor contribución", "menos aportó" o similar, responde EXCLUSIVAMENTE con \
la actividad/región que tiene el valor positivo más pequeño. Por separado, menciona las que tienen \
valores negativos como "detractoras del crecimiento" o que "restaron" al crecimiento.

Ejemplo: si los datos son Metropolitana=1.33, Biobío=-0.26, Coquimbo=0.01, Tarapacá=-0.25:
- "Más aportó" → Metropolitana (1,33 pp)
- "Menos aportó" → Coquimbo (0,01 pp), porque es el positivo más bajo
- Biobío y Tarapacá son DETRACTORAS (valores negativos), "restaron" al crecimiento. NO son las que "menos aportaron"
"""


def _build_system_prompt(*, calc_mode: Optional[str] = None) -> str:
    """Retorna el system prompt base. Punto de extensión para reglas futuras."""
    prompt = _SYSTEM_PROMPT
    if str(calc_mode or "").strip().lower() == "contribution":
        prompt += _CONTRIBUTION_PROMPT
    return prompt


# ---------------------------------------------------------------------------
# Formateo del contexto de observaciones para el prompt
# ---------------------------------------------------------------------------

def _format_observations_context(
    observations: Dict[str, Dict[str, Any]],
) -> str:
    """Convierte el dict de observaciones en un bloque de texto legible para el LLM."""
    if not observations:
        return "No se obtuvieron observaciones."

    # Límites para evitar overflow de contexto
    max_obs_per_freq = int(os.getenv("DATA_RESPONSE_MAX_OBS_PER_FREQ", "50"))
    max_series = int(os.getenv("DATA_RESPONSE_MAX_SERIES", "20"))

    def _format_obs_list(obs_list: list, indent: str = "  ") -> Tuple[List[str], bool]:
        """Formatea una lista de observaciones como líneas de texto.
        
        Returns:
            Tupla de (líneas formateadas, fue_truncado)
        """
        truncated = False
        if len(obs_list) > max_obs_per_freq:
            obs_list = obs_list[-max_obs_per_freq:]
            truncated = True
        
        lines: List[str] = []
        for obs in obs_list:
            date = obs.get("date", "")
            value = obs.get("value", "")
            yoy = obs.get("yoy_pct")
            pct = obs.get("pct")
            line = f"{indent}{date} | valor={value}"
            if yoy is not None:
                line += f" | var. interanual={yoy}%"
            if pct is not None:
                line += f" | var. período ant.={pct}%"
            lines.append(line)
        return lines, truncated

    parts: List[str] = []
    series_items = list(observations.items())
    if len(series_items) > max_series:
        parts.append(f"NOTA: Mostrando {max_series} de {len(series_items)} series disponibles (límite de contexto).\n")
        series_items = series_items[:max_series]
    
    for series_id, entry in series_items:
        meta = entry.get("meta") or {}
        obs_raw = entry.get("observations")

        title = (
            meta.get("descripEsp")
            or meta.get("descripIng")
            or series_id
        )
        freq = (
            meta.get("target_frequency")
            or meta.get("original_frequency")
            or "?"
        )

        header = f"Serie: {title} ({series_id}) | Frecuencia: {freq}"
        parts.append(header)

        note = meta.get("incomplete_annual_note")
        if note:
            parts.append(f"  NOTA: {note}")

        # observations puede ser dict {"A": [...], "Q": [...]} o lista plana
        if isinstance(obs_raw, dict):
            for freq_key in ("A", "Q", "M"):
                sub = obs_raw.get(freq_key)
                if sub is None:
                    continue
                freq_label = {"A": "Anual", "Q": "Trimestral", "M": "Mensual"}.get(freq_key, freq_key)
                parts.append(f"  [{freq_label}]")
                if not sub:
                    parts.append("    (sin observaciones)")
                else:
                    formatted_lines, was_truncated = _format_obs_list(sub, indent="    ")
                    parts.extend(formatted_lines)
                    if was_truncated:
                        parts.append(f"    (mostrando últimas {max_obs_per_freq} observaciones)")
        elif isinstance(obs_raw, list):
            if not obs_raw:
                parts.append("  (sin observaciones)")
            else:
                formatted_lines, was_truncated = _format_obs_list(obs_raw)
                parts.extend(formatted_lines)
                if was_truncated:
                    parts.append(f"  (mostrando últimas {max_obs_per_freq} observaciones)")
        else:
            parts.append("  (sin observaciones)")

    return "\n".join(parts)


def _format_classification_context(
    classification: Dict[str, Any],
) -> str:
    """Resume la clasificación de entidades como contexto adicional para el LLM."""
    if not classification:
        return ""

    relevant_keys = [
        ("indicator_ent", "Indicador"),
        ("frequency_ent", "Frecuencia"),
        ("seasonality_ent", "Estacionalidad"),
        ("activity_ent", "Actividad"),
        ("region_ent", "Región"),
        ("investment_ent", "Inversión"),
        ("period_ent", "Período"),
        ("price", "Precio"),
        ("req_form_cls", "Forma de consulta"),
    ]

    lines: List[str] = []
    for key, label in relevant_keys:
        val = classification.get(key)
        if val not in (None, "", [], {}, "none", "null"):
            lines.append(f"- {label}: {val}")

    return "\n".join(lines) if lines else ""


def _normalize_req_form(value: Any) -> str:
    """Normaliza req_form para comparar ramas como latest/point/range."""
    if isinstance(value, dict):
        value = value.get("label")
    return str(value or "").strip().lower()


def _iter_observations(obs_raw: Any) -> Iterable[Dict[str, Any]]:
    """Itera observaciones sin importar si vienen planas o agrupadas por frecuencia."""
    if isinstance(obs_raw, list):
        for item in obs_raw:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(obs_raw, dict):
        for sub in obs_raw.values():
            if not isinstance(sub, list):
                continue
            for item in sub:
                if isinstance(item, dict):
                    yield item


def _pick_latest_observation(
    observations: Dict[str, Dict[str, Any]],
    *,
    preferred_series_id: Optional[str],
) -> Optional[Tuple[str, Any, str]]:
    """Retorna (fecha, valor, series_id) de la observación más reciente."""
    if not isinstance(observations, dict) or not observations:
        return None

    def _latest_from_entry(series_id: str, entry: Any) -> Optional[Tuple[str, Any, str]]:
        if not isinstance(entry, dict):
            return None

        best_date = ""
        best_value: Any = None
        for obs in _iter_observations(entry.get("observations")):
            date = str(obs.get("date") or "").strip()[:10]
            if not date:
                continue
            if date >= best_date:
                best_date = date
                best_value = obs.get("value")

        if not best_date:
            return None
        return best_date, best_value, series_id

    preferred_id = str(preferred_series_id or "").strip()
    if preferred_id and preferred_id in observations:
        preferred_latest = _latest_from_entry(preferred_id, observations.get(preferred_id))
        if preferred_latest is not None:
            return preferred_latest

    best: Optional[Tuple[str, Any, str]] = None
    for sid, entry in observations.items():
        candidate = _latest_from_entry(str(sid), entry)
        if candidate is None:
            continue
        if best is None or candidate[0] >= best[0]:
            best = candidate

    return best


# ---------------------------------------------------------------------------
# Construcción de mensajes LLM
# ---------------------------------------------------------------------------

def _build_messages(payload: Dict[str, Any]) -> list:
    """Construye la lista de mensajes (system + user) para el LLM."""
    question = payload.get("question", "")
    classification = payload.get("classification") or {}
    observations = payload.get("observations") or {}
    family_name = payload.get("family_name", "")
    series_id = payload.get("series", "")
    req_form = _normalize_req_form(classification.get("req_form_cls"))

    calc_mode = classification.get("calc_mode_cls")
    system_content = _build_system_prompt(calc_mode=calc_mode)

    latest_hint = None
    if req_form == "latest":
        latest_hint = _pick_latest_observation(
            observations,
            preferred_series_id=str(series_id or "") or None,
        )
        if latest_hint is not None:
            system_content += (
                "\n\nRegla para consultas latest: menciona explicitamente el ultimo "
                "valor observado (fecha y valor) provisto en el contexto de datos."
            )

    # Contexto de clasificación
    cls_text = _format_classification_context(classification)
    if cls_text:
        system_content += f"\n\nClasificación de la consulta:\n{cls_text}"

    # Contexto de datos
    obs_text = _format_observations_context(observations)

    user_content = f"Pregunta del usuario: {question}\n\n"
    if family_name:
        user_content += f"Familia de series: {family_name}\n"
    if series_id:
        user_content += f"Serie principal: {series_id}\n"
    if latest_hint is not None:
        latest_date, latest_value, latest_series_id = latest_hint
        user_content += (
            "Consulta latest detectada. Debes mencionar este ultimo valor observado "
            f"en la respuesta: fecha={latest_date}, valor={latest_value}, "
            f"serie={latest_series_id}.\n"
        )
    user_content += f"\nDatos disponibles:\n{obs_text}"

    if SystemMessage is None or HumanMessage is None:
        return []

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]


# ---------------------------------------------------------------------------
# Streaming de la respuesta LLM
# ---------------------------------------------------------------------------

def stream_data_response(
    payload: Dict[str, Any],
) -> Iterable[str]:
    """Genera la respuesta en streaming a partir del payload.

    Args:
        payload: Diccionario con las claves:
            - question (str): pregunta original del usuario.
            - classification (dict): entidades resueltas (ResolvedEntities como dict).
            - observations (dict): observaciones indexadas por series_id.
            - family_name (str): nombre de la familia.
            - series (str): ID de la serie objetivo.
            - source_url (str): URL fuente de la serie.

    Yields:
        Fragmentos de texto de la respuesta del LLM.
    """
    logger.info("[DATA_RESPONSE] payload recibido: %s", payload)
    messages = _build_messages(payload)
    if not messages:
        yield "No se pudo construir la solicitud al modelo de lenguaje."
        return

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("DATA_RESPONSE_TEMPERATURE", "0.2"))

    if ChatOpenAI is None:
        logger.error("[DATA_RESPONSE] langchain_openai no disponible")
        yield "El servicio de generación de respuestas no está disponible."
        return

    try:
        chat = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True,
        )
    except Exception:
        logger.exception("[DATA_RESPONSE] Error inicializando ChatOpenAI")
        yield "Error al conectar con el modelo de lenguaje."
        return

    try:
        for chunk in chat.stream(messages):
            text = getattr(chunk, "content", None) or ""
            if text:
                yield str(text)
    except Exception:
        logger.exception("[DATA_RESPONSE] Error durante streaming")
        yield "Ocurrió un error al generar la respuesta."
