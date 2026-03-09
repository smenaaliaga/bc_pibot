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
from orchestrator.data._helpers import build_target_series_url

logger = logging.getLogger(__name__)

_BDE_SERIES_BROWSER_URL = (
    "https://si3.bcentral.cl/Siete/ES/Siete/Cuadro/CAP_CCNN/MN_CCNN76/CCNN2018_IMACEC_01_A"
)


# ===========================================================================
# Formateo de períodos y mensajes informativos
# ===========================================================================

_MONTH_NAMES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre",
}

_QUARTER_LABELS = {1: "1er trimestre", 2: "2do trimestre", 3: "3er trimestre", 4: "4to trimestre"}

_SEASONALITY_LABELS = {
    "sa": "desestacionalizado",
    "nsa": "empalmado (no desestacionalizado)",
}


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

    parts.append(
        f"Tambien puedes buscar y ver series en 🔗 [Catalogo BDE]({_BDE_SERIES_BROWSER_URL})."
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
- NUNCA menciones el ID o código técnico de la serie (e.g. "F032.PIB.FLU.R.CLP...") \
en la respuesta, a menos que el usuario lo solicite explícitamente. \
Refiere a la serie por su nombre descriptivo (e.g. "PIB real trimestral").
- SIEMPRE menciona el nombre descriptivo completo de la serie consultada \
(e.g. "IMACEC desestacionalizado", no solo "IMACEC"; "PIB real trimestral", \
no solo "PIB"). Usa el "Nombre de la serie" provisto en el contexto del usuario.
- No agregues una línea de fuente o citación al final de tu respuesta; \
la fuente se añade automáticamente por el sistema.
- Cierra tu respuesta con una breve frase de cierre amable y variada. \
Puede ser una oferta de ayuda, un comentario contextual sobre los datos, \
o una invitación a explorar otros indicadores. NUNCA repitas la misma frase; \
sé creativo y natural cada vez.

Regla de disponibilidad de datos:
- Si el contexto incluye una ALERTA DE DISPONIBILIDAD indicando que el período \
solicitado por el usuario aún no tiene datos publicados, debes comenzar tu respuesta \
indicando de forma natural y amable que aún no hay datos disponibles para ese período. \
Luego presenta el último dato publicado disponible como referencia. \
Ejemplo: "Aún no se han publicado datos de IMACEC para febrero de 2026. \
Sin embargo, el último dato disponible corresponde a enero de 2026, donde...".

Reglas para PIB con datos anuales y trimestrales:
- Cuando los datos incluyen observaciones agrupadas por frecuencia (Anual y Trimestral), \
analiza AMBAS frecuencias.
- Si existe una NOTA indicando que un año NO tiene los 4 trimestres completos, \
indica explícitamente que el dato anual de ese año no está disponible porque \
la serie aún no tiene los 4 trimestres publicados. Luego presenta los \
trimestres disponibles de ese año.
- Cuando el usuario pregunta por el crecimiento de un año y solo hay trimestres parciales, \
presenta cada trimestre disponible con su variación interanual.
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

2. "Menor contribución" / "menos aportó" / "menor aporte" / "menos contribuyó" SOLO se aplica a \
actividades/regiones con valores POSITIVOS. Debes responder EXCLUSIVAMENTE con la que tiene el valor \
positivo más pequeño (incluso si es cercano a cero). NUNCA uses una actividad con valor negativo para \
responder estas preguntas, porque los negativos no "aportan" — "restan".

3. Actividades/regiones con valores NEGATIVOS no "aportan" ni "contribuyen": son "detractoras" o \
"restaron al crecimiento". SOLO menciónalas si preguntan explícitamente por ellas, o como contexto \
adicional separado.

4. EJEMPLO OBLIGATORIO:
   Si los datos son: Metropolitana=1.33, Biobío=-0.26, Coquimbo=0.01, Tarapacá=-0.25
   - Pregunta "¿Quién MÁS aportó?" → Respuesta: Metropolitana (1,33 pp)
   - Pregunta "¿Quién MENOS aportó?" → Respuesta: Coquimbo (0,01 pp) [es el positivo más bajo]
   - LUEGO menciona: "Sin embargo, Biobío y Tarapacá fueron detractoras del crecimiento (-0,26 y -0,25 pp respectivamente)."
   - NUNCA digas "Menos aportó = Biobío" aunque sea un negativo más pequeño.
"""


_CALC_MODE_FIELD_RULES = {
    "original": (
        "value",
        "El campo principal a reportar es 'value' (nivel del indicador). "
        "PRESENTA PRIMERO el campo 'value' como el dato principal. "
        "En el primer párrafo pon en **negrita** ese valor (e.g. **72,48**). "
        "Las variaciones (yoy_pct, pct) son secundarias y opcionales."
    ),
    "yoy": (
        "yoy_pct",
        "El campo principal a reportar es 'yoy_pct' (variación interanual). "
        "PRESENTA PRIMERO el campo 'yoy_pct' como el dato principal. "
        "En el primer párrafo pon en **negrita** ese valor (e.g. **6,9%**). "
        "El nivel del índice ('value') es secundario: menciónalo AL FINAL, "
        "no en el primer párrafo."
    ),
    "prev_period": (
        "pct",
        "El campo principal a reportar es 'pct' (variación respecto al período anterior). "
        "PRESENTA PRIMERO el campo 'pct' como el dato principal. "
        "En el primer párrafo pon en **negrita** ese valor (e.g. **0,19%**). "
        "El nivel del índice ('value') es secundario: menciónalo AL FINAL, "
        "no en el primer párrafo."
    ),
}


def _build_system_prompt(*, calc_mode: Optional[str] = None) -> str:
    """Retorna el system prompt base. Punto de extensión para reglas futuras."""
    prompt = _SYSTEM_PROMPT
    cm = str(calc_mode or "").strip().lower()
    if cm == "contribution":
        prompt += _CONTRIBUTION_PROMPT
    rule = _CALC_MODE_FIELD_RULES.get(cm)
    if rule:
        _, rule_text = rule
        prompt += f"\n\nRegla de campo prioritario (calc_mode={cm}):\n{rule_text}"
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

    logger.info("[DATA_RESPONSE] Formateando %d series para contexto LLM", len(observations))

    def _format_obs_list(obs_list: list, indent: str = "  ", freq_code: str = "") -> Tuple[List[str], bool]:
        """Formatea una lista de observaciones como líneas de texto.
        
        Returns:
            Tupla de (líneas formateadas, fue_truncado)
        """
        truncated = False
        original_len = len(obs_list)
        if len(obs_list) > max_obs_per_freq:
            obs_list = obs_list[-max_obs_per_freq:]
            truncated = True
        
        logger.info("[DATA_RESPONSE] Formateando %d obs (original: %d, freq: %s, truncado: %s)",
                    len(obs_list), original_len, freq_code, truncated)
        
        lines: List[str] = []
        for obs in obs_list:
            raw_date = obs.get("date", "")
            date_label = format_period_labels(raw_date, freq_code)[0] if freq_code else raw_date
            value = obs.get("value", "")
            yoy = obs.get("yoy_pct")
            pct = obs.get("pct")
            line = f"{indent}{date_label} | valor={value}"
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

        # Log observaciones recibidas
        if isinstance(obs_raw, list) and obs_raw:
            first_date = obs_raw[0].get("date") if obs_raw else None
            last_date = obs_raw[-1].get("date") if obs_raw else None
            logger.info("[DATA_RESPONSE] Serie %s: %d obs, desde %s hasta %s",
                       series_id, len(obs_raw), first_date, last_date)
        elif isinstance(obs_raw, dict):
            for freq_key, sub in obs_raw.items():
                if isinstance(sub, list) and sub:
                    first_date = sub[0].get("date") if sub else None
                    last_date = sub[-1].get("date") if sub else None
                    logger.info("[DATA_RESPONSE] Serie %s [%s]: %d obs, desde %s hasta %s",
                               series_id, freq_key, len(sub), first_date, last_date)

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

        header = f"Serie: {title} | Frecuencia: {freq}"
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
                    _fc = {"A": "a", "Q": "q", "M": "m"}.get(freq_key, freq.lower() if freq else "")
                    formatted_lines, was_truncated = _format_obs_list(sub, indent="    ", freq_code=_fc)
                    parts.extend(formatted_lines)
                    if was_truncated:
                        parts.append(f"    (mostrando últimas {max_obs_per_freq} observaciones)")
        elif isinstance(obs_raw, list):
            if not obs_raw:
                parts.append("  (sin observaciones)")
            else:
                _fc = freq.lower() if freq and freq != "?" else ""
                formatted_lines, was_truncated = _format_obs_list(obs_raw, freq_code=_fc)
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
            if key == "seasonality_ent":
                val = _SEASONALITY_LABELS.get(str(val).strip().lower(), val)
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
# Detección de desfase entre período solicitado y datos disponibles
# ---------------------------------------------------------------------------

def _detect_period_mismatch(
    classification: Dict[str, Any],
    observations: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """Detecta si el período solicitado no tiene datos y genera un aviso legible.

    Compara el inicio del período pedido (``period_ent[0]``) con la
    ``last_available_date`` de la primera serie.  Si el período solicitado
    es posterior al último dato disponible, retorna un texto de alerta
    para inyectar en el prompt del LLM.
    """
    # Si el usuario pide "lo último disponible", no hay desfase posible
    req_form = str(classification.get("req_form_cls") or "").strip().lower()
    if req_form == "latest":
        return None

    period_ent = classification.get("period_ent")
    if not period_ent or not isinstance(period_ent, list) or len(period_ent) == 0:
        return None

    requested_start = str(period_ent[0]).strip()[:10]
    if not requested_start:
        return None

    # Tomar last_available_date de la primera serie
    last_avail: Optional[str] = None
    for entry in observations.values():
        meta = entry.get("meta")
        if isinstance(meta, dict):
            last_avail = str(meta.get("last_available_date") or "").strip()[:10]
            break

    if not last_avail:
        return None

    # Solo alertar si el período pedido empieza después de lo disponible
    if requested_start <= last_avail:
        return None

    freq = str(classification.get("frequency_ent") or "").strip().lower()
    requested_label = format_period_labels(requested_start, freq)[0]
    available_label = format_period_labels(last_avail, freq)[0]

    indicator = str(classification.get("indicator_ent") or "").strip().upper()
    if indicator in {"", "NONE", "NULL"}:
        indicator = "la serie"

    return (
        f"ALERTA DE DISPONIBILIDAD: El usuario preguntó por {requested_label}, "
        f"pero aún no hay datos publicados de {indicator} para ese período. "
        f"El último dato disponible corresponde a {available_label} (fecha {last_avail}). "
        f"Debes indicar esto al usuario de forma clara y amable, y luego presentar "
        f"los datos del último período disponible."
    )


# ---------------------------------------------------------------------------
# Construcción de mensajes LLM
# ---------------------------------------------------------------------------

def _build_messages(payload: Dict[str, Any]) -> list:
    """Construye la lista de mensajes (system + user) para el LLM."""
    question = payload.get("question", "")
    classification = payload.get("classification") or {}
    payload_price = str(payload.get("price") or "").strip().lower()
    observations = payload.get("observations") or {}
    family_name = payload.get("family_name", "")
    series_id = payload.get("series", "")
    req_form = _normalize_req_form(classification.get("req_form_cls"))

    calc_mode = classification.get("calc_mode_cls")
    cm = str(calc_mode or "").strip().lower()
    system_content = _build_system_prompt(calc_mode=calc_mode)

    series_title = str(payload.get("series_title") or "").strip()

    indicator = str(classification.get("indicator_ent") or "").strip().lower()
    classification_price = str(classification.get("price") or "").strip().lower()
    price = payload_price or classification_price
    if indicator == "pib" and price in {"enc", "co"}:
        if price == "enc":
            unit_text = "miles de millones de pesos encadenados"
        else:
            unit_text = "miles de millones de pesos a precios corrientes"
        system_content += (
            "\n\nRegla de unidades para PIB: cuando reportes cifras de NIVEL del PIB "
            f"(campo 'value', no variaciones yoy_pct/pct), indica explícitamente que están en {unit_text}."
        )

    latest_hint = None
    if req_form == "latest":
        latest_hint = _pick_latest_observation(
            observations,
            preferred_series_id=str(series_id or "") or None,
        )
        if latest_hint is not None:
            _primary_field_name = _CALC_MODE_FIELD_RULES.get(cm, ("value",))[0]
            system_content += (
                "\n\nRegla para consultas latest: menciona explícitamente el último "
                f"dato observado (fecha y campo '{_primary_field_name}') provisto en el contexto de datos."
            )

    # Contexto de clasificación
    cls_text = _format_classification_context(classification)
    if cls_text:
        system_content += f"\n\nClasificación de la consulta:\n{cls_text}"

    # Contexto de datos
    obs_text = _format_observations_context(observations)
    logger.info("[DATA_RESPONSE] Contexto de observaciones generado: %d caracteres", len(obs_text))

    # Detección de desfase período solicitado vs disponible
    period_mismatch_hint = _detect_period_mismatch(classification, observations)

    user_content = f"Pregunta del usuario: {question}\n\n"
    if period_mismatch_hint:
        user_content += f"{period_mismatch_hint}\n\n"
    if series_title:
        user_content += f"Nombre de la serie: {series_title}\n"
    if family_name:
        user_content += f"Familia de series: {family_name}\n"
    if indicator == "pib" and price in {"enc", "co"}:
        user_content += f"Precio PIB solicitado: {price}\n"
    if latest_hint is not None:
        latest_date, latest_value, _latest_series_id = latest_hint
        # Determinar campo prioritario según calc_mode
        _primary_field = {"yoy": "yoy_pct", "prev_period": "pct"}.get(cm, "value")
        _primary_value = latest_value
        _entry = observations.get(str(_latest_series_id), {})
        for _obs in _iter_observations(_entry.get("observations")):
            if str(_obs.get("date", ""))[:10] == latest_date[:10]:
                _primary_value = _obs.get(_primary_field, latest_value)
                break
        user_content += (
            f"Consulta latest detectada. El último dato disponible es: "
            f"fecha={latest_date}, {_primary_field}={_primary_value}. "
            f"DEBES mencionar {_primary_field}={_primary_value} como el valor "
            f"principal en el primer párrafo (en negrita). "
            f"El nivel del índice (value={latest_value}) menciónalo al final si es relevante.\n"
        )
    user_content += f"\nDatos disponibles:\n{obs_text}"

    logger.info("[DATA_RESPONSE] User content generado: %d caracteres", len(user_content))

    if SystemMessage is None or HumanMessage is None:
        return []

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]


def _build_source_footer(payload: Dict[str, Any]) -> Optional[str]:
    """Construye el footer de fuente con link directo a la serie en la BDE."""
    classification = payload.get("classification") or {}
    if not isinstance(classification, dict):
        classification = {}

    target_url = build_target_series_url(
        source_url=str(payload.get("source_url") or ""),
        series_id=str(payload.get("series") or ""),
        period=classification.get("period_ent"),
        req_form=str(classification.get("req_form_cls") or ""),
        frequency=str(classification.get("frequency_ent") or ""),
        calc_mode=str(classification.get("calc_mode_cls") or ""),
    )
    if not target_url:
        return None

    return (
        f"\n\n**Fuente:** 🔗 [Base de Datos Estadísticos (BDE)]({target_url}) "
        "del Banco Central de Chile."
    )


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
    source_footer = _build_source_footer(payload)
    if not messages:
        yield "No se pudo construir la solicitud al modelo de lenguaje."
    else:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("DATA_RESPONSE_TEMPERATURE", "0.2"))

        if ChatOpenAI is None:
            logger.error("[DATA_RESPONSE] langchain_openai no disponible")
            yield "El servicio de generación de respuestas no está disponible."
        else:
            try:
                chat = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    streaming=True,
                )
            except Exception:
                logger.exception("[DATA_RESPONSE] Error inicializando ChatOpenAI")
                yield "Error al conectar con el modelo de lenguaje."
            else:
                try:
                    for chunk in chat.stream(messages):
                        text = getattr(chunk, "content", None) or ""
                        if text:
                            yield str(text)
                except Exception:
                    logger.exception("[DATA_RESPONSE] Error durante streaming")
                    yield "Ocurrió un error al generar la respuesta."

    if source_footer:
        yield source_footer
