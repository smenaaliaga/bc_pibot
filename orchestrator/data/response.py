"""Generación de respuesta en streaming para el nodo DATA usando OpenAI function calling.

Recibe un payload con la pregunta del usuario y las observations (payload
completo del data_store JSON) y genera una respuesta usando herramientas
(tools) que el LLM invoca para consultar los datos del payload.

Migrado desde el patrón chatbot.py: SYSTEM_PROMPT + TOOLS + handle_tool_call
con loop de function calling y streaming de la respuesta final.

También incluye utilidades de formateo de períodos y mensajes cuando no se
encuentra la serie solicitada.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langgraph.types import StreamWriter

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

_FREQ_CODE_MAP = {
    "m": "M",
    "q": "T",
    "t": "T",
    "a": "A",
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

    quarter_match = re.fullmatch(r"((?:19|20)\d{2})-Q([1-4])", text, re.IGNORECASE)
    if quarter_match:
        year = int(quarter_match.group(1))
        quarter = int(quarter_match.group(2))
        return (f"{_QUARTER_LABELS.get(quarter, '--')} {year}",)

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
    requested_activity: Optional[str] = None,
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

    parts.append(
        "Puedes reformular tu pregunta o consultar el catálogo de series disponibles."
    )

    parts.append(
        f"También puedes buscar y ver series en 🔗 [Catálogo BDE]({_BDE_SERIES_BROWSER_URL})."
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

    text = build_no_series_message(
        requested_activity=requested_activity,
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
# OpenAI client
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# System prompt (migrado de chatbot.py — function calling aware)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Eres un asistente experto en macroeconomía chilena y en interpretación de series del Banco Central de Chile.

FECHA ACTUAL: {today}
Usa esta fecha para interpretar referencias temporales relativas:
- "el año pasado" = {last_year}, "este año" = {this_year}, "hace dos años" = {two_years_ago}.
- "el trimestre pasado" = {prev_quarter}, "este trimestre" = {current_quarter}.
- "el mes pasado" = {prev_month}.
- NUNCA calcules estos valores tú mismo. Usa EXACTAMENTE los valores indicados arriba.

Tienes acceso a herramientas que consultan un payload JSON con series económicas precalculadas.
USA SIEMPRE las herramientas para obtener datos. NUNCA inventes cifras.

HERRAMIENTAS DISPONIBLES
- list_series: lista las series disponibles con su ID, título y clasificación.
- get_series_data: obtiene datos de UNA serie para un período o rango.
- rank_series: rankea todas las series por una métrica en un período dado.
- get_extrema: obtiene máximos y mínimos históricos de una serie o de todo el cuadro.
- get_metadata: obtiene metadatos del cuadro (nombre, clasificación, unidad, frecuencia, últimos períodos, fuente).

CONTEXTO DEL CUADRO CARGADO
- Trabajas con UN solo cuadro a la vez. Cada cuadro tiene un conjunto de series
  relacionadas (ej: PIB total + sus actividades económicas componentes).
- Usa get_metadata al inicio o ante cualquier duda para conocer el alcance del cuadro.
- NO puedes cruzar datos entre cuadros distintos. Si el usuario pregunta algo fuera
  del alcance del cuadro cargado, indícalo claramente.

FLUJO DE TRABAJO
1. Interpreta la pregunta: identifica serie, período, frecuencia y métrica.
1.b. Antes de usar get_series_data o get_extrema, llama list_series para identificar
    un series_id EXACTO del cuadro.
    NUNCA inventes series_id a partir de cuadro_id, del nombre del cuadro, ni cambiando
    sufijos como .M/.T/.A.
    El parámetro frequency selecciona el bloque de datos dentro del MISMO series_id,
    y puede ser distinto de series_freq si available_frequencies lo permite.
2. Llama la herramienta adecuada para obtener los datos exactos.
3. Si el usuario pregunta por un año completo y la serie es trimestral o mensual,
   usa get_series_data con period_start y period_end para obtener TODOS los sub-períodos del año
   (ej: period_start="2024-Q1", period_end="2024-Q4" para trimestrales).
   NO uses period con un solo sub-período.
3.b. Si el usuario pide un rango de varios años, un mandato presidencial o un período histórico
    extendido, prioriza frecuencia anual ("A") si existe para la serie solicitada.
    Si no existe frecuencia anual, cubre TODOS los años y TODOS los sub-períodos relevantes
    dentro del rango pedido. NUNCA respondas un rango plurianual mostrando solo años sueltos
    o años de ejemplo si el usuario pidió todo el período.
4. Si necesitas comparar series, usa rank_series (NO consultes una por una).
5. Si necesitas aceleración, pide dos períodos consecutivos con get_series_data.
6. Responde con los datos obtenidos de las herramientas.

MÉTRICAS DISPONIBLES EN LOS DATOS
- "value": cifra o nivel de la serie en el período.
- "pct": variación % respecto al período inmediatamente anterior.
- "yoy_pct": variación % respecto al mismo período del año anterior.
- "delta_abs": cambio absoluto respecto al período anterior.
- "yoy_delta_abs": cambio absoluto respecto al mismo período del año anterior.
- "acceleration_pct": cambio en la tasa "pct" respecto al período previo.
- "acceleration_yoy": cambio en la tasa "yoy_pct" respecto al período previo.

REGLA CRÍTICA — LOS PORCENTAJES YA ESTÁN FORMATEADOS
Los campos pct, yoy_pct, acceleration_pct, acceleration_yoy vienen como STRINGS
con el símbolo % incluido (ej: "0.022698%", "5.164212%", "-6.447661%").
COPIA el valor textual tal cual. NUNCA hagas cálculos ni conversiones.
Ejemplos:
  · yoy_pct = "7.39%"     → "creció un **7,39%**"   ✓ CORRECTO
  · yoy_pct = "0.022698%" → "creció un **0,02%**"    ✓ CORRECTO
  · yoy_pct = "0.022698%" y reportas "2,27%" → ✗ INCORRECTO (multiplicaste por 100)
Si yoy_pct dice "0.022698%", es un porcentaje MUY PEQUEÑO. Repórtalo como ~0,02%.
El símbolo % en el dato CONFIRMA que ya es un porcentaje final.

UNIDADES Y VALORACIÓN
- El nombre del cuadro indica la unidad y tipo de valoración. Identifícalos con get_metadata:
  · "miles de millones de pesos encadenados" = volumen real, referencia 2018.
  · "miles de millones de pesos" (sin "encadenados") = valores nominales (precios corrientes).
  · "promedio 2018=100" = índice base 100 en 2018.
- SIEMPRE incluye la unidad al reportar niveles ("value").
  Correcto: "El PIB fue de 52.456 miles de millones de pesos encadenados en 2024."
  Incorrecto: "El PIB fue de 52.456."
- Variaciones porcentuales (pct, yoy_pct) no necesitan unidad (son %).
- En volúmenes encadenados (price "enc"), las variaciones reflejan cambio real
  (ajustado por inflación). En precios corrientes (price "co"), las variaciones
  incluyen tanto actividad real como efecto precios. Menciónalo si es relevante.

REGLAS DE INTERPRETACIÓN DE LA PREGUNTA
- "la cifra", "el valor", "el dato", "cuánto fue" → metric "value".
- "cuánto creció", "cuánto cayó", "variación" (sin más detalle) → metric "yoy_pct".
- "en el margen", "respecto al período anterior", "trimestre/mes anterior",
  "variación en el margen" → metric "pct".
  "pct" es SIEMPRE la variación respecto al período inmediatamente anterior.
- "aceleró", "desaceleró", "aceleración" → metric "acceleration_pct".
  La aceleración es el cambio en "pct" (variación período anterior) entre dos períodos consecutivos.
  SIEMPRE reporta PRIMERO la variación "pct" del período actual y del período anterior,
  y LUEGO la aceleración como la diferencia entre ambas variaciones "pct".
  NUNCA uses yoy_pct para aceleración; la aceleración se mide SIEMPRE sobre "pct".
  Agrega análisis interpretativo (se intensifica / se modera / cambia de signo).
- "cuál creció más", "cuál cayó más", comparaciones → usa rank_series.
- "máximo", "mínimo", "mejor año", "peor año" → usa get_extrema.
- Cambio absoluto → "delta_abs" (vs anterior) o "yoy_delta_abs" (vs año anterior).

FRECUENCIAS Y FORMATOS DE PERÍODO
- "A": YYYY (ej: "2020")
- "T": YYYY-QN (ej: "2020-Q1")
- "M": YYYY-MM (ej: "2020-01")
- Si el usuario no especifica frecuencia, usa la nativa de la serie.
- Si pide el dato "más reciente", usa get_metadata para consultar latest_available.

PERÍODO POR DEFECTO
- Si el usuario NO especifica período ni fecha, SIEMPRE usa el último período disponible.
  Consulta get_metadata para obtener latest_available y usa ese período.
  NUNCA elijas un período arbitrario o antiguo cuando no se especifica fecha.

DATOS NO DISPONIBLES (FALLBACK)
- CASO 1 — El usuario NO mencionó fecha ni período en su pregunta:
  NO digas "no hay datos para [fecha]". Simplemente usa el último período disponible
  (latest_available de get_metadata) y responde directamente con esos datos, sin mencionar
  que algún período futuro no tiene datos.
  Ejemplo correcto: "En enero de 2026, el IMACEC desestacionalizado creció un **0,19%** ..."
  Ejemplo INCORRECTO: "En febrero de 2026, no hay datos disponibles. Sin embargo, en enero..."
- CASO 2 — El usuario pidió EXPLÍCITAMENTE un período y ese período no tiene datos:
    Las referencias temporales relativas también cuentan como período EXPLÍCITO
    (ej: "año pasado", "este año", "trimestre pasado", "mes pasado").
  1. Indica que no hay datos para el período solicitado.
  2. Consulta get_metadata para identificar el último período disponible (latest_available).
  3. Automáticamente consulta y presenta los datos del último período disponible como alternativa.
  Ejemplo correcto: "No hay datos disponibles aún para 2026-Q1. Sin embargo, en el último
  trimestre con datos (2025-Q3), la construcción creció un **5,2%** interanual..."
  NUNCA termines la respuesta solo diciendo que no hay datos sin ofrecer la alternativa.

DESAMBIGUACIÓN
- Si la pregunta es general, prioriza la serie total/agregada sobre componentes.
- Usa list_series si no tienes claro cuál serie elegir.
- Si el usuario pide una actividad específica (ej: "minería") y esa actividad NO existe en
    las series disponibles del cuadro, indícalo explícitamente. NO sustituyas por PIB total
    ni por otra actividad y NO inventes cifras para la actividad solicitada.
- En conversaciones de seguimiento, hereda indicador/serie/frecuencia/métrica del turno anterior.
  Solo reemplaza lo que el nuevo turno cambie explícitamente.

ESTRUCTURA DE SERIES
- La primera serie del cuadro es generalmente el total o agregado (ej: "PIB", "Imacec").
- Las demás son componentes: actividades económicas, componentes del gasto, regiones, etc.
- En series de volumen encadenado, los componentes NO necesariamente suman al total
  (propiedad de la metodología de encadenamiento del Banco Central).
- Si has_activity=1 en la clasificación, las series son actividades económicas.
  Si has_region=1, se refiere a una región específica (ver campo "region").
  Si has_investment=1, las series son componentes del gasto (consumo, inversión, etc.).

CLASIFICACIÓN DEL CUADRO
- El campo "classification" en get_metadata describe el cuadro:
  · indicator: "imacec" (Índice Mensual de Actividad Económica) o "pib" (Producto Interno Bruto).
  · seasonality: "nsa" (no desestacionalizado) o "sa" (desestacionalizado).
    En series desestacionalizadas, pct refleja el cambio marginal limpio de estacionalidad.
  · price: "enc" (volumen encadenado, real) o "co" (precios corrientes, nominal).
  · calc_mode: tipo de cálculo ("original", "prev_period", "yoy", "contribution", "share").
- Si el usuario pregunta por desestacionalizado vs no desestacionalizado, verifica
  qué tipo de cuadro está cargado con get_metadata.

VALORES NULOS EN LOS DATOS
- Los primeros registros de una serie pueden tener campos nulos (pct, yoy_pct, etc.).
  Esto es normal: no existe historia previa suficiente para calcular la métrica.
  yoy_pct es null en los primeros k períodos (k=12 meses, 4 trimestres, 1 año).
- No lo reportes como error. Si el usuario pide una métrica y es null, explica que
  no está disponible para ese período por falta de base comparable.

SERIES ANUALES
- En series anuales, "pct" y "yoy_pct" pueden coincidir; trátalo como una sola interpretación.
- Lo mismo para "delta_abs" / "yoy_delta_abs" y "acceleration_pct" / "acceleration_yoy".

REGLA CARDINAL
- TODA respuesta DEBE contener al menos una cifra numérica obtenida de las herramientas.
- NUNCA respondas solo con metadata o descripciones sin datos concretos.
- NUNCA preguntes "¿quieres que consulte los datos?" ni "¿quieres que lo haga?".
  Si ya identificaste la serie y el período, consulta los datos y preséntalos directamente.
- PRIORIZACIÓN DE MÉTRICAS SEGÚN calc_mode (usar CONTEXTO DE CLASIFICACIÓN):
    · Si calc_mode="original": reporta PRIMERO "value"; luego "yoy_pct"; y "pct" solo cuando
        la pregunta sea de margen/período anterior o aporte claridad.
    · Si calc_mode="yoy": reporta SIEMPRE PRIMERO "yoy_pct".
    · Si calc_mode="prev_period": reporta SIEMPRE PRIMERO "pct".
    · Si no hay calc_mode explícito, usa la clasificación del cuadro en get_metadata y aplica
        la misma regla.
    Después de la métrica principal, puedes complementar con las otras métricas relevantes.

ESTILO DE RESPUESTA
- Español, claro, preciso y detallado.
- ORDEN DEL PRIMER ENUNCIADO (OBLIGATORIO): la primera oración del primer párrafo
    debe comenzar mencionando explícitamente el período analizado (ej: "En el 3er trimestre
    de 2025,..." o "En enero de 2026,..."). No inicies la oración sin anclar primero el período.
- FORMATO NUMÉRICO ESPAÑOL (OBLIGATORIO): usa coma (,) como separador decimal y
    punto (.) como separador de miles. Los datos llegan con punto decimal; conviértelos.
    Ejemplo: value=52975.68 → "52.975,68"; yoy_pct="1.58%" → "**1,58%**".
- FORMATO NEGRITA OBLIGATORIO: SIEMPRE escribe en **negrita** los valores de variaciones
  (pct, yoy_pct, acceleration) y contribuciones. Ejemplo:
  · "creció un **0,02%**" ✓   |   "creció un 0,02%" ✗
  · "la contribución fue de **1,3 pp**" ✓
  · "aceleró de **3,2%** a **4,1%**" ✓
  Los valores absolutos (value, delta_abs) van sin negrita, salvo que sean la métrica principal.
- Primer párrafo: respuesta directa con la cifra o resultado principal.
- CUANDO el usuario pregunta por un año completo y los datos son trimestrales o mensuales,
  SIEMPRE presenta los valores de TODOS los sub-períodos del año (ej: los 4 trimestres o los 12 meses).
  Ejemplo correcto: "La minería creció un 4.69% en 2024-Q1, 2.32% en 2024-Q2, 4.44% en 2024-Q3
  y 7.45% en 2024-Q4."
  NUNCA resumas un solo sub-período como si fuera el valor del año completo.
- CUANDO el usuario pide un rango plurianual completo (ej: "entre 2010 y 2014", "durante el
    primer gobierno de...", "entre 2018 y 2022"), SIEMPRE cubre todo el rango pedido. Si existe
    frecuencia anual para la serie, prefierela para resumir el período completo. Si usas frecuencia
    trimestral o mensual, debes cubrir todos los años del rango, no solo algunos años seleccionados.
- Si el año o semestre solicitado está INCOMPLETO porque aún faltan sub-períodos,
    NUNCA lo describas como año/semestre cerrado. Debes decir explícitamente que es un resultado
    parcial o acumulado hasta el último sub-período disponible (ej: "en lo disponible de 2025",
    "hasta el 3er trimestre de 2025").
- NUNCA uses frases de cierre como "En 2025, ..." o "Durante el primer semestre de 2025, ..."
    si no están todos los trimestres/meses requeridos para ese año o semestre.
- Segundo párrafo: contexto y análisis. Explica qué significan los datos,
  compara con períodos anteriores si es relevante, y describe la tendencia general.
  NO especules sobre causas ni des opiniones sobre qué factores explican los valores
  (ej: NO decir "asociado a la pandemia", "por la crisis", "debido al dinamismo del sector").
    Tampoco introduzcas contexto histórico o causal no observado directamente en los datos
    (ej: "tras la crisis de 2009", "por el cambio de gobierno", "debido a condiciones externas").
  Limítate a describir los datos: magnitudes, comparaciones y tendencias numéricas.
- NO agregues un bloque final de trazabilidad del tipo "Los datos se obtuvieron de las siguientes series".
- NO incluyas listados largos de series_id al final de la respuesta.
- Si necesitas citar series_id, hazlo de forma breve y solo para las series estrictamente
    necesarias para sostener el resultado principal, integrado dentro del texto.
- En rankings, menciona solo los series_id de los elementos efectivamente reportados.
- Si hay datos adicionales relevantes (valores absolutos, variaciones de períodos cercanos,
  posición relativa en el ranking completo), inclúyelos para enriquecer la respuesta.
- SIEMPRE incluye las cifras numéricas exactas obtenidas de las herramientas.
  NUNCA respondas solo con descripciones cualitativas (ej: "tuvo su mayor expansión").
  SIEMPRE acompaña con el número concreto (ej: "tuvo su mayor expansión, con un crecimiento de 12.3%").
  Si mencionas un máximo, mínimo, crecimiento o caída, incluye el valor numérico y el período.
- NO uses nombres técnicos de campos en la respuesta al usuario.
    Nunca escribas literales como "value", "pct", "yoy_pct", "delta_abs", "yoy_delta_abs",
    "acceleration_pct" o "acceleration_yoy".
    En su lugar, usa lenguaje natural: "nivel", "variación mensual/trimestral", "variación interanual",
    "cambio absoluto" o "aceleración".
- No menciones nombres de herramientas al usuario.
- IMPORTANTE: en rankings de crecimiento/caída, separa claramente los positivos de los negativos.
  Si el usuario pide "las que más crecieron" y solo algunas tienen variación positiva:
  → Primero nombra las que efectivamente crecieron (variación > 0).
  → Luego, si quedan posiciones por cubrir, indica cuáles fueron las que menos cayeron
    (NO decir que "crecieron" si su variación es negativa).
  Ejemplo correcto: "Solo Minería creció en 2020 (0.77%). Las actividades que menos
  se contrajeron fueron Industria (-2.19%) y Comercio (-2.74%)."
  Aplica la misma lógica inversa para "las que más cayeron" con valores positivos.

EJEMPLOS
- "cuanto crecio el pib en 2024" → get_series_data con metric yoy_pct.
- "cual fue la que mas crecio en 2024" → rank_series con metric yoy_pct.
- "el imacec aceleró en enero 2025?" → get_series_data para 2025-01 y 2024-12.
- "y cual fue el del 2023" → heredar serie/métrica, cambiar período a 2023.

No agregues una línea de fuente o citación al final de tu respuesta; \
la fuente se añade automáticamente por el sistema.
"""


# ---------------------------------------------------------------------------
# Tool definitions (migrado de chatbot.py)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_series",
            "description": "Lista las series disponibles en el cuadro con su ID, título corto y clasificación.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_series_data",
            "description": (
                "Obtiene registros de datos de una serie para un período exacto o un rango. "
                "El series_id debe salir EXACTAMENTE de list_series. "
                "NO derives el series_id cambiando sufijos .M/.T/.A ni usando el cuadro_id. "
                "La frecuencia pedida se selecciona con el parámetro frequency sobre el MISMO series_id, "
                "siempre que esa frecuencia aparezca en available_frequencies/list_series. "
                "Cada registro contiene: period, value (número), pct, yoy_pct, acceleration_pct, "
                "acceleration_yoy (strings con %, ej: '0.022698%', '5.164%'), delta_abs, yoy_delta_abs (números). "
                "Los campos de porcentaje YA son strings formateados. Cópialos tal cual, NO multipliques por 100."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "series_id": {"type": "string", "description": "ID de la serie (ej: F032.IMC.IND.Z.Z.EP18.Z.Z.0.M)"},
                    "frequency": {"type": "string", "enum": ["M", "T", "A"], "description": "Frecuencia de datos"},
                    "period": {"type": "string", "description": "Período exacto (ej: 2020, 2020-Q1, 2020-01). Omitir si se usa rango."},
                    "period_start": {"type": "string", "description": "Inicio del rango inclusivo (opcional)"},
                    "period_end": {"type": "string", "description": "Fin del rango inclusivo (opcional)"},
                },
                "required": ["series_id", "frequency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_series",
            "description": (
                "Rankea todas las series del cuadro por una métrica en un período dado. "
                "Devuelve la lista ordenada con series_id, short_title y el valor de la métrica."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "frequency": {"type": "string", "enum": ["M", "T", "A"]},
                    "period": {"type": "string", "description": "Período a rankear (ej: 2020, 2024-Q3, 2025-01)"},
                    "metric": {"type": "string", "enum": ["value", "pct", "yoy_pct"], "description": "Métrica de ordenamiento"},
                    "order": {"type": "string", "enum": ["desc", "asc"], "description": "Orden: desc (mayor primero) o asc (menor primero). Default: desc."},
                },
                "required": ["frequency", "period", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_extrema",
            "description": (
                "Obtiene máximos y mínimos históricos por métrica. "
                "Si se pasa series_id, devuelve extrema de esa serie; si no, de todo el cuadro."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "frequency": {"type": "string", "enum": ["M", "T", "A"]},
                    "metric": {"type": "string", "enum": ["value", "pct", "yoy_pct"]},
                    "series_id": {"type": "string", "description": "ID de serie (omitir para extrema del cuadro)"},
                },
                "required": ["frequency", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metadata",
            "description": "Obtiene metadatos del cuadro: nombre, clasificación, unidad de medida (en cuadro_name), frecuencia del archivo, último período disponible, URL fuente y fecha de generación.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution backend (migrado de chatbot.py)
# ---------------------------------------------------------------------------

def _find_series(series_list: list, series_id: str):
    """Busca una serie por ID en la lista del payload."""
    for s in series_list:
        if s["series_id"] == series_id:
            return s
    return None


def _ordered_available_frequencies(series: Dict[str, Any]) -> List[str]:
    data = series.get("data") or {}
    order = {"M": 0, "T": 1, "A": 2}
    return sorted(
        [str(freq).upper() for freq in data.keys() if str(freq).upper() in order],
        key=lambda freq: order.get(freq, 99),
    )


def _compact_series_catalog(series_list: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for series in series_list[:limit]:
        out.append(
            {
                "series_id": series.get("series_id", ""),
                "short_title": series.get("short_title", ""),
                "series_freq": series.get("series_freq"),
                "available_frequencies": _ordered_available_frequencies(series),
            }
        )
    return out


_PCT_FIELDS = ("pct", "yoy_pct", "acceleration_pct", "acceleration_yoy")

_UNITS_NOTE = (
    "pct, yoy_pct, acceleration_pct, acceleration_yoy YA son strings con %. "
    "Cópialos tal cual. NO multipliques por 100."
)


def _add_display_fields(record: dict) -> dict:
    """Reemplaza los campos numéricos de porcentaje por strings formateados.

    Los campos pct, yoy_pct, acceleration_pct, acceleration_yoy YA están
    en puntos porcentuales (la fórmula de ingesta ya aplicó ×100).
    Se reemplazan IN-PLACE por strings "X%" para que el LLM los copie
    textualmente y no pueda multiplicar por 100.
    """
    out = dict(record)
    for field in _PCT_FIELDS:
        if field in out and out[field] is not None:
            out[field] = f"{out[field]}%"
    # Eliminar _display si existiera de versiones anteriores
    out.pop("_display", None)
    return out


def handle_tool_call(name: str, args: dict, payload: dict) -> str:
    """Ejecuta una herramienta y devuelve el resultado como string JSON."""
    series_list = payload.get("series", [])

    if name == "list_series":
        result = [
            {
                "series_id": s["series_id"],
                "short_title": s["short_title"],
                "long_title": s.get("long_title", ""),
                "classification": s.get("classification_series", {}),
                "series_freq": s.get("series_freq"),
                "available_frequencies": _ordered_available_frequencies(s),
            }
            for s in series_list
        ]
        return json.dumps(result, ensure_ascii=False)

    if name == "get_metadata":
        return json.dumps(
            {
                "cuadro_name": payload.get("cuadro_name"),
                "cuadro_id": payload.get("cuadro_id"),
                "classification": payload.get("classification", {}),
                "frequency": payload.get("frequency"),
                "latest_available": payload.get("latest_available", {}),
                "series_count": len(series_list),
                "source_url": payload.get("source_url", ""),
                "generated_at_utc": payload.get("dataset_meta", {}).get("generated_at_utc", ""),
            },
            ensure_ascii=False,
        )

    if name == "get_series_data":
        series = _find_series(series_list, args.get("series_id", ""))
        if not series:
            return json.dumps(
                {
                    "error": f"Serie '{args.get('series_id')}' no encontrada en este cuadro",
                    "hint": (
                        "Usa list_series y copia exactamente uno de sus series_id. "
                        "No derives IDs desde cuadro_id ni cambies sufijos .M/.T/.A."
                    ),
                    "available_series": _compact_series_catalog(series_list),
                },
                ensure_ascii=False,
            )
        block = series["data"].get(args["frequency"])
        if not block:
            avail = _ordered_available_frequencies(series)
            return json.dumps(
                {
                    "error": f"Frecuencia '{args['frequency']}' no disponible para la serie '{series['series_id']}'",
                    "available_frequencies": avail,
                    "hint": (
                        "Mantén el mismo series_id y cambia solo el parámetro frequency "
                        "a una frecuencia disponible."
                    ),
                },
                ensure_ascii=False,
            )
        records = block.get("records", [])
        if args.get("period"):
            records = [r for r in records if r["period"] == args["period"]]
        elif args.get("period_start") or args.get("period_end"):
            start = args.get("period_start", "")
            end = args.get("period_end", "9999")
            records = [r for r in records if start <= r["period"] <= end]
        if not records:
            return json.dumps({"error": "Sin datos para los parámetros dados", "params": args})
        return json.dumps(
            {
                "series_id": series["series_id"],
                "short_title": series["short_title"],
                "frequency": args["frequency"],
                "records": [_add_display_fields(r) for r in records],
                "_units_note": _UNITS_NOTE,
            },
            ensure_ascii=False,
        )

    if name == "rank_series":
        freq = args["frequency"]
        period = args["period"]
        metric = args["metric"]
        order = args.get("order", "desc")
        entries = (payload.get("cuadro_summaries", {})
                   .get("rankings", {})
                   .get(freq, {})
                   .get(period))
        if not entries:
            return json.dumps({"error": f"Sin datos para frecuencia '{freq}' período '{period}'"})
        ranked = sorted(
            [e for e in entries if e.get(metric) is not None],
            key=lambda x: x[metric],
            reverse=(order == "desc"),
        )
        titles = {s["series_id"]: s["short_title"] for s in series_list}
        ranked = [_add_display_fields({**e, "short_title": titles.get(e["series_id"], "")}) for e in ranked]
        return json.dumps(
            {"frequency": freq, "period": period, "metric": metric,
             "order": order, "ranking": ranked,
             "_units_note": _UNITS_NOTE},
            ensure_ascii=False,
        )

    if name == "get_extrema":
        freq = args["frequency"]
        metric = args["metric"]
        sid = args.get("series_id")
        if sid:
            series = _find_series(series_list, sid)
            if not series:
                return json.dumps({"error": f"Serie '{sid}' no encontrada"})
            ext = series.get("extrema", {}).get(freq, {})
            result = {
                "series_id": sid,
                "short_title": series["short_title"],
                f"{metric}_max": ext.get(f"{metric}_max"),
                f"{metric}_min": ext.get(f"{metric}_min"),
            }
        else:
            fex = payload.get("cuadro_summaries", {}).get("cuadro_extrema", {}).get(freq, {})
            result = {
                "scope": "cuadro",
                f"{metric}_max": fex.get(f"{metric}_max"),
                f"{metric}_min": fex.get(f"{metric}_min"),
            }
        return json.dumps(result, ensure_ascii=False)

    return json.dumps({"error": f"Herramienta desconocida: {name}"})


# ---------------------------------------------------------------------------
# Construcción de mensajes iniciales con fecha interpolada
# ---------------------------------------------------------------------------

def _build_initial_messages() -> list:
    """Construye el mensaje de sistema con la fecha actual interpolada."""
    today = date.today()
    q = (today.month - 1) // 3 + 1
    current_quarter = f"{today.year}-Q{q}"
    if q == 1:
        prev_quarter = f"{today.year - 1}-Q4"
    else:
        prev_quarter = f"{today.year}-Q{q - 1}"
    if today.month == 1:
        prev_month = f"{today.year - 1}-12"
    else:
        prev_month = f"{today.year}-{today.month - 1:02d}"

    prompt = SYSTEM_PROMPT.format(
        today=today.isoformat(),
        this_year=today.year,
        last_year=today.year - 1,
        two_years_ago=today.year - 2,
        current_quarter=current_quarter,
        prev_quarter=prev_quarter,
        prev_month=prev_month,
    )
    return [{"role": "system", "content": prompt}]


# ---------------------------------------------------------------------------
# Source footer
# ---------------------------------------------------------------------------

def _build_source_footer(observations: Dict[str, Any]) -> Optional[str]:
    """Construye un footer con el link a la fuente BDE desde el payload."""
    source_url = str(observations.get("source_url") or "").strip()
    if not source_url:
        return None
    return (
        f"\n\n**Fuente:** 🔗 [Base de Datos Estadísticos (BDE)]({source_url}) "
        "del Banco Central de Chile."
    )


def _build_metric_priority_instruction(calc_mode: str) -> Optional[str]:
    """Construye una instrucción estricta de priorización por calc_mode.

    Esta instrucción se inyecta como mensaje de sistema adicional para
    endurecer el comportamiento del primer párrafo.
    """
    mode = str(calc_mode or "").strip().lower()
    if mode == "yoy":
        return (
            "REGLA ESTRICTA DE REDACCION: en el PRIMER PARRAFO (primera oracion) "
            "comienza mencionando el PERIODO analizado (ej: 'En el 3er trimestre de 2025, ...') "
            "y reporta PRIMERO el valor de 'yoy_pct'. "
            "No comiences con 'value' ni con 'pct'."
        )
    if mode == "prev_period":
        return (
            "REGLA ESTRICTA DE REDACCION: en el PRIMER PARRAFO (primera oracion) "
            "comienza mencionando el PERIODO analizado (ej: 'En el 3er trimestre de 2025, ...') "
            "y reporta PRIMERO el valor de 'pct'. "
            "No comiences con 'value' ni con 'yoy_pct'."
        )
    if mode == "original":
        return (
            "REGLA ESTRICTA DE REDACCION: en el PRIMER PARRAFO (primera oracion) "
            "comienza mencionando el PERIODO analizado (ej: 'En el 3er trimestre de 2025, ...') "
            "y reporta PRIMERO el valor de 'value' (con su unidad). "
            "Luego menciona 'yoy_pct', y 'pct' solo si aporta claridad. "
            "No comiences con 'yoy_pct' ni con 'pct'."
        )
    return None


def _normalize_freq_code(freq: Any) -> str:
    return _FREQ_CODE_MAP.get(str(freq or "").strip().lower(), "")


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _is_req_form_latest(entities_ctx: Dict[str, Any]) -> bool:
    req_form = str(entities_ctx.get("req_form_cls") or "").strip().lower()
    return req_form == "latest"


def _build_missing_activity_instruction(
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[str]:
    activity_cls = str(
        entities_ctx.get("activity_cls_resolved")
        or entities_ctx.get("activity_cls")
        or ""
    ).strip().lower()
    if activity_cls != "specific":
        return None

    requested_activity = _normalize_token(entities_ctx.get("activity_ent"))
    if requested_activity in {"", "none", "null"}:
        return None

    # IMACEC/PIB son indicadores agregados, no actividades económicas.
    if requested_activity in {"imacec", "pib"}:
        return None

    available_activities: List[str] = []
    for series in observations.get("series", []) or []:
        cls = series.get("classification_series", {})
        if not isinstance(cls, dict):
            continue
        activity = _normalize_token(cls.get("activity"))
        if activity:
            available_activities.append(activity)

    if not available_activities:
        return None

    available_set = set(available_activities)
    if requested_activity in available_set:
        return None

    options = ", ".join(sorted(available_set)[:8])
    return (
        "REGLA ESTRICTA DE ACTIVIDAD FALTANTE: la actividad solicitada no existe en este cuadro. "
        f"Actividad pedida: '{requested_activity}'. Actividades disponibles: {options}. "
        "Debes responder explícitamente que no hay datos para la actividad solicitada en este cuadro. "
        "NO reemplaces por la serie agregada (PIB total) ni por otra actividad, y NO inventes valores."
    )


def _resolve_requested_frequency(
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> str:
    for key in ("frequency_ent", "frequency"):
        freq_code = _normalize_freq_code(entities_ctx.get(key))
        if freq_code:
            return freq_code

    return _normalize_freq_code(
        (observations.get("classification") or {}).get("frequency")
        or observations.get("frequency")
    )


def _natural_period_label(period_str: str, freq_code: str) -> str:
    label = format_period_labels(period_str, "m" if freq_code == "M" else "q" if freq_code == "T" else "a")[0]
    if freq_code == "T":
        return re.sub(r" trimestre (\d{4})$", r" trimestre de \1", label)
    if freq_code == "M":
        return re.sub(r" ([12]\d{3})$", r" de \1", label)
    return label


def _extract_requested_year(entities_ctx: Dict[str, Any], question: str) -> Optional[str]:
    period_values = entities_ctx.get("period_ent")
    if isinstance(period_values, list):
        for value in period_values:
            text = str(value or "").strip()
            if re.fullmatch(r"(19|20)\d{2}", text):
                return text

    match = re.search(r"\b((?:19|20)\d{2})\b", str(question or ""))
    if match:
        return match.group(1)
    return None


def _extract_requested_semester(question: str) -> Tuple[Optional[int], Optional[str]]:
    text = str(question or "")
    patterns = [
        r"\b(primer|1er|1ro)\s+semestre(?:\s+de)?\s+((?:19|20)\d{2})\b",
        r"\b(segundo|2do)\s+semestre(?:\s+de)?\s+((?:19|20)\d{2})\b",
    ]
    for idx, pattern in enumerate(patterns, start=1):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return idx, match.group(2)
    return None, None


def _build_incomplete_period_instruction(
    question: str,
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[str]:
    freq_code = _resolve_requested_frequency(entities_ctx, observations)
    if freq_code not in {"M", "T"}:
        return None

    latest_available = observations.get("latest_available") or {}
    latest_period = str(latest_available.get(freq_code) or "").strip()
    if not latest_period:
        return None

    requested_semester, semester_year = _extract_requested_semester(question)
    if requested_semester and semester_year:
        expected_final = (
            f"{semester_year}-06" if freq_code == "M" and requested_semester == 1 else
            f"{semester_year}-12" if freq_code == "M" else
            f"{semester_year}-Q2" if requested_semester == 1 else
            f"{semester_year}-Q4"
        )
        same_year = latest_period.startswith(f"{semester_year}-")
        if same_year and latest_period < expected_final:
            latest_label = _natural_period_label(latest_period, freq_code)
            semester_label = "primer semestre" if requested_semester == 1 else "segundo semestre"
            return (
                "REGLA ESTRICTA DE COBERTURA TEMPORAL: el semestre solicitado NO está completo. "
                f"Solo hay datos hasta {latest_label}. Debes describirlo como resultado parcial del "
                f"{semester_label} de {semester_year}, nunca como semestre cerrado. "
                f"No escribas frases como 'Durante el {semester_label} de {semester_year}, ...'. "
                f"Usa formulaciones como 'En lo disponible del {semester_label} de {semester_year}, ...' "
                f"o 'Hasta {latest_label}, ...'."
            )

    requested_year = _extract_requested_year(entities_ctx, question)
    if requested_year:
        expected_final = f"{requested_year}-12" if freq_code == "M" else f"{requested_year}-Q4"
        if latest_period.startswith(f"{requested_year}-") and latest_period != expected_final:
            latest_label = _natural_period_label(latest_period, freq_code)
            return (
                "REGLA ESTRICTA DE COBERTURA TEMPORAL: el año solicitado NO está cerrado. "
                f"Solo hay datos hasta {latest_label}. Debes describirlo como resultado parcial "
                f"o acumulado de {requested_year}, nunca como el resultado anual cerrado. "
                f"No escribas frases como 'En {requested_year}, ...'. Usa formulaciones como "
                f"'En lo disponible de {requested_year}, ...' o 'Hasta {latest_label}, ...'."
            )

    return None


def _build_relative_period_fallback_instruction(
    question: str,
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[str]:
    if _is_req_form_latest(entities_ctx):
        return None

    text = str(question or "").lower()
    today = date.today()

    relative_expr = None
    freq_code = ""
    target_period = ""

    if "año pasado" in text:
        relative_expr = "año pasado"
        freq_code = "A"
        target_period = str(today.year - 1)
    elif "este año" in text:
        relative_expr = "este año"
        freq_code = "A"
        target_period = str(today.year)
    elif "hace dos años" in text:
        relative_expr = "hace dos años"
        freq_code = "A"
        target_period = str(today.year - 2)
    elif "trimestre pasado" in text:
        relative_expr = "trimestre pasado"
        q = (today.month - 1) // 3 + 1
        target_period = f"{today.year - 1}-Q4" if q == 1 else f"{today.year}-Q{q - 1}"
        freq_code = "T"
    elif "este trimestre" in text:
        relative_expr = "este trimestre"
        q = (today.month - 1) // 3 + 1
        target_period = f"{today.year}-Q{q}"
        freq_code = "T"
    elif "mes pasado" in text:
        relative_expr = "mes pasado"
        target_period = f"{today.year - 1}-12" if today.month == 1 else f"{today.year}-{today.month - 1:02d}"
        freq_code = "M"
    elif "este mes" in text:
        relative_expr = "este mes"
        target_period = f"{today.year}-{today.month:02d}"
        freq_code = "M"

    if not (relative_expr and freq_code and target_period):
        return None

    # Si hay frecuencia explícita en entidades, respetarla para evitar cruces inconsistentes.
    requested_freq = _resolve_requested_frequency(entities_ctx, observations)
    if requested_freq and requested_freq != freq_code:
        freq_code = requested_freq

    latest_available = observations.get("latest_available") or {}
    latest_period = str(latest_available.get(freq_code) or "").strip()
    if not latest_period:
        return None

    if target_period <= latest_period:
        return None

    latest_label = _natural_period_label(latest_period, freq_code)
    target_label = target_period if freq_code == "A" else _natural_period_label(target_period, freq_code)
    return (
        "REGLA ESTRICTA DE FALLBACK PARA FECHA RELATIVA: la pregunta pide un período explícito "
        f"('{relative_expr}' = {target_label}) sin datos disponibles. Debes decir explícitamente "
        f"que no hay datos para {target_label} y luego entregar el dato del período más cercano "
        f"disponible ({latest_label}). NUNCA omitas el aviso de falta de datos cuando la referencia "
        "temporal relativa apunta a un período inexistente."
    )


def _question_has_explicit_period(question: str, entities_ctx: Dict[str, Any]) -> bool:
    period_values = entities_ctx.get("period_ent")
    if isinstance(period_values, list):
        for value in period_values:
            text = str(value or "").strip()
            if re.fullmatch(r"(?:19|20)\d{2}", text):
                return True
            if re.fullmatch(r"(?:19|20)\d{2}-Q[1-4]", text, re.IGNORECASE):
                return True
            if re.fullmatch(r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])", text):
                return True

    text = str(question or "").lower()
    if re.search(r"\b(?:19|20)\d{2}\b", text):
        return True
    if re.search(r"\b(?:19|20)\d{2}-q[1-4]\b", text):
        return True
    if re.search(r"\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])\b", text):
        return True
    if re.search(r"\b(primer|1er|1ro|segundo|2do)\s+semestre\b", text):
        return True

    relative_tokens = (
        "año pasado",
        "este año",
        "hace dos años",
        "trimestre pasado",
        "este trimestre",
        "mes pasado",
        "este mes",
        "hoy",
        "ayer",
    )
    return any(token in text for token in relative_tokens)


def _build_no_explicit_period_latest_instruction(
    question: str,
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[str]:
    if not _is_req_form_latest(entities_ctx) and _question_has_explicit_period(question, entities_ctx):
        return None

    freq_code = _resolve_requested_frequency(entities_ctx, observations)
    latest_available = observations.get("latest_available") or {}
    latest_period = str(latest_available.get(freq_code) or "").strip() if freq_code else ""
    if not latest_period:
        return None

    latest_label = latest_period if freq_code == "A" else _natural_period_label(latest_period, freq_code)
    return (
        "REGLA ESTRICTA DE PERIODO POR DEFECTO: "
        "si req_form_cls='latest', o si la pregunta no especifica fecha, "
        f"Debes responder usando directamente el ultimo periodo disponible ({latest_label}). "
        "NO menciones falta de datos para meses/trimestres/años no solicitados y NO infieras "
        "automaticamente el mes/trimestre actual como periodo pedido."
    )


# ---------------------------------------------------------------------------
# CSV export & download markers
# ---------------------------------------------------------------------------

# Columns to exclude from the exported CSV
_CSV_EXCLUDE_COLS = {"delta_abs", "yoy_delta_abs", "acceleration_pct", "acceleration_yoy"}


def _export_series_csv(
    series_id: str,
    records: List[Dict[str, Any]],
    short_title: str = "",
    cuadro_name: str = "",
) -> Optional[str]:
    """Write *records* to a temporary CSV and return its path.

    Adds ASCII-safe metadata header and strips internal diagnostic columns.
    """
    if not records:
        return None
    try:
        fieldnames = [k for k in records[0] if k not in _CSV_EXCLUDE_COLS]
        buf = io.StringIO()
        buf.write(f"# Nombre: {cuadro_name}\n")
        buf.write(f"# Serie: {short_title}\n")
        buf.write(f"# Serie ID: {series_id}\n")
        buf.write("# yoy_pct: variación porcentual interanual\n")
        buf.write("# pct: variación porcentual respecto al periodo anterior\n")
        buf.write("#\n")
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="serie_",
            delete=False, encoding="utf-8-sig", newline="",
        )
        tmp.write(buf.getvalue())
        tmp.close()
        return tmp.name
    except Exception:
        logger.exception("[DATA_RESPONSE] Error exportando CSV para %s", series_id)
        return None


def _build_csv_markers(fetched_series: List[Dict[str, Any]], cuadro_name: str = "") -> str:
    """Build a single ``##CSV_DOWNLOAD_START/END`` marker using the first fetched series."""
    for entry in fetched_series:
        series_id = entry.get("series_id", "")
        short_title = entry.get("short_title", "")
        records = entry.get("records", [])
        path = _export_series_csv(series_id, records, short_title=short_title, cuadro_name=cuadro_name)
        if not path:
            continue
        filename = f"serie_{series_id}.csv" if series_id else os.path.basename(path)
        return (
            f"##CSV_DOWNLOAD_START\n"
            f"path={path}\n"
            f"filename={filename}\n"
            f"title={short_title}\n"
            f"label=Descargar CSV\n"
            f"mimetype=text/csv\n"
            f"##CSV_DOWNLOAD_END"
        )
    return ""


def _export_cuadro_csv(observations: Dict[str, Any]) -> Optional[str]:
    """Export the full cuadro payload to a CSV so the UI always has a fallback download."""
    series_list = observations.get("series") or []
    rows: List[Dict[str, Any]] = []
    dynamic_fields: List[str] = []
    dynamic_seen = set()

    for series in series_list:
        series_id = series.get("series_id", "")
        short_title = series.get("short_title", "")
        data_blocks = series.get("data") or {}
        for frequency, block in data_blocks.items():
            records = (block or {}).get("records") or []
            for record in records:
                row = {
                    "series_id": series_id,
                    "short_title": short_title,
                    "frequency": frequency,
                }
                for key, value in record.items():
                    if key in _CSV_EXCLUDE_COLS:
                        continue
                    row[key] = value
                    if key not in dynamic_seen:
                        dynamic_seen.add(key)
                        dynamic_fields.append(key)
                rows.append(row)

    if not rows:
        return None

    try:
        cuadro_name = str(observations.get("cuadro_name") or "")
        cuadro_id = str(observations.get("cuadro_id") or "")
        fieldnames = ["series_id", "short_title", "frequency", *dynamic_fields]
        buf = io.StringIO()
        buf.write(f"# Nombre: {cuadro_name}\n")
        buf.write(f"# Cuadro ID: {cuadro_id}\n")
        buf.write("# Export: cuadro completo\n")
        buf.write("#\n")
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="cuadro_",
            delete=False, encoding="utf-8-sig", newline="",
        )
        tmp.write(buf.getvalue())
        tmp.close()
        return tmp.name
    except Exception:
        logger.exception("[DATA_RESPONSE] Error exportando CSV fallback del cuadro")
        return None


def _build_fallback_csv_marker(observations: Dict[str, Any]) -> str:
    """Build a fallback marker so the UI can always render one download button."""
    path = _export_cuadro_csv(observations)
    if not path:
        return ""

    cuadro_id = str(observations.get("cuadro_id") or "cuadro")
    cuadro_name = str(observations.get("cuadro_name") or "")
    safe_cuadro_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cuadro_id).strip("_") or "cuadro"
    filename = f"cuadro_{safe_cuadro_id}.csv"
    return (
        "##CSV_DOWNLOAD_START\n"
        f"path={path}\n"
        f"filename={filename}\n"
        f"title={cuadro_name}\n"
        "label=Descargar CSV\n"
        "mimetype=text/csv\n"
        "##CSV_DOWNLOAD_END"
    )


# ---------------------------------------------------------------------------
# Streaming de la respuesta LLM con function calling
# ---------------------------------------------------------------------------

def stream_data_response(
    payload: Dict[str, Any],
) -> Iterable[str]:
    """Genera la respuesta en streaming usando OpenAI function calling.

    El LLM usa herramientas (tools) para consultar los datos del payload
    data_store en lugar de recibir un contexto pre-formateado.

    Args:
        payload: Diccionario con las claves:
            - question (str): pregunta original del usuario.
            - observations (dict): payload completo del data_store JSON
              (contiene series, clasificación, metadata, etc.).

    Yields:
        Fragmentos de texto de la respuesta del LLM.
    """
    if _OpenAI is None:
        logger.error("[DATA_RESPONSE] openai no disponible")
        yield "El servicio de generación de respuestas no está disponible."
        return

    question = payload.get("question", "")
    observations = payload.get("observations") or {}
    entities_ctx = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    calc_mode_ctx = str(entities_ctx.get("calc_mode_cls") or "").strip().lower()
    if not calc_mode_ctx:
        calc_mode_ctx = str((observations.get("classification") or {}).get("calc_mode") or "").strip().lower()

    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    temperature = float(os.getenv("DATA_RESPONSE_TEMPERATURE", "0.35"))
    max_tool_loops = int(os.getenv("MAX_TOOL_LOOPS", "16"))

    try:
        client = _OpenAI()
    except Exception:
        logger.exception("[DATA_RESPONSE] Error inicializando OpenAI client")
        yield "Error al conectar con el modelo de lenguaje."
        return

    messages = _build_initial_messages()
    context_payload = {
        "calc_mode": calc_mode_ctx or None,
        "entities": entities_ctx,
    }
    messages.append(
        {
            "role": "system",
            "content": (
                "CONTEXTO DE CLASIFICACION (usar para priorizar metricas): "
                + json.dumps(context_payload, ensure_ascii=False)
            ),
        }
    )
    strict_priority_instruction = _build_metric_priority_instruction(calc_mode_ctx)
    if strict_priority_instruction:
        messages.append({"role": "system", "content": strict_priority_instruction})
    incomplete_period_instruction = _build_incomplete_period_instruction(
        question=question,
        entities_ctx=entities_ctx,
        observations=observations,
    )
    if incomplete_period_instruction:
        messages.append({"role": "system", "content": incomplete_period_instruction})
    relative_period_fallback_instruction = _build_relative_period_fallback_instruction(
        question=question,
        entities_ctx=entities_ctx,
        observations=observations,
    )
    if relative_period_fallback_instruction:
        messages.append({"role": "system", "content": relative_period_fallback_instruction})
    no_explicit_period_instruction = _build_no_explicit_period_latest_instruction(
        question=question,
        entities_ctx=entities_ctx,
        observations=observations,
    )
    if no_explicit_period_instruction:
        messages.append({"role": "system", "content": no_explicit_period_instruction})
    missing_activity_instruction = _build_missing_activity_instruction(
        entities_ctx=entities_ctx,
        observations=observations,
    )
    if missing_activity_instruction:
        messages.append({"role": "system", "content": missing_activity_instruction})
    messages.append({"role": "user", "content": question})

    fetched_series: List[Dict[str, Any]] = []  # track get_series_data results

    try:
        for _ in range(max_tool_loops):
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                temperature=temperature,
                stream=True,
            )

            # Acumular respuesta streameada
            tool_calls_by_idx: Dict[int, Dict[str, str]] = {}
            content_chunks: List[str] = []

            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                delta = choice.delta

                # Contenido de texto → yield inmediato
                if delta and delta.content:
                    content_chunks.append(delta.content)
                    yield delta.content

                # Acumular tool call deltas
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_by_idx:
                            tool_calls_by_idx[idx] = {
                                "id": "", "name": "", "arguments": "",
                            }
                        if tc_delta.id:
                            tool_calls_by_idx[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_by_idx[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_by_idx[idx]["arguments"] += tc_delta.function.arguments

            if tool_calls_by_idx:
                # Hay tool calls: ejecutar y continuar el loop
                messages.append({
                    "role": "assistant",
                    "content": "".join(content_chunks) or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for _, tc in sorted(tool_calls_by_idx.items())
                    ],
                })
                for _, tc in sorted(tool_calls_by_idx.items()):
                    fn_args = json.loads(tc["arguments"])
                    result = handle_tool_call(tc["name"], fn_args, observations)
                    logger.debug("[DATA_RESPONSE] tool=%s args=%s result_len=%d",
                                 tc["name"], tc["arguments"][:120], len(result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                    # Track fetched series data for CSV export
                    if tc["name"] == "get_series_data":
                        try:
                            parsed = json.loads(result)
                            if "records" in parsed:
                                fetched_series.append(parsed)
                        except Exception:
                            pass
            else:
                # Sin tool calls → respuesta final ya fue yielded via content_chunks
                source_footer = _build_source_footer(observations)
                if source_footer:
                    yield source_footer
                csv_block = ""
                if fetched_series:
                    csv_block = _build_csv_markers(
                        fetched_series,
                        cuadro_name=str(observations.get("cuadro_name") or ""),
                    )
                if not csv_block:
                    csv_block = _build_fallback_csv_marker(observations)
                if csv_block:
                    yield "\n" + csv_block
                return
        else:
            # Se agotó el loop de herramientas
            yield "(Se alcanzó el límite de consultas internas. Intenta reformular la pregunta.)"
    except Exception:
        logger.exception("[DATA_RESPONSE] Error durante streaming con function calling")
        yield "Ocurrió un error al generar la respuesta."
