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
import time
import unicodedata
from dataclasses import asdict
from datetime import date
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
REGLA CRÍTICA: get_metadata NO contiene valores numéricos de series. Para reportar
cualquier cifra (variación, nivel, contribución), DEBES llamar a get_series_data o rank_series.
Si en tu flujo solo llamaste get_metadata, NO tienes datos numéricos y debes llamar
get_series_data antes de responder.

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
- Para preguntas generales del cuadro (definición, alcance, lectura global, resumen),
    puedes apoyarte en `cuadro_summaries` del payload como contexto descriptivo.
    Aun así, la respuesta final debe incluir cifras concretas obtenidas con herramientas.
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
7. VERIFICACIÓN OBLIGATORIA: antes de redactar tu respuesta, confirma que llamaste
   a get_series_data o rank_series para obtener las cifras. Si solo llamaste
   get_metadata o list_series, DETENTE y llama get_series_data para el período y
   serie correspondientes. NUNCA respondas con cifras sin haberlas obtenido
   de get_series_data o rank_series.

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
NUNCA multipliques por 100. El símbolo % en el dato CONFIRMA que ya es un porcentaje final.
SIEMPRE redondea a UN (1) decimal TODAS las cifras porcentuales y contribuciones:
variaciones (pct, yoy_pct), aceleraciones y contribuciones en puntos porcentuales (pp).
Ejemplos:
  · yoy_pct = "5.164%"    → reporta "**5,2%**"    ✓ CORRECTO
  · yoy_pct = "1.58%"     → reporta "**1,6%**"    ✓ CORRECTO
  · yoy_pct = "-0.34%"    → reporta "**-0,3%**"   ✓ CORRECTO
  · yoy_pct = "0.022698%" → reporta "**0,0%**"    ✓ CORRECTO
  · contribución = 0.54    → reporta "**0,5 pp**"  ✓ CORRECTO
  · contribución = -0.67   → reporta "**-0,7 pp**" ✓ CORRECTO
  · contribución = 0.54    → reporta "0,54 pp"     ✗ INCORRECTO (2 decimales)
  · yoy_pct = "0.022698%" y reportas "2,27%" → ✗ INCORRECTO (multiplicaste por 100)
  · yoy_pct = "5.164%"    y reportas "5,164%" → ✗ INCORRECTO (más de 1 decimal)

UNIDADES Y VALORACIÓN
- El nombre del cuadro indica la unidad y tipo de valoración. Identifícalos con get_metadata:
  · "miles de millones de pesos encadenados" = volumen real, referencia 2018.
  · "miles de millones de pesos" (sin "encadenados") = valores nominales (precios corrientes).
  · "promedio 2018=100" = índice base 100 en 2018.
- REGLA ESPECÍFICA IMACEC:
    · IMACEC es un índice (base 2018=100). No lo trates como moneda, volumen físico ni porcentaje.
    · NO reportes el nivel del índice (value) a menos que el usuario pida explícitamente
      "el nivel del índice", "el índice" o "IMACEC en nivel". Por defecto reporta yoy_pct.
- SIEMPRE incluye la unidad al reportar niveles ("value").
  Correcto: "El PIB fue de 52.456 miles de millones de pesos encadenados en 2024."
  Incorrecto: "El PIB fue de 52.456."
- Si reportas niveles de inversión (FBCF), menciona SIEMPRE además el tipo de valoración:
    "encadenado" o "a precios corrientes" según corresponda.
    Ejemplo: "La inversión fue de 13.188,68 miles de millones de pesos encadenados..."
    o "...a precios corrientes...".
- Variaciones porcentuales (pct, yoy_pct) no necesitan unidad (son %).
- En volúmenes encadenados (price "enc"), las variaciones reflejan cambio real
  (ajustado por inflación). En precios corrientes (price "co"), las variaciones
  incluyen tanto actividad real como efecto precios. Menciónalo si es relevante.

REGLAS DE INTERPRETACIÓN DE LA PREGUNTA
- SINONIMIA OBLIGATORIA:
    · "Formación Bruta de Capital Fijo", "Formacion Bruta de Capital Fijo", "FBCF" e "inversión"/"inversion"
        se refieren al mismo concepto. Trátalos como equivalentes.
    · NO confundas "Formación Bruta de Capital Fijo (FBCF)" con "Formación Bruta de Capital".
        Si el usuario pide inversión, prioriza FBCF y no cambies a "Formación Bruta de Capital"
        salvo que el usuario lo solicite explícitamente.
    · Si el usuario habla de "inversión", prioriza series/componentes etiquetados como
        "Formación Bruta de Capital Fijo" cuando corresponda en el cuadro.
- "cuánto creció", "cuánto cayó", "variación" (sin más detalle),
  "el valor", "el dato", "la cifra", "cuánto fue" → metric "yoy_pct".
- "en el margen", "respecto al período anterior", "trimestre/mes anterior",
  "variación en el margen" → metric "pct".
  "pct" es SIEMPRE la variación respecto al período inmediatamente anterior.
- POR DEFECTO, reporta SIEMPRE variaciones porcentuales (yoy_pct).
    NO reportes niveles (value) ni aumentos/disminuciones absolutas salvo que
    el usuario pida explícitamente "PIB en pesos", "a precios corrientes",
    "PIB per cápita" o "a cuánto asciende".
    NO incluyas la variación en el margen (pct) como complemento.
    Reporta SOLO yoy_pct salvo que el usuario pida explícitamente "en el margen".
- "aceleró", "desaceleró", "aceleración" → metric "acceleration_pct".
  La aceleración es el cambio en "pct" (variación período anterior) entre dos períodos consecutivos.
  SIEMPRE reporta PRIMERO la variación "pct" del período actual y del período anterior,
  y LUEGO la aceleración como la diferencia entre ambas variaciones "pct".
  NUNCA uses yoy_pct para aceleración; la aceleración se mide SIEMPRE sobre "pct".
    Describe el resultado en términos estrictamente numéricos (magnitud y signo).
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
- SIEMPRE usa get_metadata para conocer latest_available antes de responder.
  Compara el período solicitado contra latest_available de la frecuencia correspondiente.
- CASO 1 — El usuario NO mencionó fecha ni período:
  Usa el último período disponible y responde directamente.
  Ejemplo: "El último dato disponible corresponde a enero de 2026. La variación anual
  fue **-0,5%** respecto al mismo período del año anterior."
- CASO 2 — El usuario pidió un período SIN DATOS (explícito o por referencia relativa):
  Las referencias temporales relativas también cuentan como período EXPLÍCITO
  (ej: "año pasado", "este año", "trimestre pasado", "mes pasado", "ultimo mes").
  Formato OBLIGATORIO:
  1. "Los datos de [período solicitado] aún no han sido publicados según los datos de la Base de Datos Estadísticos."
  2. "El último dato disponible corresponde a [período]. Con esa referencia, [indicador]
     registró una variación anual de **X,X%** respecto al [indicador] del mismo período
     del año anterior."
  Ejemplo: "Los datos de febrero de 2026 aún no han sido publicados según los datos de la Base de Datos Estadísticos.
  El último dato disponible corresponde a enero de 2026. Con esa referencia, el IMACEC
  registró una variación anual de **-0,5%** respecto al IMACEC del mismo período del año anterior."
  NUNCA termines solo diciendo que no hay datos. SIEMPRE entrega el último dato disponible.

DESAMBIGUACIÓN
- Si la pregunta es general, prioriza la serie total/agregada sobre componentes.
- Usa list_series si no tienes claro cuál serie elegir.
- Si el usuario pide una actividad específica (ej: "minería") y esa actividad NO existe en
    las series disponibles del cuadro, indícalo explícitamente. NO sustituyas por PIB total
    ni por otra actividad y NO inventes cifras para la actividad solicitada.
- En conversaciones de seguimiento, hereda indicador/serie/frecuencia/métrica del turno anterior.
  Solo reemplaza lo que el nuevo turno cambie explícitamente.
- REGLA ANUAL PIB vs IMACEC: si el usuario pregunta por un dato ANUAL de la economía
  y el cuadro cargado es IMACEC, indica que para cifras anuales la referencia oficial es el PIB.
  IMACEC es un indicador de frecuencia mensual. Para reportes anuales, prioriza siempre PIB.
- PREGUNTAS METODOLÓGICAS: si el usuario pregunta "qué mide", "cómo se calcula", "qué es",
  da una definición breve y agrega: "Para mayor detalle metodológico, puedes consultar los
  documentos disponibles en la web oficial del Banco Central de Chile."
- CONTINUIDAD CONVERSACIONAL: cuando la pregunta es genérica ("cuánto creció la economía"),
  identifica el indicador y frecuencia más relevante y menciónalo explícitamente.
  Ejemplo: "El último dato disponible para el crecimiento de la economía corresponde
  al PIB del 4to trimestre de 2025. La serie se publica en frecuencia trimestral."

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

REGLA DE REDACCIÓN — ESTACIONALIDAD
- Si seasonality = "nsa", asume serie estacional/sin ajuste estacional por defecto y NO lo menciones
    explícitamente en la redacción, salvo que el usuario lo pida.
- Si seasonality = "sa", SIEMPRE menciona explícitamente en la respuesta que la serie es
    "desestacionalizada" o "desestacionalizado".
  Esta mención debe aparecer en la oración principal donde reportas el dato.
- Si el usuario pide distinguir ajuste estacional vs no ajustado, explicita ambos conceptos con claridad.

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
- Incluso en respuestas explicativas/causales (ej: "qué impulsó...", "qué explicó...")
    debes mencionar explícitamente los valores numéricos que sustentan la conclusión
    (porcentaje, contribución, nivel o variación) y su período.
- NUNCA preguntes "¿quieres que consulte los datos?" ni "¿quieres que lo haga?".
  Si ya identificaste la serie y el período, consulta los datos y preséntalos directamente.
- PRIORIZACIÓN DE MÉTRICAS — REGLA GENERAL:
    · POR DEFECTO reporta SIEMPRE "yoy_pct" (variación respecto al mismo período del año
      anterior). Esto aplica independientemente del calc_mode.
    · EXCEPCIÓN 1 — VARIACIÓN PERÍODO ANTERIOR: si el usuario dice "en el margen",
      "respecto al período anterior", "variación mensual/trimestral anterior", reporta "pct".
    · EXCEPCIÓN 2 — CIFRA ORIGINAL / NIVELES: reporta "value" con su unidad
      SOLO si el usuario pide explícitamente "PIB en pesos", "PIB en dólares",
      "PIB per cápita", "a precios corrientes" o "a cuánto asciende".
      Expresiones como "el valor", "la cifra", "el dato", "cuánto fue" NO son
      solicitud de nivel; en esos casos reporta yoy_pct como métrica principal.
      Para PIB per cápita: muestra el valor aproximado SIN decimales.
    · En CUALQUIER otro caso (incluidos "cuánto creció", "cuánto cayó", "variación",
      "crecimiento", preguntas genéricas), reporta "yoy_pct".
    Después de la métrica principal, NO complementes con otras métricas (pct, value)
    salvo que el usuario lo pida explícitamente.

ESTILO DE RESPUESTA
- Español, claro, preciso y detallado.
- REGLA DE LENGUAJE PARA VARIACIONES (CRÍTICA):
    · Cuando reportes variaciones porcentuales (mensual/trimestral/interanual), NO menciones
        la palabra "índice"/"indice".
    · En esas oraciones, NO uses "aceleración", "aceleró", "desaceleró", "cambio",
        "delta" ni "variación del cambio".
    · En respuestas de variación, NO incluyas niveles ni montos absolutos.
    · Redacta solo en términos de "variación mensual/trimestral" o "variación interanual"
        y su valor numérico.
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
- REGLA DE REDACCIÓN PARA VARIACIONES NEGATIVAS (CRÍTICA):
    · Si la variación es negativa, usa formulaciones como:
      "varió **-0,3%**" ✓ / "registró una variación de **-0,3%**" ✓ / "cayó **0,3%**" ✓
    · NUNCA combines un verbo negativo con un valor negativo:
      "disminuyó **-0,3%**" ✗ / "cayó **-0,3%**" ✗ / "retrocedió **-0,3%**" ✗
    · Regla simple: si usas el signo "-", el verbo debe ser neutro ("varió", "registró").
      Si usas verbo negativo ("cayó", "retrocedió"), omite el signo "-".
        · Para contribuciones menores a cero (pp), usa redacción neutral con signo:
            "tuvo una contribución de **-0,5 pp**" ✓ / "registró un aporte de **-0,5 pp**" ✓
        · PROHIBIDO en contribuciones con signo: "contribución negativa de **-0,5 pp**" ✗,
            "caída de **-0,5 pp**" ✗, "disminuyó **-0,5 pp**" ✗.
- REGLA DE CONTRIBUCIONES (CRÍTICA):
    · Cuando el usuario pregunta "qué actividad impulsó", "qué afectó a la baja",
      "qué explicó el crecimiento/caída":
      1. Usa rank_series con la métrica adecuada para el período solicitado.
      2. Ordena los resultados por VALOR ABSOLUTO de la contribución (mayor primero).
      3. Presenta las actividades que más contribuyeron con su valor numérico.
    · Siempre separa claramente contribuciones positivas de negativas.
    · UNIDAD OBLIGATORIA: las contribuciones se expresan en "pp" (puntos porcentuales),
      NUNCA en "%". Ejemplo correcto: "**0,5 pp**". Incorrecto: "**0,5%**".
    · REDONDEO: las contribuciones en puntos porcentuales (pp) deben redondearse
      SIEMPRE a UN (1) decimal. Ejemplo: 0,54 → **0,5 pp**; -0,67 → **-0,7 pp**.
- ESTRUCTURA OBLIGATORIA DE RESPUESTA (3 bloques separados por salto de línea):
    IMPORTANTE: cada bloque debe ir en un PÁRRAFO SEPARADO (con un salto de línea entre bloques).
    Los 3 bloques son OBLIGATORIOS en TODA respuesta, sin excepción.
    1. INTRODUCCIÓN (1-2 oraciones, SIEMPRE presente): línea contextual que ancla indicador, período y frecuencia.
       - Si los datos del período solicitado EXISTEN: "El IMACEC de enero de 2026 muestra los siguientes resultados."
       - Si el período solicitado NO tiene datos: "Los datos de [período] aún no han
         sido publicados según los datos de la Base de Datos Estadísticos. El último dato disponible corresponde a [período]."
       NUNCA omitas la introducción. NUNCA la fusiones con el bloque de DATOS.
    2. DATOS (1-3 oraciones): cifra principal con formato correcto (1 decimal, negrita, formato
       español). Siempre indicar contra qué se compara: "respecto al mismo período del año
       anterior" o "respecto al período anterior".
    3. RECOMENDACIÓN (OBLIGATORIA, 1 oración): SIEMPRE incluye una oración final de
       recomendación. Ejemplos:
       · "Puedes profundizar revisando el desglose sectorial del IMACEC y su trayectoria reciente."
       · "Puedes profundizar comparando la trayectoria reciente de la actividad económica y sus componentes."
       · "Puedes profundizar revisando el detalle de contribuciones por actividad en el mismo período."
       NO des opinión. La recomendación es solo una sugerencia de exploración de datos.
       NUNCA omitas este bloque.
    EJEMPLO DE FORMATO CORRECTO (3 párrafos separados):
    ---
    Los datos de febrero de 2026 aún no han sido publicados según los datos de la Base de Datos Estadísticos. El último dato disponible corresponde a enero de 2026.

    El IMACEC registró una variación interanual de **-0,1%** respecto al mismo período del año anterior.

    Puedes profundizar revisando el desglose sectorial del IMACEC y su trayectoria reciente.
    ---
- Si el usuario pregunta "qué impulsó" una variable agregada (ej: demanda interna),
  identifica el/los componente(s) con mayor aporte y reporta sus valores numéricos
  (contribución/variación) junto con el período antes de cualquier interpretación.
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
- Segundo párrafo (opcional): contexto descriptivo estrictamente basado en datos.
    Puedes comparar con períodos anteriores si es relevante, pero sin causalidad ni opinión.
- REGLA DE OBJETIVIDAD (CRÍTICA):
    · NO des opiniones, valoraciones ni juicios (ej: "desafíos", "fortaleza", "debilidad", "preocupante").
    · NO especules sobre causas ni factores explicativos no observados en los datos.
    · NO introduzcas contexto histórico, político o externo que no provenga de las herramientas.
    · Limítate a describir cifras, variaciones, comparaciones y tendencia numérica observada.
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

REGLA ESTRICTA DE PIB ANUAL:
- El PIB se publica en frecuencia trimestral. Un dato anual del PIB solo es válido
  si los 4 trimestres del año están publicados.
- Antes de responder una consulta anual de PIB, verifica con get_series_data
  cuántos trimestres del año solicitado tienen datos:
    · 4/4 trimestres → entrega el dato anual normalmente.
    · 1-3 trimestres → responde: "Aún no se han publicado todos los datos del PIB [año].
      Solo se dispone de [N]/4 trimestres publicados. El último dato anual completo
      corresponde a [año anterior], con una variación de **X,X%**."
    · 0/4 trimestres → responde: "Aún no se han publicado datos del PIB [año].
      El último dato anual completo corresponde a [año anterior]."
- NUNCA presentes un valor anual del PIB si faltan trimestres por publicar.

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

_MONTH_NAME_TO_NUM = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

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

    # Normaliza aliases de calculo para asegurar columnas estables en CSV.
    if "yoy_pct" not in out and "yoy" in out:
        out["yoy_pct"] = out.get("yoy")
    if "pct" not in out and "prev_period" in out:
        out["pct"] = out.get("prev_period")

    for field in _PCT_FIELDS:
        if field in out and out[field] is not None:
            out[field] = f"{out[field]}%"
    # Alias explicito para compatibilidad con nomenclatura BDE.
    if "yoy_pct" in out and "YTYPCT" not in out:
        out["YTYPCT"] = out.get("yoy_pct")
    # Eliminar _display si existiera de versiones anteriores
    out.pop("_display", None)
    return out


def _safe_float(value: Any) -> Optional[float]:
    """Parsea valores numéricos o strings tipo 'X%' a float."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    text = text.replace("%", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _contribution_direction(value: float) -> str:
    if value > 0:
        return "alza"
    if value < 0:
        return "baja"
    return "neutral"


def _sanitize_contribution_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Entrega un registro de contribución con magnitud absoluta y referencia de signo."""
    out = dict(record)
    raw_value = _safe_float(out.get("value"))
    if raw_value is None:
        return out

    out["value_signed"] = out.get("value")
    out["value"] = abs(raw_value)
    out["value_abs"] = abs(raw_value)
    out["contribution_direction"] = _contribution_direction(raw_value)
    out["contribution_sign"] = "positive" if raw_value > 0 else "negative" if raw_value < 0 else "neutral"
    return out


def _sanitize_contribution_tool_result(name: str, args: Dict[str, Any], result: str) -> str:
    """Sanitiza payload de contribuciones antes de pasarlo al LLM."""
    try:
        parsed = json.loads(result)
    except Exception:
        return result

    is_contribution = str(args.get("calc_mode") or "").lower() == "contribution"
    if name == "rank_series":
        metric = str(parsed.get("metric") or args.get("metric") or "").lower()
        is_contribution = is_contribution or metric == "value"
        if is_contribution and isinstance(parsed.get("ranking"), list):
            parsed["ranking"] = [
                _sanitize_contribution_record(row) if isinstance(row, dict) else row
                for row in parsed["ranking"]
            ]
    elif name == "get_series_data":
        if is_contribution and isinstance(parsed.get("records"), list):
            parsed["records"] = [
                _sanitize_contribution_record(row) if isinstance(row, dict) else row
                for row in parsed["records"]
            ]

    try:
        return json.dumps(parsed, ensure_ascii=False)
    except Exception:
        return result


def _canonicalize_period_token(freq: str, raw_period: Any) -> str:
    """Normaliza períodos a la llave esperada por frecuencia.

    Ejemplos:
    - M: 2024-02-01 -> 2024-02
    - T: 2024-02-01 -> 2024-Q1
    - A: 2024-02-01 -> 2024
    """
    text = str(raw_period or "").strip()
    if not text:
        return ""

    freq_code = str(freq or "").strip().upper()

    normalized_text = unicodedata.normalize("NFKD", text.lower())
    normalized_text = "".join(ch for ch in normalized_text if not unicodedata.combining(ch))
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    quarter_match = re.fullmatch(r"((?:19|20)\d{2})-Q([1-4])", text, re.IGNORECASE)
    if quarter_match:
        year = quarter_match.group(1)
        quarter = quarter_match.group(2)
        if freq_code == "A":
            return year
        return f"{year}-Q{quarter}"

    month_match = re.fullmatch(r"((?:19|20)\d{2})[-/]([0-1]\d)(?:[-/]([0-3]\d))?", text)
    if month_match:
        year = int(month_match.group(1))
        month = int(month_match.group(2))
        if month < 1 or month > 12:
            return text
        if freq_code == "M":
            return f"{year:04d}-{month:02d}"
        if freq_code == "T":
            quarter = ((month - 1) // 3) + 1
            return f"{year:04d}-Q{quarter}"
        if freq_code == "A":
            return f"{year:04d}"
        return text

    month_name_match = re.fullmatch(
        r"(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?((?:19|20)\d{2})",
        normalized_text,
    )
    if month_name_match:
        month_name = month_name_match.group(1)
        year = int(month_name_match.group(2))
        month = _MONTH_NAME_TO_NUM.get(month_name)
        if month is None:
            return text
        if freq_code == "M":
            return f"{year:04d}-{month:02d}"
        if freq_code == "T":
            quarter = ((month - 1) // 3) + 1
            return f"{year:04d}-Q{quarter}"
        if freq_code == "A":
            return f"{year:04d}"
        return text

    year_match = re.fullmatch(r"((?:19|20)\d{2})", text)
    if year_match and freq_code == "A":
        return year_match.group(1)

    return text


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
        freq_code = str(args.get("frequency") or "").upper()
        if args.get("period"):
            period = _canonicalize_period_token(freq_code, args.get("period"))
            records = [r for r in records if r["period"] == period]
        elif args.get("period_start") or args.get("period_end"):
            start = _canonicalize_period_token(freq_code, args.get("period_start"))
            end = _canonicalize_period_token(freq_code, args.get("period_end")) or "9999"
            records = [r for r in records if start <= r["period"] <= end]
        if not records:
            debug_params = dict(args)
            if args.get("period"):
                debug_params["period"] = _canonicalize_period_token(freq_code, args.get("period"))
            else:
                if args.get("period_start"):
                    debug_params["period_start"] = _canonicalize_period_token(freq_code, args.get("period_start"))
                if args.get("period_end"):
                    debug_params["period_end"] = _canonicalize_period_token(freq_code, args.get("period_end"))
            return json.dumps({"error": "Sin datos para los parámetros dados", "params": debug_params})
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

def _build_source_footer(
    observations: Dict[str, Any],
    source_url_override: Optional[str] = None,
) -> Optional[str]:
    """Construye un footer con el link a la fuente BDE desde el payload."""
    source_url = str(source_url_override or observations.get("source_url") or "").strip()
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
    if mode == "contribution":
        return (
            "REGLA ESTRICTA DE REDACCION PARA CONTRIBUCIONES:\n"
            "0. Estas reglas aplican de forma genérica para consultas de contribución en PIB e IMACEC.\n"
            "1. En el PRIMER PARRAFO comienza mencionando el PERIODO analizado "
            "(ej: 'En el 3er trimestre de 2025, ...').\n"
            "2. Los datos son CONTRIBUCIONES al crecimiento. Usa SIEMPRE la unidad '%' y NUNCA 'pp'.\n"
            "   Correcto: '**0,5%**'  |  Incorrecto: '**0,5 pp**'.\n"
            "3. Redondea SIEMPRE a 1 decimal: 0,54 → **0,5%**; -0,67 → 'disminuyó **0,7%**'.\n"
            "4. Ordena por VALOR ABSOLUTO de la contribución (mayor primero).\n"
            "5. Separa claramente contribuciones positivas de negativas.\n"
            "6. REGLA DE SIGNO Y LÉXICO (OBLIGATORIA): muestra SIEMPRE valor absoluto y usa redacción consistente con el signo.\n"
            "   - si dato < 0: usa 'disminuyó **X,X%**' (sin signo '-').\n"
            "   - si dato > 0: usa 'creció **X,X%**' o 'aumentó **X,X%**'.\n"
            "   - si dato = 0: usa 'mostró una variación de **0,0%**'.\n"
            "7. MATRIZ DE SINÓNIMOS ESTILO INFORME IMACEC (ASOCIADA AL SIGNO):\n"
            "   - positivos: 'creció', 'aumentó', 'registró un incremento de', 'presentó una expansión de'.\n"
            "   - negativos: 'disminuyó', 'registró una contracción de', 'presentó una disminución de'.\n"
            "   - neutro: 'mostró una variación de'.\n"
            "8. PROHIBIDO mostrar valores con signo negativo en contribuciones (ej: '-0,7%').\n"
            "8.b. También está PROHIBIDO usar 'pp' en cualquier oración de contribución.\n"
            "8.c. Si usas 'disminuyó/creció/aumentó' o sinónimos, debes complementar con "
            "'respecto al mismo período del año anterior'.\n"
            "8.d. CHEQUEO FINAL OBLIGATORIO ANTES DE RESPONDER: si en el borrador aparece cualquier "
            "porcentaje con signo '-' (por ejemplo '-5,2%'), debes reescribir usando valor absoluto y "
            "verbo consistente ('disminuyó').\n"
            "8.e. RESPUESTA INVÁLIDA si incluye patrones de texto con signo negativo en contribución: "
            "'-X,X%', 'con -X,X%', 'de -X,X%'.\n"
            "8.e.1. Esta regla aplica a TODAS las menciones de contribución del texto (principal, secundarias y cierre).\n"
            "8.f. RESPUESTA INVÁLIDA si usa frases como 'contribuciones negativas de ...' o "
            "'incidencias negativas ... con -X,X%'. Reescribe con plantilla por actividad y valor absoluto.\n"
            "8.f.1. RESPUESTA INVÁLIDA si usa la frase 'a la baja' en contribuciones. "
            "Para signo negativo usa únicamente 'disminuyó' o 'disminución'.\n"
            "8.g. PLANTILLA OBLIGATORIA POR ACTIVIDAD EN CONTRIBUCIÓN: "
            "'[Actividad]: disminuyó/creció/aumentó **X,X%** respecto al mismo período del año anterior'.\n"
            "8.g.1. REGLA OBLIGATORIA POR CADA PORCENTAJE: toda cifra porcentual de contribución "
            "(positiva o negativa) debe ir acompañada explícitamente de la frase "
            "'respecto al mismo período del año anterior'. Si una actividad queda solo con '0,3%' "
            "sin esa referencia, la respuesta es inválida.\n"
            "9. Respuesta objetiva: describe cifras y relaciones de compensación, sin juicios de valor.\n"
            "10. Mantén longitud habitual de respuesta: no imites un informe extenso; solo ajusta vocabulario y estilo.\n"
            "7. FLUJO OBLIGATORIO: primero llama list_series y luego llama rank_series "
            "con metric='value' y order='desc' para obtener el ranking de contribuciones.\n"
            "7.b. TOOLING MÍNIMO OBLIGATORIO: antes de redactar debes haber ejecutado "
            "get_metadata y rank_series para el periodo objetivo.\n"
            "8. Si calc_mode='contribution', está PROHIBIDO responder solo con la serie "
            "agregada del PIB total. Debes reportar actividades/componentes.\n"
            "8.b. En preguntas del tipo 'qué actividad impulsó', está PROHIBIDO redactar "
            "el bloque DATOS como 'el PIB registró X%'. Debes listar actividades con su aporte en %.\n"
            "9. Debes reportar al menos 5 actividades (si existen) con mayor aporte y, "
            "si hay aportes negativos, mencionar explícitamente la principal caída.\n"
            "10. Si rank_series no devuelve ranking para el periodo solicitado, debes "
            "decir explícitamente que no hay contribuciones disponibles para ese periodo. "
            "NO reemplaces esa situación con 'PIB creció X%'.\n"
            "11. CHEQUEO FINAL OBLIGATORIO: si no hay salida de rank_series en el contexto "
            "de herramientas, no redactes la respuesta final y vuelve a llamar rank_series.\n"
            "12. Si existe ranking, usa SOLO esos resultados para el bloque DATOS; no cambies "
            "a una métrica de variación agregada del PIB."
        )
    if mode == "prev_period":
        return (
            "REGLA ESTRICTA DE REDACCION: en el PRIMER PARRAFO (primera oracion) "
            "comienza mencionando el PERIODO analizado (ej: 'En el 3er trimestre de 2025, ...') "
            "y reporta PRIMERO el valor de 'pct'. "
            "No comiences con 'value' ni con 'yoy_pct'."
        )
    # Para "original", "yoy" y cualquier otro: siempre yoy_pct por defecto
    return (
        "REGLA ESTRICTA DE REDACCION:\n"
        "1. En el PRIMER PARRAFO comienza mencionando el PERIODO analizado "
        "(ej: 'En el 3er trimestre de 2025, ...').\n"
        "2. Reporta SOLO la variación interanual (yoy_pct) como dato principal.\n"
        "3. PROHIBIDO incluir niveles (value) o cifras de índice en la respuesta. "
        "NO uses get_series_data con metric='value' ni menciones niveles de índice.\n"
        "4. La única excepción para reportar 'value' es si el usuario pide explícitamente "
        "'PIB en pesos', 'PIB en dólares', 'PIB per cápita' o 'a precios corrientes'.\n"
        "5. Expresiones como 'el valor', 'la cifra', 'cuánto fue', 'el dato' "
        "NO son solicitud de nivel; en esos casos reporta yoy_pct.\n"
        "6. No comiences con 'value' ni con 'pct' salvo que la pregunta lo requiera explícitamente."
    )


def _build_seasonality_strict_instruction(
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[str]:
    """Construye instrucción estricta de estacionalidad según contexto resuelto.

    Prioriza la estacionalidad resuelta por entidades (clasificador/reglas de negocio)
    por sobre metadatos agregados del cuadro.
    """
    seasonality = str(entities_ctx.get("seasonality_ent") or entities_ctx.get("seasonality") or "").strip().lower()
    if not seasonality:
        seasonality = str((observations.get("classification") or {}).get("seasonality") or "").strip().lower()

    if seasonality == "sa":
        return (
            "REGLA ESTRICTA DE ESTACIONALIDAD: la serie consultada es desestacionalizada. "
            "Debes escribir explícitamente 'desestacionalizado' o 'desestacionalizada' "
            "en la primera oración donde reportas el dato principal."
        )
    if seasonality == "nsa":
        return (
            "REGLA ESTRICTA DE ESTACIONALIDAD: la serie consultada es no desestacionalizada. "
            "No menciones desestacionalización salvo que el usuario lo pida explícitamente."
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


def _humanize_activity_label(label: str) -> str:
    """Convierte etiquetas técnicas de actividad a texto más natural."""
    raw = str(label or "").strip()
    if not raw:
        return ""
    humanized = re.sub(r"[_\-]+", " ", raw)
    humanized = re.sub(r"\s+", " ", humanized).strip()
    return humanized


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

    requested_activity_raw = str(entities_ctx.get("activity_ent") or "").strip()
    requested_activity = _normalize_token(requested_activity_raw)
    if requested_activity in {"", "none", "null"}:
        return None

    # IMACEC/PIB son indicadores agregados, no actividades económicas.
    if requested_activity in {"imacec", "pib"}:
        return None

    available_activities: List[Tuple[str, str]] = []
    for series in observations.get("series", []) or []:
        cls = series.get("classification_series", {})
        if not isinstance(cls, dict):
            continue
        raw_label = str(cls.get("activity") or "").strip()
        activity = _normalize_token(raw_label)
        if activity and raw_label:
            available_activities.append((activity, raw_label))

    if not available_activities:
        return None

    available_set = {token for token, _ in available_activities}
    if requested_activity in available_set:
        return None

    labels_by_token: Dict[str, str] = {}
    for token, label in available_activities:
        if token not in labels_by_token:
            labels_by_token[token] = _humanize_activity_label(label)
    options = ", ".join(sorted(labels_by_token.values())[:8])

    indicator = str(
        entities_ctx.get("indicator_ent")
        or (observations.get("classification") or {}).get("indicator")
        or ""
    ).strip().upper()
    if indicator not in {"PIB", "IMACEC"}:
        indicator = "PIB o IMACEC"

    return (
        "REGLA ESTRICTA DE ACTIVIDAD FALTANTE: la actividad solicitada no está disponible para este indicador en la Base de datos estadísticos. "
        f"Debes responder literalmente: 'Esta actividad no se encuentra disponible en la Base de datos estadísticos para el indicador {indicator}'. "
        "PRIMER PÁRRAFO OBLIGATORIO: inicia con la estructura exacta "
        "'En [PERIODO], esta actividad no se encuentra disponible en la Base de datos estadísticos para el indicador "
        f"{indicator}'. "
        "PROHIBIDO agregar en ese primer párrafo frases como 'no corresponde a la actividad solicitada' "
        "o mencionar una actividad proxy/cercana por nombre. "
        f"Actividad pedida: '{requested_activity_raw or requested_activity}'. "
        "NO escribas una introducción que afirme contribución de la actividad solicitada en el período, "
        "porque contradice la no disponibilidad. "
        f"Luego sugiere actividades disponibles del mismo cuadro: {options}. "
        "Usa nombres naturales de actividades (sin guiones bajos, sin códigos técnicos). "
        "NO reemplaces por la serie agregada (PIB total) ni por otra actividad, y NO inventes valores."
    )


def _build_prevalidated_missing_specific_activity_instruction(
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[str]:
    """Construye una instrucción alternativa cuando activity es specific pero viene vacía.

    Esta validación se hace antes de generar el resto de instrucciones para que el
    LLM no tenga que decidir si la actividad está o no en el cuadro.
    """
    activity_cls = str(
        entities_ctx.get("activity_cls_resolved")
        or entities_ctx.get("activity_cls")
        or ""
    ).strip().lower()
    if activity_cls != "specific":
        return None

    requested_activity_raw = str(entities_ctx.get("activity_ent") or "").strip()
    if requested_activity_raw:
        return None

    labels_by_token: Dict[str, str] = {}
    for series in observations.get("series", []) or []:
        cls = series.get("classification_series", {})
        if not isinstance(cls, dict):
            continue
        raw_label = str(cls.get("activity") or "").strip()
        token = _normalize_token(raw_label)
        if token and raw_label and token not in labels_by_token:
            labels_by_token[token] = _humanize_activity_label(raw_label)

    options = ", ".join(sorted(labels_by_token.values())[:8]) if labels_by_token else "(sin actividades disponibles en este cuadro)"

    indicator = str(
        entities_ctx.get("indicator_ent")
        or (observations.get("classification") or {}).get("indicator")
        or ""
    ).strip().upper()
    if indicator not in {"PIB", "IMACEC"}:
        indicator = "PIB o IMACEC"

    return (
        "VALIDACIÓN PREVIA DEL SISTEMA: activity_cls='specific' pero activity normalizada vacía. "
        "No debes inferir ni decidir una actividad alternativa. "
        f"Debes responder literalmente: 'Esta actividad no se encuentra disponible en la Base de datos estadísticos para el indicador {indicator}'. "
        "PRIMER PÁRRAFO OBLIGATORIO: inicia con la estructura exacta "
        "'En [PERIODO], esta actividad no se encuentra disponible en la Base de datos estadísticos para el indicador "
        f"{indicator}'. "
        "PROHIBIDO agregar en ese primer párrafo frases como 'no corresponde a la actividad solicitada' "
        "o mencionar una actividad proxy/cercana por nombre. "
        "NO escribas una introducción que afirme contribución de la actividad solicitada en el período, "
        "porque contradice la no disponibilidad. "
        f"Luego menciona actividades disponibles del cuadro: {options}. "
        "Usa nombres naturales de actividades (sin guiones bajos, sin códigos técnicos). "
        "NO reemplaces por PIB total, NO uses actividades proxy y NO inventes cifras. "
        "Esta regla alternativa reemplaza el flujo de contribución específica para este turno."
    )


def _build_contribution_activity_focus_instruction(
    question: str,
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[str]:
    calc_mode = str(
        entities_ctx.get("calc_mode_cls")
        or entities_ctx.get("calc_mode")
        or ""
    ).strip().lower()
    if calc_mode != "contribution":
        return None

    text = str(question or "").lower()
    if not any(token in text for token in ("actividad", "impuls", "aporte", "contribu")):
        return None

    latest_t = str((observations.get("latest_available") or {}).get("T") or "").strip()
    latest_label = _natural_period_label(latest_t, "T") if latest_t else "el último trimestre disponible"

    return (
        "REGLA ESTRICTA PARA PREGUNTAS DE ACTIVIDAD-CONTRIBUCIÓN: "
        "esta pregunta exige DESGLOSE POR ACTIVIDADES, no variación agregada del PIB. "
        f"Si el período mensual pedido no existe para PIB, usa directamente {latest_label}. "
        "NO escribas 'enero no publicado' para este tipo de pregunta; ancla la respuesta al trimestre disponible. "
        "OBLIGATORIO: llama rank_series (metric='value') para ese trimestre y construye el bloque DATOS "
        "con al menos 5 actividades (si existen) en formato de lista, cada una con su contribución en %. "
        "Para valores negativos, usa verbo negativo ('disminuyó') y valor absoluto sin signo. "
        "PROHIBIDO resumir como 'contribuciones negativas de varias actividades'. "
        "PROHIBIDO usar la frase 'a la baja'. En contribuciones negativas usa solo 'disminuyó' o 'disminución'. "
        "Debes explicitar cada actividad con plantilla obligatoria: "
        "'[Actividad]: disminuyó/creció/aumentó **X,X%** respecto al mismo período del año anterior'. "
        "REGLA OBLIGATORIA: cada actividad listada debe incluir explícitamente 'respecto al mismo período del año anterior'; "
        "no dejes porcentajes sueltos (por ejemplo, 'Servicios financieros, 0,3%'). "
        "CHEQUEO FINAL OBLIGATORIO: ninguna actividad puede quedar con porcentaje firmado con '-'. "
        "Esta validación aplica a todas las actividades reportadas, sin excepciones. "
        "Si aparece '-X,X%', debes convertirlo a 'disminuyó **X,X%** respecto al mismo período del año anterior'. "
        "La respuesta es inválida si no incluye lista de actividades con %. "
        "Está PROHIBIDO responder solo con 'el PIB registró X%'."
    )


def _build_economy_wording_instruction(
    question: str,
    entities_ctx: Dict[str, Any],
) -> Optional[str]:
    text = str(question or "").lower()
    if "econom" not in text:
        return None

    indicator = str(entities_ctx.get("indicator_ent") or "").strip().lower()
    if indicator == "pib":
        return (
            "REGLA DE CONTINUIDAD PARA CONSULTAS DE ECONOMÍA: en la INTRODUCCIÓN "
            "incluye literalmente la frase 'El último dato disponible para el crecimiento "
            "de la economía, medida por el PIB,'."
        )
    if indicator == "imacec":
        return (
            "REGLA DE CONTINUIDAD PARA CONSULTAS DE ECONOMÍA: en la INTRODUCCIÓN "
            "incluye literalmente la frase 'El último dato disponible para el crecimiento "
            "de la economía, medida por el IMACEC,'."
        )
    return None


def _build_recommendation_length_instruction() -> str:
    return (
        "REGLA DE RECOMENDACIÓN: el bloque final de recomendación debe tener una sola oración, "
        "pero más desarrollada (mínimo 16 palabras), conectada con la consulta del usuario, "
        "sin opiniones ni proyecciones."
    )


def _build_special_query_mapping_instruction(
    question: str,
    entities_ctx: Dict[str, Any],
) -> Optional[str]:
    text = str(question or "").lower()
    text_norm = unicodedata.normalize("NFKD", text)
    text_norm = "".join(ch for ch in text_norm if not unicodedata.combining(ch))

    rules: List[str] = []
    price_ent = str(entities_ctx.get("price_ent") or entities_ctx.get("price") or "").strip().lower()

    if "imacec" in text_norm and "ultimo trimestre" in text_norm:
        rules.append(
            "CASO IMACEC TRIMESTRAL: cuando pregunten por 'IMACEC del último trimestre', "
            "debes usar frecuencia trimestral (T), no mensual. "
            "Usa el último trimestre disponible del IMACEC y reporta una sola variación anual "
            "del trimestre (no desgloses por meses dentro del trimestre)."
        )

    if "demanda interna" in text_norm:
        rules.append(
            "CASO DEMANDA INTERNA: interpreta 'demanda interna' como gasto interno. "
            "Responde con la serie de demanda interna en frecuencia trimestral y su variación anual."
        )

    if "inversion" in text_norm or "inversión" in text:
        rules.append(
            "CASO INVERSIÓN: cuando la pregunta sea 'inversión en Chile', usa como referencia "
            "la serie 'Formación Bruta de Capital Fijo' del cuadro de demanda interna/gasto del PIB. "
            "No reemplaces por PIB total salvo que no exista esa serie en el cuadro."
        )

    if "pib per capita" in text_norm or "pib per cápita" in text:
        rules.append(
            "CASO PIB PER CÁPITA: reporta SIEMPRE valor original (value), nunca yoy_pct ni pct, "
            "salvo que el usuario pida explícitamente variación. "
            "REGLA DE UN SOLO VALOR: entrega una única cifra principal para el período solicitado "
            "(o el último disponible si no hay período explícito); NO listes múltiples años/trimestres/meses "
            "a menos que el usuario pida explícitamente un rango. "
            "Redondea a entero sin decimales y con separador de miles '.'."
            " (ej: 16.586). Para rangos anuales, lista cada año con su valor aproximado. "
            "No reemplaces PIB per cápita por PIB total ni por otra serie agregada. "
            "Si no hay serie de PIB per cápita en los resultados, indica esa falta y sugiere reformular."
        )

    if (
        "pib en pesos" in text_norm
        or "precios corrientes" in text_norm
        or "precio corriente" in text_norm
        or "pib nominal" in text_norm
        or "a cuanto asciende el pib" in text_norm
        or price_ent == "co"
    ):
        rules.append(
            "CASO PIB EN PESOS / PRECIOS CORRIENTES: reporta SIEMPRE valor original (value) en pesos, "
            "nunca yoy_pct ni pct salvo solicitud explícita de variación. "
            "REGLA DE UN SOLO VALOR: entrega una única cifra principal para el período solicitado "
            "(o el último disponible si no hay período explícito); NO listes múltiples períodos "
            "a menos que el usuario pida explícitamente un rango. "
            "redondeado a entero sin decimales y con separador de miles '.'. "
            "NO multipliques ni reescales el valor; úsalo tal como lo entrega get_series_data. "
            "Si la unidad es 'miles de millones de pesos', NO agregues ceros adicionales."
        )

    calc_mode = str(
        entities_ctx.get("calc_mode_cls")
        or entities_ctx.get("calc_mode")
        or ""
    ).strip().lower()
    activity_cls = str(
        entities_ctx.get("activity_cls_resolved")
        or entities_ctx.get("activity_cls")
        or ""
    ).strip().lower()
    if calc_mode == "contribution" and activity_cls == "specific":
        rules.append(
            "CASO CONTRIBUCIÓN ESPECÍFICA: cuando consulten cuánto contribuyó una actividad específica, "
            "entrega primero una respuesta directa de ESA actividad para el período objetivo. "
            "No incluyas ranking de otras actividades salvo que el usuario lo solicite explícitamente."
        )

    if not rules:
        return None

    return "REGLAS ESPECIALES DE INTERPRETACIÓN:\n- " + "\n- ".join(rules)


def _build_specific_contribution_directness_instruction(question: str) -> Optional[str]:
    text = str(question or "").lower()
    text_norm = unicodedata.normalize("NFKD", text)
    text_norm = "".join(ch for ch in text_norm if not unicodedata.combining(ch))

    if ("contribuy" in text_norm or "contribuyo" in text_norm) and ("imacec" in text_norm or "pib" in text_norm):
        return (
            "REGLA DE CONTRIBUCIÓN DIRECTA: si la pregunta es 'cuánto contribuyó [actividad] ...', "
            "responde de forma directa con ESA actividad y su valor para el período objetivo. "
            "No agregues ranking de otras actividades salvo que el usuario lo pida explícitamente. "
            "PROHIBIDO usar la palabra 'negativa' o frases equivalentes. "
            "Usa redacción consistente con el signo y valor absoluto: "
            "si dato<0 usa 'disminuyó **X,X%** respecto al mismo período del año anterior'; "
            "si dato>0 usa 'creció **X,X%** respecto al mismo período del año anterior'; "
            "si dato=0 usa 'mostró una variación de **0,0%** respecto al mismo período del año anterior'. "
            "PROHIBIDO usar la frase 'a la baja'; para signo negativo usa solo 'disminuyó' o 'disminución'. "
            "REGLA OBLIGATORIA: toda cifra porcentual escrita en la respuesta debe incluir la referencia "
            "'respecto al mismo período del año anterior'; no dejes porcentajes sin contexto temporal. "
            "PROHIBIDO usar 'pp' y PROHIBIDO mostrar signo negativo en el valor. "
            "CHEQUEO FINAL OBLIGATORIO: si redactaste '-X,X%' debes reescribir con valor absoluto y verbo consistente. "
            "Esta regla aplica a cualquier mención adicional de contribución en la respuesta. "
            "En el bloque DATOS usa solo una oración con la contribución de esa actividad."
        )
    return None


def _build_original_series_force_instruction(
    question: str,
    entities_ctx: Dict[str, Any],
) -> Optional[str]:
    """Fuerza respuesta de serie original para PIB precios corrientes y PIB per cápita."""
    text = str(question or "").lower()
    text_norm = unicodedata.normalize("NFKD", text)
    text_norm = "".join(ch for ch in text_norm if not unicodedata.combining(ch))

    price_ent = str(entities_ctx.get("price_ent") or entities_ctx.get("price") or "").strip().lower()
    indicator_ent = str(entities_ctx.get("indicator_ent") or "").strip().lower()
    activity_ent = str(entities_ctx.get("activity_ent") or "").strip().lower()

    is_per_capita = (
        "pib per capita" in text_norm
        or "pib per cápita" in text
        or indicator_ent == "pib_per_capita"
        or activity_ent == "per_capita"
    )
    is_current_prices = (
        "precios corrientes" in text_norm
        or "precio corriente" in text_norm
        or "pib en pesos" in text_norm
        or "pib nominal" in text_norm
        or "a cuanto asciende el pib" in text_norm
        or price_ent == "co"
    )

    if not (is_per_capita or is_current_prices):
        return None

    return (
        "REGLA DE MAXIMA PRIORIDAD — SERIE ORIGINAL OBLIGATORIA: para PIB per cápita o PIB a precios corrientes, "
        "la cifra principal debe salir SIEMPRE del campo value (serie original). "
        "PROHIBIDO usar yoy_pct o pct como dato principal, salvo que el usuario pida explícitamente variación. "
        "OBLIGATORIO: llama get_series_data y usa value del período solicitado (o del último disponible). "
        "OBLIGATORIO: entrega una sola cifra principal en el bloque DATOS. "
        "Si respondes con yoy_pct/pct en estos casos, la respuesta es inválida y debes rehacerla con value."
    )


def _build_value_no_rescale_instruction(question: str, entities_ctx: Optional[Dict[str, Any]] = None) -> Optional[str]:
    text = str(question or "").lower()
    text_norm = unicodedata.normalize("NFKD", text)
    text_norm = "".join(ch for ch in text_norm if not unicodedata.combining(ch))
    ctx = entities_ctx or {}
    price_ent = str(ctx.get("price_ent") or ctx.get("price") or "").strip().lower()
    indicator_ent = str(ctx.get("indicator_ent") or "").strip().lower()
    activity_ent = str(ctx.get("activity_ent") or "").strip().lower()

    is_original_value_query = any(
        token in text_norm
        for token in (
            "pib en pesos",
            "precios corrientes",
            "precio corriente",
            "pib nominal",
            "pib per capita",
            "pib per cápita",
            "a cuanto asciende",
        )
    ) or price_ent == "co" or indicator_ent == "pib_per_capita" or activity_ent == "per_capita"

    if is_original_value_query:
        return (
            "REGLA DE ESCALA PARA VALORES ORIGINALES: cuando entregues value, conserva exactamente "
            "la escala entregada por get_series_data (no multiplicar, no dividir, no convertir unidades). "
            "Solo redondea y formatea separadores de miles. "
            "Ejemplo: si get_series_data entrega 57246,76 (miles de millones), reporta 57.247 y nunca 57.247.000. "
            "OBLIGATORIO: para PIB a precios corrientes o PIB per cápita, el bloque DATOS debe usar "
            "solo value (serie original) como cifra principal; no uses yoy_pct ni pct salvo solicitud explícita de variación. "
            "OBLIGATORIO: entrega una sola cifra principal para el período solicitado (o último disponible si no se especifica período)."
        )
    return None


def _build_annual_pib_completeness_instruction(
    question: str,
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[str]:
    """Inyecta regla de PIB anual: solo válido si 4/4 trimestres están publicados."""
    indicator = str(entities_ctx.get("indicator_ent") or "").strip().lower()
    if indicator != "pib":
        return None

    # Solo aplica si se pregunta por un año completo
    requested_year = _extract_requested_year(entities_ctx, question)
    if not requested_year:
        return None

    freq_code = _resolve_requested_frequency(entities_ctx, observations)

    # For annual frequency, check if requested year is beyond latest annual data
    if freq_code == "A":
        latest_available = observations.get("latest_available") or {}
        latest_a = str(latest_available.get("A") or "").strip()
        if latest_a and requested_year > latest_a:
            return (
                f"REGLA ESTRICTA DE PIB ANUAL: aún no se han publicado datos del PIB {requested_year}. "
                f"Debes indicar: 'Aún no se han publicado datos del PIB {requested_year}.' "
                f"OBLIGATORIO: llama list_series para obtener el series_id, luego llama "
                f"get_series_data con frequency='A' y period='{latest_a}' para obtener la cifra real. "
                "NO reportes niveles absolutos en pesos en la respuesta final; "
                "reporta solo la variación anual (yoy_pct). "
                f"NO respondas con ninguna cifra sin haber llamado get_series_data."
            )
        if latest_a and requested_year == latest_a:
            return None  # data exists for the requested year
        # If latest_a is not set, fall through to quarterly check

    if freq_code not in {"T", ""}:
        # Si la frecuencia resuelta no es trimestral ni vacía, no aplica
        if freq_code != "T":
            return None

    latest_available = observations.get("latest_available") or {}
    latest_t = str(latest_available.get("T") or "").strip()
    if not latest_t:
        return None

    # Contar trimestres disponibles para el año solicitado
    q_match = re.fullmatch(r"(\d{4})-Q([1-4])", latest_t)
    if not q_match:
        return None

    latest_year = q_match.group(1)
    latest_q = int(q_match.group(2))

    if requested_year > latest_year:
        # 0/4 trimestres
        return (
            f"REGLA ESTRICTA DE PIB ANUAL: aún no se han publicado datos del PIB {requested_year}. "
            f"Debes indicar: 'Aún no se han publicado datos del PIB {requested_year}.' "
            "Luego USA get_series_data para obtener el último dato anual completo disponible "
            "y reporta solo la variación anual (yoy_pct), sin niveles absolutos."
        )
    elif requested_year == latest_year and latest_q < 4:
        return (
            f"REGLA ESTRICTA DE PIB ANUAL: el año {requested_year} NO está completo. "
            f"Solo se dispone de {latest_q}/4 trimestres publicados. "
            f"Debes indicar: 'Aún no se han publicado todos los datos del PIB {requested_year}. "
            f"Solo se dispone de {latest_q}/4 trimestres publicados.' "
            "Luego USA get_series_data para obtener el último dato anual completo disponible "
            "y reporta solo la variación anual (yoy_pct), sin niveles absolutos."
        )
    return None


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
                f"Debes indicar: 'Los datos del {semester_label} de {semester_year} aún no han sido "
                f"publicados en su totalidad según los datos de la Base de Datos Estadísticos. "
                f"El último dato disponible corresponde a {latest_label}.' "
                f"Luego USA get_series_data para obtener la cifra de {latest_label} y entrégala."
            )

    requested_year = _extract_requested_year(entities_ctx, question)
    if requested_year:
        expected_final = f"{requested_year}-12" if freq_code == "M" else f"{requested_year}-Q4"
        if latest_period.startswith(f"{requested_year}-") and latest_period != expected_final:
            latest_label = _natural_period_label(latest_period, freq_code)
            return (
                "REGLA ESTRICTA DE COBERTURA TEMPORAL: el año solicitado NO está cerrado. "
                f"Debes indicar: 'Los datos de {requested_year} aún no han sido publicados "
                f"en su totalidad según los datos de la Base de Datos Estadísticos. El último dato disponible "
                f"corresponde a {latest_label}.' "
                f"Luego USA get_series_data para obtener la cifra de {latest_label} y entrégala."
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
    elif re.search(r"\b[uú]ltimo\s+trimestre\b", text):
        relative_expr = "último trimestre"
        q = (today.month - 1) // 3 + 1
        target_period = f"{today.year - 1}-Q4" if q == 1 else f"{today.year}-Q{q - 1}"
        freq_code = "T"
    elif re.search(r"\b[uú]ltimo\s+mes\b", text):
        relative_expr = "último mes"
        target_period = f"{today.year - 1}-12" if today.month == 1 else f"{today.year}-{today.month - 1:02d}"
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
        f"('{relative_expr}' = {target_label}) sin datos disponibles. "
        f"Debes indicar: 'Los datos de {target_label} aún no han sido publicados según los datos de la Base de Datos Estadísticos. "
        f"El último dato disponible corresponde a {latest_label}.' "
        f"Luego USA get_series_data para obtener la cifra de {latest_label} y entrégala con su variación anual."
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
    if re.search(
        r"\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?(?:19|20)\d{2}\b",
        text,
    ):
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
        "ultimo trimestre",
        "último trimestre",
        "ultimo mes",
        "último mes",
    )
    return any(token in text for token in relative_tokens)


def _build_no_explicit_period_latest_instruction(
    question: str,
    entities_ctx: Dict[str, Any],
    observations: Dict[str, Any],
) -> Optional[str]:
    calc_mode = str(
        entities_ctx.get("calc_mode_cls")
        or entities_ctx.get("calc_mode")
        or ""
    ).strip().lower()
    # En contribution se usa una regla específica con rank_series; evitar conflicto
    # con la regla genérica que fuerza get_series_data sobre la serie agregada.
    if calc_mode == "contribution":
        return None

    # Si el usuario sí especificó período (aunque req_form_cls venga como latest),
    # nunca fuerces la regla de "último disponible".
    if _question_has_explicit_period(question, entities_ctx):
        return None

    # Evitar conflicto con consultas de serie original (PIB en pesos / precios corrientes / per cápita).
    # En estos casos otras instrucciones fuerzan value como dato principal; no debemos empujar yoy_pct aquí.
    text = str(question or "").lower()
    text_norm = unicodedata.normalize("NFKD", text)
    text_norm = "".join(ch for ch in text_norm if not unicodedata.combining(ch))
    price_ent = str(entities_ctx.get("price_ent") or entities_ctx.get("price") or "").strip().lower()
    indicator_ent = str(entities_ctx.get("indicator_ent") or "").strip().lower()
    activity_ent = str(entities_ctx.get("activity_ent") or "").strip().lower()
    is_original_value_query = (
        "pib en pesos" in text_norm
        or "precios corrientes" in text_norm
        or "precio corriente" in text_norm
        or "pib nominal" in text_norm
        or "pib per capita" in text_norm
        or "pib per cápita" in text
        or "a cuanto asciende el pib" in text_norm
        or price_ent == "co"
        or indicator_ent == "pib_per_capita"
        or activity_ent == "per_capita"
    )
    if is_original_value_query:
        return None

    freq_code = _resolve_requested_frequency(entities_ctx, observations)
    latest_available = observations.get("latest_available") or {}
    latest_period = str(latest_available.get(freq_code) or "").strip() if freq_code else ""
    if not latest_period:
        return None

    latest_label = latest_period if freq_code == "A" else _natural_period_label(latest_period, freq_code)
    freq_label = "mensual" if freq_code == "M" else "trimestral" if freq_code == "T" else "anual"
    return (
        "REGLA ESTRICTA DE PERIODO POR DEFECTO: "
        "si req_form_cls='latest', o si la pregunta no especifica fecha, "
        f"usa el ultimo periodo disponible ({latest_label}). "
        f"USA get_series_data para obtener la cifra de {latest_label} antes de responder. "
        f"En la INTRODUCCION incluye literalmente esta oración: 'La serie se reporta con frecuencia {freq_label}.' "
        "Cuando reportes yoy_pct, usa la expresión 'variación anual'. "
        "NO menciones falta de datos para meses/trimestres/años no solicitados y NO infieras "
        "automaticamente el mes/trimestre actual como periodo pedido. "
        "Mantén la estructura de 3 bloques: INTRODUCCIÓN (ancla indicador y período), DATOS, RECOMENDACIÓN."
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
        normalized_records: List[Dict[str, Any]] = []
        for row in records:
            if not isinstance(row, dict):
                continue
            clean_row = dict(row)
            if "YTYPCT" not in clean_row and "yoy_pct" in clean_row:
                clean_row["YTYPCT"] = clean_row.get("yoy_pct")
            clean_row.pop("yoy_pct", None)
            normalized_records.append(clean_row)

        if not normalized_records:
            return None

        fieldnames: List[str] = []
        seen = set()
        for row in normalized_records:
            for key in row.keys():
                if key in _CSV_EXCLUDE_COLS or key in seen:
                    continue
                seen.add(key)
                fieldnames.append(key)
        if not fieldnames:
            return None
        buf = io.StringIO()
        buf.write(f"# Nombre: {cuadro_name}\n")
        buf.write(f"# Serie: {short_title}\n")
        buf.write(f"# Serie ID: {series_id}\n")
        buf.write("# YTYPCT: alias BDE de variación porcentual interanual\n")
        buf.write("# pct: variación porcentual respecto al periodo anterior\n")
        buf.write("#\n")
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_records)
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


def _pick_series_frequency(series: Dict[str, Any], preferred_frequency: str = "") -> Optional[str]:
    data = series.get("data") or {}
    if not isinstance(data, dict) or not data:
        return None

    preferred = str(preferred_frequency or "").strip().upper()
    if preferred and preferred in data:
        return preferred

    for candidate in ("M", "T", "A"):
        if candidate in data:
            return candidate

    try:
        return str(next(iter(data.keys()))).upper()
    except Exception:
        return None


def _resolve_selected_series_context(
    observations: Dict[str, Any],
    entities_ctx: Dict[str, Any],
    selected_from_tools: Optional[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    if selected_from_tools:
        sid = str(selected_from_tools.get("series_id") or "").strip()
        freq = str(selected_from_tools.get("frequency") or "").strip().upper()
        if sid and freq:
            return {"series_id": sid, "frequency": freq}

    series_list = observations.get("series") or []
    if not isinstance(series_list, list) or not series_list:
        return None

    first_series = series_list[0] if isinstance(series_list[0], dict) else {}
    series_id = str(first_series.get("series_id") or "").strip()
    if not series_id:
        return None

    preferred_freq = _resolve_requested_frequency(entities_ctx, observations)
    frequency = _pick_series_frequency(first_series, preferred_freq)
    if not frequency:
        return None

    return {"series_id": series_id, "frequency": frequency}


def _find_series_records(
    observations: Dict[str, Any],
    series_id: str,
    frequency: str,
) -> Optional[Dict[str, Any]]:
    sid = str(series_id or "").strip()
    freq = str(frequency or "").strip().upper()
    if not sid or not freq:
        return None

    for series in observations.get("series") or []:
        if not isinstance(series, dict):
            continue
        if str(series.get("series_id") or "").strip() != sid:
            continue

        block = (series.get("data") or {}).get(freq) or {}
        raw_records = block.get("records") or []
        if not raw_records:
            return None

        # data_store local puede traer solo value; completar variaciones para CSV.
        lag = {"M": 12, "T": 4, "A": 1}.get(freq)
        enriched_records: List[Dict[str, Any]] = []
        value_series: List[Optional[float]] = []

        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            val_num: Optional[float]
            try:
                val = row.get("value")
                val_num = float(val) if val is not None else None
            except Exception:
                val_num = None
            value_series.append(val_num)
            enriched_records.append(row)

        for idx, row in enumerate(enriched_records):
            curr_val = value_series[idx]

            if "pct" not in row and "prev_period" not in row:
                pct_val: Optional[float] = None
                if idx > 0:
                    prev_val = value_series[idx - 1]
                    if curr_val is not None and prev_val not in (None, 0):
                        pct_val = (curr_val / prev_val - 1.0) * 100.0
                row["pct"] = pct_val

            if "yoy_pct" not in row and "yoy" not in row:
                yoy_val: Optional[float] = None
                if lag is not None and idx >= lag:
                    prev_yoy_val = value_series[idx - lag]
                    if curr_val is not None and prev_yoy_val not in (None, 0):
                        yoy_val = (curr_val / prev_yoy_val - 1.0) * 100.0
                row["yoy_pct"] = yoy_val

        formatted_records = [
            _add_display_fields(record)
            for record in enriched_records
            if isinstance(record, dict)
        ]
        if not formatted_records:
            return None

        return {
            "series_id": sid,
            "short_title": str(series.get("short_title") or ""),
            "frequency": freq,
            "records": formatted_records,
        }
    return None


def _build_full_history_csv_marker(
    observations: Dict[str, Any],
    selected_series_ctx: Optional[Dict[str, str]],
    cuadro_name: str = "",
) -> str:
    if not selected_series_ctx:
        return ""

    selected = _find_series_records(
        observations,
        selected_series_ctx.get("series_id", ""),
        selected_series_ctx.get("frequency", ""),
    )
    if not selected:
        return ""

    series_id = selected.get("series_id", "")
    short_title = selected.get("short_title", "")
    frequency = selected.get("frequency", "")
    records = selected.get("records", [])
    path = _export_series_csv(series_id, records, short_title=short_title, cuadro_name=cuadro_name)
    if not path:
        return ""

    safe_series_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(series_id)).strip("_") or "serie"
    filename = f"serie_{safe_series_id}_{frequency}.csv"
    return (
        f"##CSV_DOWNLOAD_START\n"
        f"path={path}\n"
        f"filename={filename}\n"
        f"title={short_title}\n"
        f"label=Descargar CSV\n"
        f"mimetype=text/csv\n"
        f"##CSV_DOWNLOAD_END"
    )


def _build_filtered_source_url(
    observations: Dict[str, Any],
    entities_ctx: Dict[str, Any],
    selected_series_ctx: Optional[Dict[str, str]],
) -> Optional[str]:
    if not selected_series_ctx:
        return None

    source_url = str(observations.get("source_url") or "").strip()
    if not source_url:
        return None

    series_id = selected_series_ctx.get("series_id", "")
    frequency = selected_series_ctx.get("frequency", "")
    series_data = _find_series_records(observations, series_id, frequency)
    if not series_data:
        return None

    period_values = entities_ctx.get("period_ent")
    period: Optional[List[Any]] = period_values if isinstance(period_values, list) else None
    req_form = entities_ctx.get("req_form_cls")
    calc_mode = (
        entities_ctx.get("calc_mode_cls")
        or (observations.get("classification") or {}).get("calc_mode")
    )
    calc_mode_for_url = str(calc_mode or "").strip().lower()

    # En consultas de niveles (PIB per cápita o PIB a precios corrientes),
    # el enlace debe conservar cálculo NONE para mostrar la serie original.
    indicator_ent = str(entities_ctx.get("indicator_ent") or "").strip().lower()
    activity_ent = str(entities_ctx.get("activity_ent") or "").strip().lower()
    price_ent = str(entities_ctx.get("price_ent") or entities_ctx.get("price") or "").strip().lower()
    question_text = str(entities_ctx.get("question") or "").strip().lower()

    is_per_capita_query = indicator_ent == "pib_per_capita" or activity_ent == "per_capita"
    is_current_prices_query = price_ent == "co" or (
        "precios corrientes" in question_text
        or "pib en pesos" in question_text
        or "a cuanto asciende el pib" in question_text
    )

    if calc_mode_for_url == "original" and (is_per_capita_query or is_current_prices_query):
        calc_mode_for_url = "none"

    frequency_for_url = {
        "M": "m",
        "T": "q",
        "A": "a",
    }.get(str(frequency).upper(), str(frequency or "").strip().lower())

    filtered = build_target_series_url(
        source_url=source_url,
        series_id=series_id,
        period=period,
        req_form=req_form,
        observations=series_data.get("records") or [],
        frequency=frequency_for_url,
        calc_mode=calc_mode_for_url,
    )
    return str(filtered or "").strip() or None


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
                    if key == "yoy_pct":
                        row["YTYPCT"] = value
                        if "YTYPCT" not in dynamic_seen:
                            dynamic_seen.add("YTYPCT")
                            dynamic_fields.append("YTYPCT")
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
    if "question" not in entities_ctx:
        entities_ctx = {**entities_ctx, "question": question}
    timing = payload.get("_timing") if isinstance(payload.get("_timing"), dict) else None
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
    seasonality_strict_instruction = _build_seasonality_strict_instruction(
        entities_ctx=entities_ctx,
        observations=observations,
    )
    if seasonality_strict_instruction:
        messages.append({"role": "system", "content": seasonality_strict_instruction})
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
    prevalidated_missing_specific_activity_instruction = _build_prevalidated_missing_specific_activity_instruction(
        entities_ctx=entities_ctx,
        observations=observations,
    )
    if prevalidated_missing_specific_activity_instruction:
        messages.append({"role": "system", "content": prevalidated_missing_specific_activity_instruction})
    missing_activity_instruction = _build_missing_activity_instruction(
        entities_ctx=entities_ctx,
        observations=observations,
    )
    if missing_activity_instruction:
        messages.append({"role": "system", "content": missing_activity_instruction})
    contribution_activity_focus_instruction = _build_contribution_activity_focus_instruction(
        question=question,
        entities_ctx=entities_ctx,
        observations=observations,
    )
    if contribution_activity_focus_instruction:
        messages.append({"role": "system", "content": contribution_activity_focus_instruction})
    economy_wording_instruction = _build_economy_wording_instruction(
        question=question,
        entities_ctx=entities_ctx,
    )
    if economy_wording_instruction:
        messages.append({"role": "system", "content": economy_wording_instruction})
    special_query_mapping_instruction = _build_special_query_mapping_instruction(
        question=question,
        entities_ctx=entities_ctx,
    )
    if special_query_mapping_instruction:
        messages.append({"role": "system", "content": special_query_mapping_instruction})
    specific_contribution_directness_instruction = _build_specific_contribution_directness_instruction(
        question=question,
    )
    if specific_contribution_directness_instruction:
        messages.append({"role": "system", "content": specific_contribution_directness_instruction})
    value_no_rescale_instruction = _build_value_no_rescale_instruction(
        question=question,
        entities_ctx=entities_ctx,
    )
    if value_no_rescale_instruction:
        messages.append({"role": "system", "content": value_no_rescale_instruction})
    original_series_force_instruction = _build_original_series_force_instruction(
        question=question,
        entities_ctx=entities_ctx,
    )
    if original_series_force_instruction:
        messages.append({"role": "system", "content": original_series_force_instruction})
    messages.append({"role": "system", "content": _build_recommendation_length_instruction()})
    annual_pib_instruction = _build_annual_pib_completeness_instruction(
        question=question,
        entities_ctx=entities_ctx,
        observations=observations,
    )
    if annual_pib_instruction:
        messages.append({"role": "system", "content": annual_pib_instruction})
    messages.append({"role": "user", "content": question})

    fetched_series: List[Dict[str, Any]] = []  # track get_series_data results
    selected_series_ctx: Optional[Dict[str, str]] = None
    tool_calls_elapsed_ms = 0.0

    try:
        for _ in range(max_tool_loops):
            loop_t0 = time.perf_counter()
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
                    tool_content = _sanitize_contribution_tool_result(tc["name"], fn_args, result)
                    logger.debug("[DATA_RESPONSE] tool=%s args=%s result_len=%d",
                                 tc["name"], tc["arguments"][:120], len(result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_content,
                    })
                    # Track fetched series data for CSV export
                    if tc["name"] == "get_series_data":
                        try:
                            parsed = json.loads(result)
                            if "records" in parsed:
                                fetched_series.append(parsed)
                                selected_series_ctx = {
                                    "series_id": str(parsed.get("series_id") or fn_args.get("series_id") or ""),
                                    "frequency": str(parsed.get("frequency") or fn_args.get("frequency") or "").upper(),
                                }
                        except Exception:
                            pass
                tool_calls_elapsed_ms += (time.perf_counter() - loop_t0) * 1000.0
            else:
                # Sin tool calls → respuesta final ya fue yielded via content_chunks
                post_t0 = time.perf_counter()
                final_series_ctx = _resolve_selected_series_context(
                    observations=observations,
                    entities_ctx=entities_ctx,
                    selected_from_tools=selected_series_ctx,
                )
                filtered_source_url = _build_filtered_source_url(
                    observations=observations,
                    entities_ctx=entities_ctx,
                    selected_series_ctx=final_series_ctx,
                )
                source_footer = _build_source_footer(
                    observations,
                    source_url_override=filtered_source_url,
                )
                if source_footer:
                    yield source_footer
                csv_block = ""
                if final_series_ctx:
                    csv_block = _build_full_history_csv_marker(
                        observations,
                        selected_series_ctx=final_series_ctx,
                        cuadro_name=str(observations.get("cuadro_name") or ""),
                    )
                if not csv_block and fetched_series:
                    csv_block = _build_csv_markers(
                        fetched_series,
                        cuadro_name=str(observations.get("cuadro_name") or ""),
                    )
                if not csv_block:
                    csv_block = _build_fallback_csv_marker(observations)
                if csv_block:
                    yield "\n" + csv_block
                if timing is not None:
                    timing["openai_tool_calls_ms"] = round(tool_calls_elapsed_ms, 2)
                    timing["post_response_blocks_ms"] = round((time.perf_counter() - post_t0) * 1000.0, 2)
                return
        else:
            # Se agotó el loop de herramientas
            yield "(Se alcanzó el límite de consultas internas. Intenta reformular la pregunta.)"
    except Exception:
        logger.exception("[DATA_RESPONSE] Error durante streaming con function calling")
        yield "Ocurrió un error al generar la respuesta."
