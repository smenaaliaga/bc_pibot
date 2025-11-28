"""
answer.py
---------
Estandariza y centraliza las plantillas de respuesta del asistente:
- Instrucciones para respuestas metodológicas, de datos (primera fase) y genéricas.
- Banner/aviso de procesamiento de datos.

Este módulo NO altera la lógica de orquestación; solo provee piezas
reutilizables para que el orquestador las utilice.
"""

from enum import Enum
import json
from pathlib import Path
from typing import Dict, Optional


class ResponseType(str, Enum):
    """Tipos de respuesta del asistente."""

    METHODOLOGICAL = "METHODOLOGICAL"
    DATA = "DATA"
    GENERIC = "GENERIC"


# -----------------------------
# Utilidades locales (solo lectura de defaults)
# -----------------------------

def _load_default_series_codes() -> Dict[str, Dict[str, str]]:
    """Carga los códigos por defecto desde series/config_default.json.

    Retorna un dict con claves: IMACEC, PIB_TOTAL, PIB_REGIONAL. Cada una con:
      - cod_serie
      - nkname_esp
      - freq_por_defecto (si existe)
    En caso de fallo, usa valores de respaldo embebidos.
    """
    fallback = {
        "IMACEC": {
            "cod_serie": "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",
            "nkname_esp": "Imacec empalmado, serie original (índice 2018=100)",
            "freq_por_defecto": "M",
        },
        "PIB_TOTAL": {
            "cod_serie": "F032.PIB.V12.N.CLP.2018.Z.Z.0.T",
            "nkname_esp": "PIB total a precios del año anterior encadenado, trimestral, referencia 2018",
            "freq_por_defecto": "T",
        },
        "PIB_REGIONAL": {
            "cod_serie": "F035.PIB.V12.R.CLP.2018.CONT.Z.Z.TOTAL.0.T",
            "nkname_esp": "PIB total regionalizado, volumen a precios del año anterior encadenado, referencia 2018",
            "freq_por_defecto": "T",
        },
    }
    try:
        cfg_path = Path(__file__).resolve().parent / "series" / "config_default.json"
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        d = data.get("defaults", {})
        out = {
            "IMACEC": {
                "cod_serie": d.get("IMACEC", {}).get("cod_serie", fallback["IMACEC"]["cod_serie"]),
                "nkname_esp": d.get("IMACEC", {}).get("nkname_esp", fallback["IMACEC"]["nkname_esp"]),
                "freq_por_defecto": d.get("IMACEC", {}).get("freq_por_defecto", fallback["IMACEC"]["freq_por_defecto"]),
            },
            "PIB_TOTAL": {
                "cod_serie": d.get("PIB_TOTAL", {}).get("cod_serie", fallback["PIB_TOTAL"]["cod_serie"]),
                "nkname_esp": d.get("PIB_TOTAL", {}).get("nkname_esp", fallback["PIB_TOTAL"]["nkname_esp"]),
                "freq_por_defecto": d.get("PIB_TOTAL", {}).get("freq_por_defecto", fallback["PIB_TOTAL"]["freq_por_defecto"]),
            },
            "PIB_REGIONAL": {
                "cod_serie": d.get("PIB_REGIONAL", {}).get("cod_serie", fallback["PIB_REGIONAL"]["cod_serie"]),
                "nkname_esp": d.get("PIB_REGIONAL", {}).get("nkname_esp", fallback["PIB_REGIONAL"]["nkname_esp"]),
                "freq_por_defecto": d.get("PIB_REGIONAL", {}).get("freq_por_defecto", fallback["PIB_REGIONAL"]["freq_por_defecto"]),
            },
        }
        return out
    except Exception:
        return fallback


# -----------------------------
# Instrucciones por tipo
# -----------------------------

def get_methodological_instruction() -> str:
    """Instrucción estándar para respuestas metodológicas.

    Enfatiza definiciones, metodología y contexto. Evita tablas/datos.
    """
    return (
        "Responde en modo METHODOLOGICAL con una explicación breve en un máximo de dos párrafos. "
        "Céntrate en qué es el indicador, para qué sirve y, de forma muy resumida, cómo se calcula. "
        "Menciona como referencia al Banco Central de Chile y, si aplica, la página BDE del indicador "
        "(por ejemplo, IMACEC: https://si3.bcentral.cl/Siete/ES/Siete/Cuadro/CAP_CCNN/MN_CCNN76/CCNN2018_IMACEC_01_A). "
        "No incluyas tablas ni cifras concretas. Cierra con una o dos preguntas cortas que inviten a seguir "
        "la conversación (por ejemplo, si el usuario quiere ver datos recientes del indicador)."
    )


def get_data_first_phase_instruction() -> str:
    """Instrucción para la PRIMERA FASE (datos) del flujo.

    Objetivo: entregar SOLO una introducción breve experta y preparar la obtención
    real de datos (que el orquestador mostrará después). No generar tablas ficticias.

    Guía para el orquestador/LLM:
    - Eres un experto en indicadores económicos (PIB e IMACEC) del Banco Central de Chile.
    - Comienza con UN solo párrafo resumen (2–4 líneas) citando como fuente al Banco Central de Chile.
    - Explica qué mide el indicador y su utilidad interpretativa.
    - NO inventes valores ni muestres tablas de ejemplo con meses y "X%".
    - No repitas luego la misma información cuando aparezcan los datos reales.
    - Si el usuario menciona un año (ej. 2025) adelanta que se mostrará una comparación año anterior vs año actual.
    - No hables de proyecciones ni escenarios para años posteriores al consultado (por ejemplo, no menciones 2026 si la pregunta es sobre 2025).
    - Para traer datos reales el orquestador usará get_series.get_series(..., calc_type="PCT") y después
      build_year_comparison_table(...) cuando corresponda.
    - Serie por defecto (según dominio) tomada de series/config_default.json.
    """
    defaults = _load_default_series_codes()

    imacec = defaults.get("IMACEC", {})
    pib_total = defaults.get("PIB_TOTAL", {})
    pib_reg = defaults.get("PIB_REGIONAL", {})

    return (
        "Responde en modo DATA (fase 1). Produce SOLO un párrafo inicial (2–4 líneas) resumen experto del indicador, "
        "mencionando que la fuente es el Banco Central de Chile. Explica qué mide y su interpretación sin incluir enlaces. "
        "NO generes tablas ficticias ni ejemplos con meses y valores simbólicos. No inventes cifras ni menciones proyecciones "
        "sobre años posteriores al consultado. Si la pregunta incluye un año y se mostrarán datos reales después, NO escribas "
        "frases como 'no puedo proporcionar cifras'; simplemente anuncia que la tabla de datos reales aparecerá a continuación.\n\n"
        "Reglas de selección por defecto (si el usuario no especifica claramente):\n"
        f"- IMACEC → cod_serie={imacec.get('cod_serie')} (freq={imacec.get('freq_por_defecto','M')})\n"
        f"- PIB total → cod_serie={pib_total.get('cod_serie')} (freq={pib_total.get('freq_por_defecto','T')})\n"
        f"- PIB regional → cod_serie={pib_reg.get('cod_serie')} (freq={pib_reg.get('freq_por_defecto','T')})\n\n"
        "El orquestador obtendrá los datos reales usando get_series.get_series(calc_type=\"PCT\") y, si hay año específico, "
        "mostrará una tabla de comparación Año anterior vs Año actual (variación anual). Tú NO debes fabricarla aquí."
    )


def get_generic_instruction() -> str:
    """Instrucción estándar para respuestas genéricas no estrictamente de datos."""
    return (
        "La consulta no encaja claramente en IMACEC/PIB/PIB regional o no es "
        "estrictamente de datos. Responde de manera general explicando el tema "
        "económico relevante, sin inventar cifras."
    )


# -----------------------------
# Segunda fase (conclusión sobre los datos)
# -----------------------------

def get_data_second_phase_instruction() -> str:
    """Instrucción para la SEGUNDA FASE (conclusión sobre los datos).

    Para consultas de tipo DATA, en esta fase NO se debe volver a interpretar los
    datos ni redactar un análisis, ni siquiera breve. El objetivo es únicamente
    proponer preguntas de seguimiento para mantener el flujo conversacional,
    asumiendo que la tabla y la metadata ya fueron mostradas por el orquestador.

    Sugerencias de seguimiento requeridas (siempre referidas al período mostrado):
      1) ¿Quieres cambiar la frecuencia de la serie anterior?
      2) ¿Necesitas consultar por otra serie?
      3) Una pregunta adicional generada por el LLM, coherente con el contexto y
         el mismo período analizado (sin proyecciones a años futuros).
    """

    return (
        "Responde en modo DATA (fase 2). No agregues ninguna conclusión, resumen ni interpretación de los datos ya mostrados; "
        "no describas tendencias, crecimientos ni estabilidad y no menciones cifras nuevamente. Tras la tabla y los metadatos, "
        "escribe solo una frase breve, neutra y conversacional que introduzca las opciones de seguimiento, por ejemplo: "
        "'A continuación, te propongo algunas opciones para continuar con esta consulta:'. Esa frase no debe contener juicios "
        "sobre la evolución del indicador. Luego limítate exclusivamente a proponer preguntas de seguimiento sobre el mismo "
        "período de datos. No sugieras proyecciones ni análisis de años futuros (por ejemplo, no propongas hablar de 2026 si la "
        "consulta fue sobre 2025). Incluye exactamente estas tres preguntas al final, en este orden y sin más texto después de "
        "ellas: 1) ¿Quieres cambiar la frecuencia de la serie anterior? 2) ¿Necesitas consultar por otra serie? 3) Propón una "
        "pregunta adicional relevante generada por ti en base al contexto y al mismo año consultado."
    )


# -----------------------------
# Banners / Avisos
# -----------------------------

def get_processing_banner() -> str:
    """Mensaje estandarizado mostrado entre fases al procesar datos."""
    return (
        "\n\n---\n\nProcesando los datos solicitados, esto puede tomar unos segundos...\n\n"
    )
# -----------------------------
# Footer descarga CSV
# -----------------------------

def get_csv_download_footer(csv_path: Optional[str]) -> str:
    """Devuelve una línea para ofrecer descarga de CSV.

    Para desactivar globalmente, comenta esta función o retorna siempre "".
    El orquestador tolera la ausencia de esta función con un fallback.
    """
    if not csv_path:
        return ""
    return f"Archivo CSV disponible: {csv_path}"

