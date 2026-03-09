"""Envía un payload directo a flow_data.stream_data_flow para hacer streaming
de una respuesta económica usando el LLM GPT.

El payload incluye:
  - question: la pregunta del usuario
  - classification: entidades clasificadas (indicator, frequency, period, req_form, etc.)
  - observations (result): datos ya procesados listos para renderizar
  - series, family_name, family_series, family_source_url: metadatos de la serie

Uso:
    python tools/debug_data_response_stream.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración base
# ---------------------------------------------------------------------------
os.environ.setdefault("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator.data.flow_data import stream_data_flow  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payload de ejemplo — ajústalo a tu caso de uso
# ---------------------------------------------------------------------------

def build_sample_payload() -> dict:
    """Construye un payload de ejemplo para streaming de respuesta.

    Estructura esperada por ``stream_data_flow``:
        classification : dict  → entidades normalizadas + clasificaciones
        series         : str   → ID de la serie BDE objetivo
        family_name    : str   → nombre de la familia de series
        family_series  : list  → lista de series de la familia (id, title)
        family_source_url : str → URL base de la familia en la BDE
        result         : list  → observaciones ya procesadas (date, value, yoy, prev_period, title)
        all_series_data: list | None → datos de todas las series de la familia (para contribución)
        question       : str   → pregunta original del usuario
        intro_llm_temperature : float → temperatura del LLM para la introducción
    """

    # --- Classification ---------------------------------------------------
    # Simula lo que produce el clasificador + normalizador del grafo.
    classification = {
        # Entidades normalizadas
        "indicator_ent": "imacec",
        "seasonality_ent": "nsa",
        "frequency_ent": "M",
        "activity_ent": "total",
        "region_ent": None,
        "investment_ent": None,
        "period_ent": ["2025-12-01"],        # lista de fechas ISO (punto o rango)
        # Clasificaciones del agente
        "calc_mode_cls": "variation",         # variation | level | contribution
        "activity_cls": "none",               # none | general | specific
        "region_cls": "none",
        "investment_cls": "none",
        "req_form_cls": "latest",             # latest | point | range | specific_point
        # Campos derivados por reglas de negocio (opcionales)
        "price": "enc",
        "hist": None,
    }

    # --- Observations (result) -------------------------------------------
    # Observaciones ya procesadas — cada fila es un período.
    result = [
        {
            "date": "2025-12-01",
            "value": 146.3,
            "yoy": 4.5,              # variación interanual (%)
            "prev_period": 0.8,       # variación respecto al período anterior (%)
            "title": "Imacec empalmado",
        },
    ]

    # --- Metadatos de la familia -----------------------------------------
    series_id = "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M"
    family_name = "Imacec empalmado"
    family_series = [
        {"id": "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M", "title": "Imacec empalmado"},
    ]
    family_source_url = "https://si3.bcentral.cl/Siete/ES/Siete/Cuadro/CAP_ESTADIST_MACRO/MN_EST_MACRO_IV/PEM_IMACEC/637797"

    return {
        "question": "¿Cuál fue el Imacec de diciembre 2025?",
        "classification": classification,
        "series": series_id,
        "family_name": family_name,
        "family_series": family_series,
        "family_source_url": family_source_url,
        "result": result,
        "all_series_data": None,          # None para consulta simple sin desglose
        "intro_llm_temperature": 0.7,
    }


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------

def main() -> None:
    payload = build_sample_payload()

    logger.info("=== Payload ===")
    for key, value in payload.items():
        if key == "result":
            logger.info("  %s: %d observaciones", key, len(value) if isinstance(value, list) else 0)
        else:
            logger.info("  %s: %s", key, value)

    logger.info("=== Streaming respuesta (GPT) ===\n")

    full_response = []
    for chunk in stream_data_flow(payload, session_id="debug-session-001"):
        text = str(chunk)
        if text:
            print(text, end="", flush=True)
            full_response.append(text)

    print()  # salto de línea final
    logger.info("\n=== Respuesta completa (%d caracteres) ===", len("".join(full_response)))


if __name__ == "__main__":
    main()
