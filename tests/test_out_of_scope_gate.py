"""Harness para el gate 'Solo PIB / IMACEC'.

Valida tres invariantes:
  1. OUT_OF_SCOPE_BLOCK: preguntas fuera de alcance → _is_out_of_scope==True
     y el nodo scope_block emite el guardrail + sugerencia.
  2. IN_SCOPE_PASS: preguntas sobre PIB/IMACEC (incluyendo metodológicas y
     calendario) → _is_out_of_scope==False.
  3. FOLLOWUP_PASS: turnos cortos sin keyword pero con indicador previo
     PIB/IMACEC → _is_out_of_scope==False.

Adicional:
  - Sanitizer del segundo párrafo: rechaza fugas (tokens prohibidos),
    acepta sugerencias válidas y trunca largos.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from orchestrator.graph.nodes.ingest import _is_out_of_scope
from orchestrator.graph.nodes.llm import (
    OUT_OF_SCOPE_FALLBACK_SUGGESTION,
    OUT_OF_SCOPE_GUARDRAIL,
    _sanitize_scope_suggestion,
    make_scope_block_node,
)


# ---------------------------------------------------------------------------
# Grupo OUT_OF_SCOPE_BLOCK
# ---------------------------------------------------------------------------

OUT_OF_SCOPE_QUESTIONS: List[str] = [
    # Ejemplos del requerimiento
    "quien es la presidenta del banco central",
    "Quien es la presidenta del banco central",
    "cual es el valor del dolar observado",
    "Cuál es el valor del dólar observado",
    "Quienes son los consejeros del banco central",
    "quiénes son los consejeros del banco central",
    "Cual es el valor de tipos de cambio nominal",
    "cuál es el tipo de cambio nominal hoy",
    "Cual es el valor de las paridades",
    "cuáles son las paridades",
    "Cuales son las divisiones del banco central",
    "cuáles son las divisiones del banco central de chile",
    "Quien es Hernán Fernandez",
    "quién es Hernán Fernández",
    # Variantes adicionales
    "cuál es la TPM actual",
    "cuál es la tasa de política monetaria",
    "cuál es el IPC de marzo",
    "cuánto es la inflación",
    "cuál es el valor de la UF",
    "cuál es el valor de la UTM",
    "cuánto vale el euro hoy",
    "cuánto vale el yuan",
    "cuánto vale el yen",
    "precio del bitcoin",
    "qué es una criptomoneda",
    "cuál es el horario del banco central",
    "dónde queda el banco central",
    "quién dirige el banco central",
    "cuál es el correo del banco central",
    "qué hace el banco central",
    "cuántos empleados tiene el banco central",
    "cuál es el sueldo del presidente del banco central",
    "qué opina el banco central del gobierno",
    "cuál es la receta del pastel de choclo",
    "quién ganó el partido de Chile",
    "qué tiempo hará mañana",
    "recomiéndame un libro",
    "qué hora es",
    "cómo te llamas",
    "hola, cómo estás",
    "ayúdame con un código en python",
]

assert len(OUT_OF_SCOPE_QUESTIONS) >= 30


# ---------------------------------------------------------------------------
# Grupo IN_SCOPE_PASS (deben pasar como antes)
# ---------------------------------------------------------------------------

IN_SCOPE_QUESTIONS: List[str] = [
    # PIB básicas
    "cuál es el PIB de Chile en 2024",
    "cuánto creció el PIB el último trimestre",
    "PIB del 4to trimestre de 2025",
    "variación trimestral del PIB",
    "variación interanual del PIB",
    "PIB desestacionalizado del 3er trimestre",
    "PIB regional de Antofagasta",
    "PIB anual 2023",
    "valor del producto interno bruto",
    "cuál es el PIB nominal",
    "PIB a precios corrientes",
    "PIB a precios encadenados",
    # IMACEC básicas
    "cuál es el IMACEC más reciente",
    "IMACEC de marzo 2026",
    "variación mensual del IMACEC",
    "variación interanual del IMACEC",
    "IMACEC desestacionalizado",
    "IMACEC minero",
    "IMACEC no minero",
    "IMACEC de servicios",
    "IMACEC de bienes",
    # Cuentas nacionales
    "cuentas nacionales del 2024",
    "cuenta nacional trimestral",
    # Calendario (siempre debe pasar a RAG, no out_of_scope)
    "cuándo se publica el próximo IMACEC",
    "cuándo sale el PIB del 1er trimestre",
    "calendario de publicaciones",
    "próxima publicación del IMACEC",
    "fecha de publicación del PIB",
    # Metodológicas (irían a rag pero no a out_of_scope)
    "cómo se calcula el PIB encadenado",
    "qué es el PIB desestacionalizado",
    "metodología del IMACEC",
    "qué significa serie desestacionalizada del PIB",
    # Variación / contribución
    "contribución de la minería al PIB",
    "cuánto pesa el consumo en el PIB",
    "participación del sector servicios en el PIB",
    # Históricas
    "PIB de 1985",
    "IMACEC histórico de 1990",
]

assert len(IN_SCOPE_QUESTIONS) >= 30


# ---------------------------------------------------------------------------
# Grupo FOLLOWUP_PASS (sin keyword pero con indicador previo)
# ---------------------------------------------------------------------------

FOLLOWUP_QUESTIONS: List[str] = [
    "y desestacionalizado",
    "¿y en 2024?",
    "¿y desestacionalizado?",
    "¿y en Antofagasta?",
    "y la variación trimestral",
    "y el último valor",
    "y respecto al período anterior",
    "y en términos nominales",
    "muéstralo a precios corrientes",
    "ahora con la serie original",
    "muéstrame el rango 2020 a 2024",
    "y por sector",
    "y por región",
    "y en encadenado",
    "ahora con datos del 4to trimestre",
]

assert len(FOLLOWUP_QUESTIONS) >= 10


# ---------------------------------------------------------------------------
# Tests OUT_OF_SCOPE_BLOCK
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", OUT_OF_SCOPE_QUESTIONS)
def test_out_of_scope_predicate_blocks(question: str):
    assert _is_out_of_scope(
        question=question,
        current_norm={},  # NER no normaliza nada
        prev_indicator=None,
        context_label="standalone",
    ) is True, f"Debió bloquearse: {question!r}"


@pytest.mark.parametrize("question", OUT_OF_SCOPE_QUESTIONS[:10])
def test_scope_block_node_emits_guardrail_and_suggestion(question: str):
    node = make_scope_block_node(llm_adapter=None)  # forzamos fallback determinístico
    out = node({"question": question})
    assert out["route_decision"] == "out_of_scope"
    assert out["output"].startswith(OUT_OF_SCOPE_GUARDRAIL)
    assert "\n\n" in out["output"]
    body = out["output"].split("\n\n", 1)[1]
    assert ("PIB" in body) or ("IMACEC" in body)
    # Sin tokens prohibidos
    for forbidden in ("dólar", "dolar", "tipo de cambio", "TPM", "IPC", "presidenta"):
        assert forbidden.lower() not in body.lower(), (
            f"Token prohibido '{forbidden}' filtrado en sugerencia: {body!r}"
        )


def test_scope_block_node_uses_fallback_when_no_adapter():
    node = make_scope_block_node(llm_adapter=None)
    out = node({"question": "cuál es el dólar"})
    assert OUT_OF_SCOPE_FALLBACK_SUGGESTION in out["output"]


# ---------------------------------------------------------------------------
# Tests IN_SCOPE_PASS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", IN_SCOPE_QUESTIONS)
def test_in_scope_predicate_does_not_block(question: str):
    assert _is_out_of_scope(
        question=question,
        current_norm={},
        prev_indicator=None,
        context_label="standalone",
    ) is False, f"NO debió bloquearse: {question!r}"


@pytest.mark.parametrize(
    "norm_key,value",
    [
        ("indicator", "pib"),
        ("indicator", "imacec"),
        ("activity", "mineria"),
        ("activity", "servicios"),
        ("region", "antofagasta"),
        ("investment", "consumo"),
    ],
)
def test_in_scope_when_ner_normalizes_entity(norm_key: str, value: str):
    """Si el normalizador detectó una entidad macro, no se bloquea aunque la
    pregunta sin keyword salga del allowlist directo."""
    assert _is_out_of_scope(
        question="cuánto fue en el último período",
        current_norm={norm_key: value},
        prev_indicator=None,
        context_label="standalone",
    ) is False


# ---------------------------------------------------------------------------
# Tests FOLLOWUP_PASS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", FOLLOWUP_QUESTIONS)
@pytest.mark.parametrize("prev_indicator", ["pib", "imacec"])
def test_followup_with_prior_indicator_passes(question: str, prev_indicator: str):
    assert _is_out_of_scope(
        question=question,
        current_norm={},
        prev_indicator=prev_indicator,
        context_label="followup",
    ) is False


def test_followup_without_prior_indicator_blocks():
    """Followup pero sin indicador previo PIB/IMACEC → se bloquea."""
    assert _is_out_of_scope(
        question="y eso cómo se calcula",
        current_norm={},
        prev_indicator=None,
        context_label="followup",
    ) is True


# ---------------------------------------------------------------------------
# Tests del sanitizer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_empty",
    [
        ("", True),
        ("   ", True),
        ("Puedes consultar el dólar y el PIB.", True),  # blocklist
        ("Pregunta por la TPM o el PIB.", True),
        ("Información sobre el IPC.", True),
        ("Texto general sin temas relevantes.", True),  # no menciona PIB/IMACEC
        ("- Pregunta por el PIB.", True),  # markdown bullet
        # Aceptados
        ("Puedes consultar la variación trimestral del PIB.", False),
        ("¿Te interesa el IMACEC mensual o el PIB anual?", False),
        ("Pregunta por el PIB desestacionalizado del último trimestre.", False),
    ],
)
def test_sanitizer_behavior(raw: str, expected_empty: bool):
    out = _sanitize_scope_suggestion(raw)
    if expected_empty:
        assert out == "", f"Debió rechazarse: {raw!r} -> {out!r}"
    else:
        assert out, f"Debió aceptarse: {raw!r}"
        # No tokens prohibidos
        for forbidden in ("dólar", "dolar", "TPM", "IPC", "presidenta"):
            assert forbidden.lower() not in out.lower()


def test_sanitizer_strips_guardrail_echo():
    raw = (
        "Esta IA responde solamente consultas del PIB e IMACEC. "
        "Pregunta por la variación mensual del IMACEC."
    )
    out = _sanitize_scope_suggestion(raw)
    assert out
    assert "Esta IA responde" not in out
    assert "IMACEC" in out


def test_sanitizer_truncates_long_text():
    raw = ("Puedes consultar el PIB. " * 40).strip()
    out = _sanitize_scope_suggestion(raw)
    assert out
    assert len(out) <= 320
