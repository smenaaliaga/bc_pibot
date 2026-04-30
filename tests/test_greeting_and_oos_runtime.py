"""Regresiones para el gate de saludo (RuleGreeting) y blocklist OOS
(Rule05_FueraDeAlcance) basadas en el run_detail_20260430.

- Saludo: "hola", "como estas", "hola, como estas", "buenos dias",
  variantes con/sin tildes.
- OOS blocklist: paridades, dólar, tipo de cambio, presidenta,
  consejeros, divisiones, etc. Estas preguntas llegan al servidor
  con indicator=imacec por DEFAULT del clasificador remoto. El gate
  debe bloquearlas igualmente.
"""

from __future__ import annotations

import pytest

from rules.post_classifier import _is_greeting, _is_out_of_scope


# ---------------------------------------------------------------------------
# Saludos: deben matchear
# ---------------------------------------------------------------------------

GREETING_OK = [
    "hola",
    "Hola",
    "HOLA",
    "holaa",
    "holi",
    "hola!",
    "hola.",
    "hola, ",
    "como estas",
    "cómo estás",
    "como estás",
    "cómo estas",
    "como va",
    "cómo te va",
    "que tal",
    "qué tal",
    "buenos dias",
    "buenos días",
    "buenas tardes",
    "buenas noches",
    "buen dia",
    "hola, como estas",
    "hola, cómo estás",
    "hola como estas?",
    "hola, como estas?",
    "hola que tal",
    "hola, que tal todo bien",
    "hey",
    "hi",
    "hello",
    "saludos",
    "todo bien?",
]


@pytest.mark.parametrize("q", GREETING_OK)
def test_greeting_matches(q: str):
    assert _is_greeting(q) is True, f"debió ser saludo: {q!r}"


# ---------------------------------------------------------------------------
# Saludos: NO deben matchear (preguntas reales)
# ---------------------------------------------------------------------------

NOT_GREETING = [
    "cual es el imacec",
    "como esta el pib",  # contiene 'como esta' pero también 'pib'
    "hola dame el pib",
    "buenos dias dame el imacec",
]


@pytest.mark.parametrize("q", NOT_GREETING)
def test_not_greeting(q: str):
    assert _is_greeting(q) is False, f"NO debió ser saludo: {q!r}"


# ---------------------------------------------------------------------------
# OOS: con indicator=imacec del clasificador (caso real run_detail_20260430).
# ---------------------------------------------------------------------------

CLASSIFIER_DEFAULT_NORM = {
    "indicator": ["imacec"],
    "seasonality": ["nsa"],
    "frequency": ["m"],
    "activity": [],
    "region": [],
    "investment": [],
    "price": [],
    "period": ["2026-03-01", "2026-03-31"],
}


OOS_RUNTIME_QUESTIONS = [
    "Cual es el valor de las paridades",
    "cuáles son las paridades hoy",
    "Cual es el valor de tipos de cambio nominal",
    "cuál es el tipo de cambio nominal",
    "cual es el valor del dolar observado",
    "cuál es el valor del dólar observado",
    "quien es la presidenta del banco central",
    "Quién es el presidente del banco central",
    "quienes son los consejeros del banco central",
    "cuál es la TPM",
    "cuál es la tasa de política monetaria",
    "cuál es la inflación de marzo",
    "cuál es el IPC",
    "cuál es el valor del euro",
]


@pytest.mark.parametrize("q", OOS_RUNTIME_QUESTIONS)
def test_oos_blocklist_overrides_classifier_default_imacec(q: str):
    assert _is_out_of_scope(
        question=q,
        current_norm=dict(CLASSIFIER_DEFAULT_NORM),
        prev_indicator=None,
        context_label="standalone",
    ) is True, f"debió bloquearse aún con indicator=imacec del clasificador: {q!r}"


@pytest.mark.parametrize("q", OOS_RUNTIME_QUESTIONS)
def test_oos_blocklist_overrides_followup_macro(q: str):
    """Aunque el turno previo haya sido sobre PIB/IMACEC, si el usuario
    pregunta explícitamente por dólar/paridades/persona, debe bloquearse."""
    assert _is_out_of_scope(
        question=q,
        current_norm=dict(CLASSIFIER_DEFAULT_NORM),
        prev_indicator="pib",
        context_label="followup",
    ) is True, f"blocklist debe ganar a followup macro: {q!r}"


# ---------------------------------------------------------------------------
# Sanidad: preguntas de PIB/IMACEC NO deben bloquearse.
# ---------------------------------------------------------------------------

IN_SCOPE = [
    "cual es el imacec",
    "cuál fue el pib del trimestre",
    "como creció la economia en los dos ultimos años",
    "variación del PIB anual",
]


@pytest.mark.parametrize("q", IN_SCOPE)
def test_in_scope_not_blocked(q: str):
    assert _is_out_of_scope(
        question=q,
        current_norm=dict(CLASSIFIER_DEFAULT_NORM),
        prev_indicator=None,
        context_label="standalone",
    ) is False, f"NO debió bloquearse: {q!r}"
