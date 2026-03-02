from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGIC_PATH = ROOT / "docker" / "intent_api" / "intent_api" / "logic.py"


spec = importlib.util.spec_from_file_location("local_intent_logic", LOGIC_PATH)
assert spec and spec.loader
logic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logic)


classify_intent = logic.classify_intent


def test_methodology_query_routes_to_method():
    result = classify_intent("como se calcula el pib")
    assert result["macro"]["label"] == 1
    assert result["intent"]["label"] == "method"


def test_factors_query_routes_to_method():
    result = classify_intent("que factores influyen en el calculo del pib")
    assert result["macro"]["label"] == 1
    assert result["intent"]["label"] == "method"


def test_value_query_remains_value():
    result = classify_intent("cual es el valor del pib")
    assert result["macro"]["label"] == 1
    assert result["intent"]["label"] == "value"
