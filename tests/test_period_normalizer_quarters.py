import pytest
from orchestrator.utils.period_normalizer import standardize_imacec_time_ref
from datetime import date

test_cases = [
    ("1T 2025", {"granularity": "quarter", "period_key": "2025-Q1"}),
    ("primer trimestre 2025", {"granularity": "quarter", "period_key": "2025-Q1"}),
    ("1er trimestre 2025", {"granularity": "quarter", "period_key": "2025-Q1"}),
    ("I trimestre 2025", {"granularity": "quarter", "period_key": "2025-Q1"}),
    ("1 trimestre 2025", {"granularity": "quarter", "period_key": "2025-Q1"}),
    ("3er trimestre 2024", {"granularity": "quarter", "period_key": "2024-Q3"}),
    ("1Q 2025", {"granularity": "quarter", "period_key": "2025-Q1"}),
    ("I Trimestre 2025", {"granularity": "quarter", "period_key": "2025-Q1"}),
    ("primer trimstre 2025", {"granularity": "quarter", "period_key": "2025-Q1"}),
    ("3er trimestr 2024", {"granularity": "quarter", "period_key": "2024-Q3"}),
]

def _assert_period(result, expected):
    assert result is not None, f"No se detectó periodo. Resultado: {result}"
    for k, v in expected.items():
        if result[k] != v:
            print(f"\n[DEBUG TEST] Resultado real: {result}\n[DEBUG TEST] Esperado: {expected}\n")
        assert result[k] == v, f"Esperado {k}={v}, obtenido {result[k]}"

def test_trimestrales():
    for texto, esperado in test_cases:
        res = standardize_imacec_time_ref(texto)
        _assert_period(res, esperado)
