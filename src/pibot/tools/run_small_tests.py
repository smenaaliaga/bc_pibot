import os
import sys
from typing import List

import pytest

# Ensure repository root is on sys.path when executing directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TESTS: List[str] = [
    "tests/test_followup_routing.py::test_followup_value_region_backfills_indicator_and_time_fields",
    "tests/test_prompt_registry.py::test_data_summary_prompt_is_parameterized",
]

failures: List[str] = []

for test_path in TESTS:
    print(f"Running pytest {test_path}...")
    result = pytest.main([test_path, "-q"])
    if result == 0:
        print("  OK")
    else:
        print("  FAIL")
        failures.append(test_path)

print("\nSummary:\n")
if not failures:
    print("All tests passed")
else:
    print(f"{len(failures)} test(s) failed:")
    for test_path in failures:
        print(f" - {test_path}")
    raise SystemExit(1)
