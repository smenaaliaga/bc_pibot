from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta


def _find_by_date(series: List[Tuple[datetime, float]], target: datetime) -> Optional[float]:
    for d, v in series:
        if d == target:
            return v
    return None


def yoy(series: List[Tuple[datetime, float]], d: datetime, v: float) -> Optional[float]:
    prev = datetime(d.year - 1, d.month, d.day)
    v_prev = _find_by_date(series, prev)
    if v_prev is None or v_prev == 0:
        return None
    return (v / v_prev - 1.0) * 100.0


def prev_period(series: List[Tuple[datetime, float]], d: datetime, v: float) -> Optional[float]:
    # Find prior observation chronologically
    prev_v = None
    prev_d = None
    for i, (di, vi) in enumerate(series):
        if di == d and i > 0:
            prev_d, prev_v = series[i - 1]
            break
    if prev_v is None or prev_v == 0:
        return None
    return (v / prev_v - 1.0) * 100.0


def compute_variations_for_range(series: List[Tuple[datetime, float]], calc_mode: str, 
                                 start: datetime, end: datetime) -> List[Dict]:
    out: List[Dict] = []
    for d, v in series:
        if start <= d <= end:
            row: Dict = {"date": d.strftime("%d-%m-%Y"), "value": v}
            if calc_mode == "yoy":
                var = yoy(series, d, v)
                if var is not None:
                    row["yoy"] = var
            elif calc_mode == "prev_period":
                var = prev_period(series, d, v)
                if var is not None:
                    row["prev_period"] = var
            out.append(row)
    return out
