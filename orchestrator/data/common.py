from typing import List, Tuple
from datetime import datetime


def _normalize_first_day(d: datetime, frequency: str) -> datetime:
    if frequency == "m":
        return datetime(d.year, d.month, 1)
    if frequency == "q":
        month = ((d.month - 1) // 3) * 3 + 1
        return datetime(d.year, month, 1)
    if frequency == "a":
        return datetime(d.year, 1, 1)
    return d


def normalize_series_obs(obs: List[dict], frequency: str = "m") -> List[Tuple[datetime, float]]:
    out: List[Tuple[datetime, float]] = []
    for row in obs:
        try:
            d = datetime.strptime(row["date"], "%d-%m-%Y")
            d = _normalize_first_day(d, frequency)
            v = float(row["value"])
            out.append((d, v))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out
