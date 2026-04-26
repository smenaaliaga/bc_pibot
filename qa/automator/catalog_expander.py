"""Lee el catálogo de cuadros para obtener actividades, regiones e indicadores reales."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "orchestrator" / "catalog" / "catalog.json"


def _load_catalog() -> Dict[str, Any]:
    if not CATALOG_PATH.exists():
        return {}
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_catalog_facts() -> Dict[str, List[str]]:
    cat = _load_catalog()
    indicators: Set[str] = set()
    activities: Set[str] = set()
    regions: Set[str] = set()
    cuadros = cat if isinstance(cat, dict) else {}
    for _, entry in cuadros.items():
        if not isinstance(entry, dict):
            continue
        cls = entry.get("classification") or {}
        ind = cls.get("indicator")
        if isinstance(ind, str):
            indicators.add(ind.strip().lower())
        for s in entry.get("series") or []:
            if not isinstance(s, dict):
                continue
            scl = s.get("classification") or {}
            act = scl.get("activity")
            reg = scl.get("region")
            if isinstance(act, str) and act.strip():
                activities.add(act.strip().lower())
            if isinstance(reg, str) and reg.strip():
                regions.add(reg.strip().lower())
    return {
        "indicators": sorted(i for i in indicators if i),
        "activities": sorted(a for a in activities if a),
        "regions": sorted(r for r in regions if r),
    }
