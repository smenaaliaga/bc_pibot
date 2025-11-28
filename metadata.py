# filepath: /Users/hernanfernandez/Documents/01 Workspace/Notebooks/pibot/metadata.py
"""
metadata.py
-----------
Funciones utilitarias para recuperar metadatos de una serie desde
`catalog/series_catalog.json` de forma determinista por `series_id`.

Campos devueltos:
- code           : series_id
- title          : title
- default_frequency : default_frequency
- unit           : unit
- source_url     : source_url
- metodologia    : notes.metodologia (si existe)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

_CATALOG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "catalog", "series_catalog.json")


def get_series_metadata(series_id: str) -> Optional[Dict[str, Any]]:
    """Devuelve metadatos de la serie por su `series_id`.
    Retorna None si no se encuentra.
    """
    if not series_id:
        return None
    try:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as f:
            catalog = json.load(f)
    except Exception:
        return None
    entry = catalog.get(series_id)
    if not entry:
        return None
    notes = entry.get("notes") or {}
    return {
        "code": series_id,
        "title": entry.get("title"),
        "default_frequency": entry.get("default_frequency"),
        "unit": entry.get("unit"),
        "source_url": entry.get("source_url"),
        "metodologia": notes.get("metodologia"),
    }
