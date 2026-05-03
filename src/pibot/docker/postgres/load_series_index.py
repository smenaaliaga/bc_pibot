import json
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any, List
import unicodedata
import re

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None  # type: ignore


def _load_from_json() -> Dict[str, Dict[str, Any]]:
    """
    Deprecated: JSON fallback removed. Return empty so callers rely on Postgres.
    """
    return {}


def _load_from_db() -> Dict[str, Dict[str, Any]]:
    """Attempt to load series metadata from Postgres (series_metadata table)."""
    dsn = os.getenv("PG_DSN")
    if not dsn or not psycopg:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT cod_serie, freq, desc_serie_esp, nkname_esp, cap_esp, cod_capitulo, cod_cuadro, desc_cuad_esp, url, metadata_unidad, metadata_fuente, metadata_rezago, metadata_base, metadata_metodologia, metadata_concep_est, metadata_recom_uso FROM series_metadata"
                )
                for row in cur.fetchall() or []:
                    (
                        code,
                        freq,
                        desc_serie_esp,
                        nkname_esp,
                        cap_esp,
                        cod_capitulo,
                        cod_cuadro,
                        desc_cuad_esp,
                        url,
                        metadata_unidad,
                        metadata_fuente,
                        metadata_rezago,
                        metadata_base,
                        metadata_metodologia,
                        metadata_concep_est,
                        metadata_recom_uso,
                    ) = row
                    # Skip unwanted PIB 2013 series
                    if re.search(r"^F035\.PIB\..*\.2013\..*", code or ""):
                        continue
                    out[code] = {
                        "COD_SERIE": code,
                        "FREQ": freq,
                        "DESC_SERIE_ESP": desc_serie_esp,
                        "NKNAME_ESP": nkname_esp,
                        "CAP_ESP": cap_esp,
                        "COD_CAPITULO": cod_capitulo,
                        "COD_CUADRO": cod_cuadro,
                        "DESC_CUAD_ESP": desc_cuad_esp,
                        "URL": url,
                        "METADATA_UNIDAD": metadata_unidad,
                        "METADATA_FUENTE": metadata_fuente,
                        "METADATA_REZAGO": metadata_rezago,
                        "METADATA_BASE": metadata_base,
                        "METADATA_METODOLOGIA": metadata_metodologia,
                        "METADATA_CONCEP_EST": metadata_concep_est,
                        "METADATA_RECOM_USO": metadata_recom_uso,
                    }
    except Exception:
        return {}
    return out


@lru_cache(maxsize=1)
def _load_index() -> Dict[str, Dict[str, Any]]:
    """
    Load the series index from Postgres (table series_metadata). JSON fallback
    intentionally disabled; ensure PG_DSN is set and series_metadata is loaded.
    """
    db_idx = _load_from_db()
    if db_idx:
        return db_idx
    return _load_from_json()


def get_series_metadata(code: str) -> Optional[Dict[str, Any]]:
    """Return metadata for a given series code (COD_SERIE)."""
    if not code:
        return None
    idx = _load_index()
    return idx.get(code)


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    return t.lower()


def _tokenize(text: str) -> set[str]:
    """Simple tokenization: alfanumérico, sin tildes, minúsculas."""
    norm = _normalize(text)
    return set(re.findall(r"[a-z0-9]+", norm))


def search_series(text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Accent-insensitive substring + token-overlap search over NKNAME_ESP/DESC_SERIE_ESP.
    """
    q_norm = _normalize(text or "")
    if not q_norm:
        return []
    q_tokens = _tokenize(q_norm)
    idx = _load_index()
    scored: List[tuple[int, Dict[str, Any]]] = []
    for code, meta in idx.items():
        nk_raw = meta.get("NKNAME_ESP") or ""
        desc_raw = meta.get("DESC_SERIE_ESP") or ""
        nk = _normalize(nk_raw)
        desc = _normalize(desc_raw)
        nk_tokens = _tokenize(nk_raw)
        desc_tokens = _tokenize(desc_raw)
        score = 0
        if q_norm in nk or q_norm in desc:
            score += 5
        overlap_nk = len(q_tokens & nk_tokens)
        overlap_desc = len(q_tokens & desc_tokens)
        score += overlap_nk + overlap_desc
        if q_tokens and q_tokens.issubset(nk_tokens | desc_tokens):
            score += 2
        if score > 0:
            scored.append((score, {"code": code, **meta}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]
