#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
warm_redis_cache.py
-------------------
Pre-calienta el caché Redis con todas las series del catálogo del bot.

Estrategia:
  - Lee orchestrator/catalog/catalog.json (941 series en ~75 familias).
  - Para cada serie, llama get_series_api_rest_bcch(...) que:
    * Obtiene el historial completo desde la API BCCh.
    * Almacena en Redis con clave auto:auto:{freq}:sum.
  - Consultas posteriores del bot son instantáneas (cache hit).

Uso:
  python tools/warm_redis_cache.py [--workers N] [--force] [--dry-run]

  --workers N   Hilos paralelos (default: 8, max recomendado: 10).
  --force       Refresca incluso series ya cacheadas.
  --dry-run     Solo lista las series sin llamar a la API.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# sys.path: permite ejecutar desde raíz del proyecto
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("warm_redis_cache")

CATALOG_PATH = os.path.join(ROOT, "orchestrator", "catalog", "catalog.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_target_frequency(series_id: str) -> Optional[str]:
    """Infiere target_frequency desde el sufijo del series_id."""
    if series_id.endswith(".M"):
        return "M"
    if series_id.endswith(".T"):
        return "Q"
    if series_id.endswith(".A"):
        return "A"
    return None


def _collect_series(catalog: dict) -> List[Tuple[str, Optional[str], str]]:
    """Retorna lista de (series_id, target_frequency, family_name) deduplicada."""
    seen = set()
    result = []
    for family_name, family_data in catalog.items():
        if not isinstance(family_data, dict):
            continue
        for s in family_data.get("series", []):
            sid = s.get("id") or s.get("series_id")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            freq = _infer_target_frequency(sid)
            result.append((sid, freq, family_name))
    return result


def _is_already_cached(redis_client, series_id: str, freq: Optional[str], agg: str) -> bool:
    """Verifica si la clave base (sin rango de fechas) ya existe en Redis."""
    freq_str = (freq or "orig").upper()
    key = f"bcch:series:{series_id}:auto:auto:{freq_str}:{agg.lower()}"
    try:
        return bool(redis_client.exists(key))
    except Exception:
        return False


def _warm_one(
    series_id: str,
    target_frequency: Optional[str],
    agg: str,
    force: bool,
    dry_run: bool,
) -> Tuple[str, str, Optional[str]]:
    """
    Carga una serie en Redis.
    Returns: (series_id, status, error_msg)
      status in {"ok", "skip", "error", "dry_run"}
    """
    if dry_run:
        return series_id, "dry_run", None

    # Import aquí para que funcione en cada thread
    from orchestrator.data.get_data_serie import (
        get_series_api_rest_bcch,
        _ensure_redis_client,
    )

    if not force:
        client = _ensure_redis_client()
        if client and _is_already_cached(client, series_id, target_frequency, agg):
            return series_id, "skip", None

    try:
        result = get_series_api_rest_bcch(
            series_id=series_id,
            target_frequency=target_frequency,
            agg=agg,
        )
        if result and result.get("observations"):
            return series_id, "ok", None
        return series_id, "ok_empty", None
    except Exception as e:
        return series_id, "error", str(e)


# ---------------------------------------------------------------------------
# Startup helper (se llama desde main.py en background thread)
# ---------------------------------------------------------------------------

def warm_cache_background(workers: int = 8, force: bool = False) -> None:
    """Corre el precalentamiento en un hilo de background (sin bloquear).
    Diseñado para llamarse desde main.py al arrancar la aplicación.
    """
    import threading

    def _run():
        try:
            main(workers=workers, force=force, dry_run=False, quiet=True)
        except Exception as e:
            logger.warning("[warm_redis_cache] Error en precalentamiento background: %s", e)

    t = threading.Thread(target=_run, name="warm-redis-cache", daemon=True)
    t.start()
    logger.info("[warm_redis_cache] Precalentamiento Redis iniciado en background | workers=%d", workers)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    workers: int = 8,
    force: bool = False,
    dry_run: bool = False,
    quiet: bool = False,
) -> Dict[str, int]:
    t0 = time.monotonic()

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    series_list = _collect_series(catalog)
    total = len(series_list)

    if not quiet:
        logger.info("Catálogo cargado | familias=%d | series=%d", len(catalog), total)
        if dry_run:
            logger.info("Modo dry-run: listando series sin llamar a la API")
        elif force:
            logger.info("Modo force: refresca aunque ya esté en caché")

    agg = "sum"
    counters: Dict[str, int] = {"ok": 0, "skip": 0, "error": 0, "ok_empty": 0, "dry_run": 0}
    errors: List[Tuple[str, str]] = []

    effective_workers = min(workers, total) if total > 0 else 1

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(_warm_one, sid, freq, agg, force, dry_run): sid
            for sid, freq, _ in series_list
        }

        for i, future in enumerate(as_completed(futures), 1):
            sid, status, err = future.result()
            counters[status] += 1
            if status == "error":
                errors.append((sid, err or ""))
                if not quiet:
                    logger.warning("  [error] %s | %s", sid, err)
            elif not quiet and i % 50 == 0:
                elapsed = time.monotonic() - t0
                remaining = (elapsed / i) * (total - i)
                logger.info(
                    "Progreso: %d/%d | ok=%d skip=%d error=%d | ~%.0fs restantes",
                    i, total, counters["ok"], counters["skip"], counters["error"], remaining,
                )

    elapsed_total = time.monotonic() - t0
    if not quiet:
        logger.info(
            "Precalentamiento completo | total=%d ok=%d skip=%d ok_empty=%d error=%d dry_run=%d | tiempo=%.1fs",
            total,
            counters["ok"],
            counters["skip"],
            counters["ok_empty"],
            counters["error"],
            counters["dry_run"],
            elapsed_total,
        )
        if errors:
            logger.warning("Series con error:")
            for sid, err in errors[:20]:
                logger.warning("  %s: %s", sid, err)
            if len(errors) > 20:
                logger.warning("  ... y %d más", len(errors) - 20)

    return counters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-calienta el caché Redis con todas las series del catálogo.")
    parser.add_argument("--workers", type=int, default=8, help="Hilos paralelos (default: 8)")
    parser.add_argument("--force", action="store_true", help="Refresca incluso series ya cacheadas")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Lista series sin llamar a la API")
    args = parser.parse_args()

    sys.exit(0 if main(workers=args.workers, force=args.force, dry_run=args.dry_run) else 1)
