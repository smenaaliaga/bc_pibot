"""Simple connectivity check against the BCCh REST API.

Usage:
    python tools/test_bcch_connectivity.py --domain IMACEC --year 2024

The script fetches the default series for the selected domain using
`get_series_api_rest_bcch` and prints a short summary so you can quickly
confirm credentials, network reachability, and API health.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Dict, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import BCCH_PASS, BCCH_USER
from orchestrator.data.get_series import get_series_api_rest_bcch

_DEFAULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "series", "config_default.json"))
_DOMAIN_KEYS = {"IMACEC": "IMACEC", "PIB": "PIB_TOTAL", "PIB_REGIONAL": "PIB_REGIONAL"}


def _load_default_series(domain: str) -> Tuple[str, str]:
    if not os.path.exists(_DEFAULTS_PATH):
        raise FileNotFoundError(f"No pude encontrar {_DEFAULTS_PATH}")
    with open(_DEFAULTS_PATH, "r", encoding="utf-8") as fh:
        data: Dict[str, Dict[str, Dict[str, str]]] = json.load(fh)
    block = data.get("defaults", {}).get(_DOMAIN_KEYS[domain])
    if not block:
        raise KeyError(f"No hay configuración por defecto para {domain}")
    return block.get("cod_serie", ""), block.get("freq_por_defecto", "")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida la conectividad con la API REST del BCCh")
    parser.add_argument("--domain", default="IMACEC", choices=sorted(_DOMAIN_KEYS.keys()), help="Dominio estándar a consultar")
    parser.add_argument("--year", type=int, default=dt.date.today().year, help="Año objetivo para construir el rango (incluye year-1)")
    parser.add_argument("--series-id", dest="series_id", help="Sobrescribe el código de serie por defecto")
    parser.add_argument("--target-frequency", dest="target_frequency", help="Sobrescribe la frecuencia objetivo")
    parser.add_argument("--agg", default="avg", help="Agregación para remuestreo (avg/sum/first/last)")
    return parser.parse_args()


def main() -> int:
    if not BCCH_USER or not BCCH_PASS:
        print("BCCH_USER/BCCH_PASS no configurados; revisa tu .env", file=sys.stderr)
        return 2

    args = _parse_args()
    domain = args.domain.upper()
    series_id = args.series_id
    target_frequency = args.target_frequency

    if not series_id:
        series_id, default_freq = _load_default_series(domain)
        if not series_id:
            print(f"No encontré un cod_serie para {domain}", file=sys.stderr)
            return 3
        if not target_frequency:
            target_frequency = default_freq

    year = args.year
    target_date = f"{year}-12-31"

    print(
        f"Consultando {series_id} | dominio={domain} | target_date={target_date} | freq={target_frequency or 'orig'}"
    )
    data = get_series_api_rest_bcch(
        series_id=series_id,
        target_date=target_date,
        target_frequency=target_frequency,
        agg=args.agg,
    )

    observations = (data or {}).get("observations", [])
    meta = (data or {}).get("meta", {})
    print(
        "Resultado: filas={rows} freq={freq} serie={sid} descripcion={desc}".format(
            rows=len(observations),
            freq=meta.get("freq_effective"),
            sid=meta.get("series_id"),
            desc=meta.get("descripEsp", ""),
        )
    )
    if observations:
        last_obs = observations[-1]
        print(
            "Último dato: fecha={date} valor={value} yoy_pct={yoy}".format(
                date=last_obs.get("date"),
                value=last_obs.get("value"),
                yoy=last_obs.get("yoy_pct"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
