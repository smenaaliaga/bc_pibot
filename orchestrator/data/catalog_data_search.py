"""
catalog_data_search.py
======================
Busca directamente en los JSON del data_store por clasificacion,
retornando los payloads completos que hacen match.

Uso:
    python catalog_data_search.py --output orchestrator/memory/data_store --indicator pib --region nuble
    python catalog_data_search.py --output orchestrator/memory/data_store --indicator imacec --seasonality sa --frequency m
    python catalog_data_search.py --output orchestrator/memory/data_store --indicator pib --price enc --first
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _normalize_frequency(freq: str) -> str:
    """Normaliza frecuencia (m/q/a o M/T/A) a M/T/A."""
    if not freq:
        return ""
    f = str(freq).strip().upper()
    if f == "Q":
        return "T"
    return f


def _match_value(payload_value: Any, search_value: Any) -> bool:
    """Evalua si search_value hace match con payload_value.

    - payload_value es lista -> match si search_value esta contenido.
    - payload_value es escalar -> match si son iguales.
    """
    if isinstance(payload_value, list):
        return search_value in payload_value
    return payload_value == search_value


def search_output_payloads(
    output_dir: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Busca payloads en data_store directamente por classification + frequency."""
    if not kwargs:
        return []

    # Caso especial: region implica has_region=1
    if "region" in kwargs and "has_region" not in kwargs:
        kwargs["has_region"] = 1

    # Separar region: puede vivir en classification (single-region files)
    # o en classification_series de cada serie (multi-region files).
    wanted_region = kwargs.pop("region", None)

    # Separar frequency (campo top-level en el payload, no dentro de classification)
    wanted_freq = _normalize_frequency(kwargs.pop("frequency", "") or "")

    out_path = Path(output_dir)
    if not out_path.exists():
        raise FileNotFoundError(f"No existe directorio de output: {output_dir}")

    matches: List[Dict[str, Any]] = []
    for fp in sorted(out_path.glob("*.json")):
        with fp.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        # Filtrar por frequency (top-level)
        if wanted_freq:
            payload_freq = str(payload.get("frequency", "")).upper()
            if payload_freq != wanted_freq:
                continue

        # Filtrar por classification
        cls = payload.get("classification", {})
        match = True
        for key, search_value in kwargs.items():
            cls_value = cls.get(key)
            if cls_value is None:
                match = False
                break
            if not _match_value(cls_value, search_value):
                match = False
                break

        if not match:
            continue

        # Filtrar por region: buscar en classification (single-region)
        # o en classification_series de alguna serie (multi-region)
        if wanted_region:
            if cls.get("region") and _match_value(cls["region"], wanted_region):
                pass  # match at payload level
            else:
                series_match = any(
                    s.get("classification_series", {}).get("region") == wanted_region
                    for s in payload.get("series", [])
                )
                if not series_match:
                    continue

        matches.append(
            {
                "file_path": str(fp),
                "cuadro_name": payload.get("cuadro_name", ""),
                "frequency": payload.get("frequency", ""),
                "classification": cls,
                "payload": payload,
            }
        )

    return matches


def _build_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    for field in [
        "indicator",
        "calc_mode",
        "seasonality",
        "frequency",
        "price",
        "has_activity",
        "has_region",
        "has_investment",
        "region",
        "hist",
    ]:
        val = getattr(args, field, None)
        if val is not None:
            kwargs[field] = val
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Busca payloads JSON en output/ que hagan match con filtros "
            "de clasificacion del catalogo"
        )
    )
    parser.add_argument("--output", default="output", help="Directorio con JSON de salida (data_store)")

    parser.add_argument("--indicator", help="imacec | pib")
    parser.add_argument("--calc-mode", dest="calc_mode", help="original | prev_period | yoy | contribution | share")
    parser.add_argument("--seasonality", help="nsa | sa")
    parser.add_argument("--frequency", help="m | q | a")
    parser.add_argument("--price", help="enc | co")
    parser.add_argument("--has-activity", dest="has_activity", type=int, choices=[0, 1])
    parser.add_argument("--has-region", dest="has_region", type=int, choices=[0, 1])
    parser.add_argument("--has-investment", dest="has_investment", type=int, choices=[0, 1])
    parser.add_argument("--region", help="Ej: nuble, metropolitana, tarapaca")
    parser.add_argument("--hist", type=int, choices=[0, 1])

    parser.add_argument(
        "--first",
        action="store_true",
        help="Retorna solo el primer payload encontrado",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Muestra solo resumen (sin payload completo)",
    )

    args = parser.parse_args()

    kwargs = _build_kwargs(args)
    if not kwargs:
        parser.error("Debe proporcionar al menos un parametro de clasificacion.")

    matches = search_output_payloads(args.output, **kwargs)

    print(f"\n{'=' * 70}")
    print(f"Busqueda: {kwargs}")
    print(f"JSON output encontrados: {len(matches)}")
    print(f"{'=' * 70}\n")

    if not matches:
        return

    if args.summary:
        data = [
            {
                "file_path": m["file_path"],
                "cuadro_name": m["cuadro_name"],
                "frequency": m["frequency"],
                "classification": m["classification"],
            }
            for m in matches
        ]
    elif args.first:
        data = matches[0]["payload"]
    else:
        data = [m["payload"] for m in matches]

    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
