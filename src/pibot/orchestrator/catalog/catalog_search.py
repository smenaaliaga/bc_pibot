"""
catalog_data_search.py
=================
Buscador de cuadros de series en catalog.json por clasificación.

Dado un conjunto de parámetros de clasificación (kwargs), filtra los cuadros
cuya classification haga match con todos los criterios entregados.

Reglas de matching:
  - Si el campo en el catálogo es una lista → match si el valor buscado está contenido.
  - Si el campo en el catálogo es un escalar → match si el valor buscado es igual.
  - Si el campo no existe en la classification del cuadro → no hace match.
  - Pasar region="X" implica automáticamente has_region=1.

Uso:
    python catalog_data_search.py --indicator pib --region nuble
    python catalog_data_search.py --indicator imacec --seasonality sa --frequency m
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_catalog(catalog_path: str = "catalog.json") -> Dict[str, Any]:
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _match_value(catalog_value: Any, search_value: Any) -> bool:
    """Evalúa si search_value hace match con catalog_value.

    - catalog_value es lista → match si search_value está contenido.
    - catalog_value es escalar → match si son iguales.
    """
    if isinstance(catalog_value, list):
        return search_value in catalog_value
    return catalog_value == search_value


def search_cuadros(
    catalog: Dict[str, Any],
    **kwargs,
) -> List[Dict[str, Any]]:
    """Busca cuadros cuya classification haga match con todos los kwargs.

    Args:
        catalog: dict cargado de catalog.json {cuadro_name: {classification, source_url, series}}
        **kwargs: pares clave-valor de clasificación, ej: indicator="pib", region="nuble"

    Returns:
        Lista de dicts con cuadro_name, classification, source_url y series.
    """
    if not kwargs:
        return []

    # Caso especial: si se pasa region, implica has_region=1
    if "region" in kwargs and "has_region" not in kwargs:
        kwargs["has_region"] = 1

    results = []
    for cuadro_name, cuadro_data in catalog.items():
        cls = cuadro_data.get("classification", {})

        match = True
        for key, search_value in kwargs.items():
            catalog_value = cls.get(key)
            if catalog_value is None:
                match = False
                break
            if not _match_value(catalog_value, search_value):
                match = False
                break

        if match:
            results.append({
                "cuadro_name": cuadro_name,
                "classification": cls,
                "source_url": cuadro_data.get("source_url", ""),
                "series": cuadro_data.get("series", []),
            })

    return results


def search_cuadros_json(
    catalog: Dict[str, Any],
    **kwargs,
) -> str:
    """Wrapper que retorna el resultado de search_cuadros como JSON formateado."""
    results = search_cuadros(catalog, **kwargs)
    return json.dumps(results, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Campos conocidos del catálogo y su tipo esperado
_INT_FIELDS = {"has_activity", "has_region", "has_investment", "hist"}


def main():
    parser = argparse.ArgumentParser(
        description="Busca cuadros de series por clasificación en catalog.json"
    )
    parser.add_argument("--catalog", default="catalog.json", help="Ruta a catalog.json")
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
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)

    # Construir kwargs sólo con los argumentos proporcionados
    kwargs = {}
    for field in ["indicator", "calc_mode", "seasonality", "frequency", "price",
                   "has_activity", "has_region", "has_investment", "region", "hist"]:
        val = getattr(args, field, None)
        if val is not None:
            kwargs[field] = val

    if not kwargs:
        parser.error("Debe proporcionar al menos un parámetro de clasificación.")

    results = search_cuadros(catalog, **kwargs)

    print(f"\n{'='*60}")
    print(f"  Búsqueda: {kwargs}")
    print(f"  Cuadros encontrados: {len(results)}")
    print(f"{'='*60}\n")

    for i, fam in enumerate(results, 1):
        print(f"[{i}] {fam['cuadro_name']}")
        print(f"    Classification: {json.dumps(fam['classification'], ensure_ascii=False)}")
        print(f"    Series: {len(fam['series'])} series")
        print(f"    Source: {fam['source_url']}")
        for s in fam["series"]:
            print(f"      - {s['id']}: {s['short_title']}")
        print()


if __name__ == "__main__":
    main()
