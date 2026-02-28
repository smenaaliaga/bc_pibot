from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _normalize_value(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    return value


def _parse_key_values(items: Optional[Iterable[str]]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    if not items:
        return parsed

    for item in items:
        if "=" not in item:
            raise ValueError(f"Formato inválido '{item}'. Usa key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Clave inválida en '{item}'")
        parsed[key] = _normalize_value(value)
    return parsed


def _read_catalog_payload(catalog_path: str | Path) -> Dict[str, Any]:
    path = Path(catalog_path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def load_catalog_series(catalog_path: str | Path) -> List[Dict[str, Any]]:
    payload = _read_catalog_payload(catalog_path)
    if not isinstance(payload, dict):
        return []

    series = payload.get("series")
    if isinstance(series, list):
        return [row for row in series if isinstance(row, dict)]

    rows: List[Dict[str, Any]] = []
    for _, family_payload in payload.items():
        if not isinstance(family_payload, dict):
            continue

        family_classification = family_payload.get("classification")
        family_source_url = family_payload.get("source_url")
        family_series = family_payload.get("series")
        if not isinstance(family_series, list):
            continue

        for item in family_series:
            if not isinstance(item, dict):
                continue

            row = dict(item)
            row_classification = row.get("classification")
            merged_classification: Dict[str, Any] = {}
            if isinstance(family_classification, dict):
                merged_classification.update(family_classification)
            if isinstance(row_classification, dict):
                merged_classification.update(row_classification)
            row["classification"] = merged_classification

            if "source_url" not in row and family_source_url is not None:
                row["source_url"] = family_source_url

            rows.append(row)

    return rows


def _to_flag(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "si"}:
        return 1
    if raw in {"0", "false", "no"}:
        return 0
    return None


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {}, "none")


def find_family_by_classification(
    catalog_path: str | Path,
    *,
    indicator: Any,
    activity_value: Any = None,
    region_value: Any = None,
    investment_value: Any = None,
    calc_mode: Any = None,
    seasonality: Any = None,
    frequency: Any = None,
) -> Optional[Dict[str, Any]]:
    payload = _read_catalog_payload(catalog_path)
    if not payload:
        return None

    requested_indicator = str(indicator or "").strip().lower()
    if not requested_indicator:
        return None

    requested_has_activity = 0 if _is_empty(activity_value) else 1
    requested_has_region = 0 if _is_empty(region_value) else 1
    requested_has_investment = 0 if _is_empty(investment_value) else 1
    requested_calc_mode = str(calc_mode).strip().lower() if calc_mode not in (None, "") else None
    requested_seasonality = str(seasonality).strip().lower() if seasonality not in (None, "") else None
    requested_frequency = str(frequency).strip().lower() if frequency not in (None, "") else None

    candidates: List[Dict[str, Any]] = []
    for family_name, family_payload in payload.items():
        if not isinstance(family_payload, dict):
            continue

        family_classification = family_payload.get("classification")
        family_series = family_payload.get("series")
        if not isinstance(family_classification, dict) or not isinstance(family_series, list):
            continue

        indicator_value = str(family_classification.get("indicator") or "").strip().lower()
        if indicator_value != requested_indicator:
            continue

        has_activity = _to_flag(family_classification.get("has_activity"))
        has_region = _to_flag(family_classification.get("has_region"))
        has_investment = _to_flag(family_classification.get("has_investment"))

        if has_activity is not None and has_activity != requested_has_activity:
            continue
        if has_region is not None and has_region != requested_has_region:
            continue
        if has_investment is not None and has_investment != requested_has_investment:
            continue

        family_calc_mode = family_classification.get("calc_mode")
        if requested_calc_mode is not None and family_calc_mode not in (None, ""):
            if str(family_calc_mode).strip().lower() != requested_calc_mode:
                continue

        family_seasonality = family_classification.get("seasonality")
        if requested_seasonality is not None and family_seasonality not in (None, ""):
            if str(family_seasonality).strip().lower() != requested_seasonality:
                continue

        family_frequency = family_classification.get("frequency")
        if requested_frequency is not None and family_frequency not in (None, ""):
            if str(family_frequency).strip().lower() != requested_frequency:
                continue

        score = 0
        if family_calc_mode not in (None, ""):
            score += 1
        if family_seasonality not in (None, ""):
            score += 1
        if family_frequency not in (None, ""):
            score += 1

        candidates.append(
            {
                "family_name": family_name,
                "classification": family_classification,
                "source_url": family_payload.get("source_url"),
                "series": family_series,
                "_score": score,
            }
        )

    if not candidates:
        return None

    candidates.sort(key=lambda row: row.get("_score", 0), reverse=True)
    selected = dict(candidates[0])
    selected.pop("_score", None)
    return selected


def family_to_series_rows(family_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(family_payload, dict):
        return []

    family_classification = family_payload.get("classification")
    family_source_url = family_payload.get("source_url")
    family_series = family_payload.get("series")
    if not isinstance(family_series, list):
        return []

    rows: List[Dict[str, Any]] = []
    for item in family_series:
        if not isinstance(item, dict):
            continue

        row = dict(item)
        row_classification = row.get("classification")
        merged_classification: Dict[str, Any] = {}
        if isinstance(family_classification, dict):
            merged_classification.update(family_classification)
        if isinstance(row_classification, dict):
            merged_classification.update(row_classification)
        row["classification"] = merged_classification

        if "source_url" not in row and family_source_url is not None:
            row["source_url"] = family_source_url

        rows.append(row)

    return rows


def _classification_value(classification: Dict[str, Any], key: str) -> Any:
    if key in classification:
        return classification.get(key)

    general = classification.get("general")
    if isinstance(general, dict) and key in general:
        return general.get(key)

    specific = classification.get("specific")
    if isinstance(specific, dict) and key in specific:
        return specific.get(key)

    return None


def find_series_by_classification(
    catalog_path: str | Path,
    *,
    eq: Optional[Dict[str, Any]] = None,
    ne: Optional[Dict[str, Any]] = None,
    require_not_null: Optional[Iterable[str]] = None,
    require_null: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Busca series por filtros de classification en catalog.json.

    Ejemplo típico:
      - indicator=imacec, calc_mode=original, seasonality=nsa
      - activity distinto de null
    """

    equals = dict(eq or {})
    not_equals = dict(ne or {})
    not_null_fields = set(require_not_null or [])
    null_fields = set(require_null or [])
    rows = load_catalog_series(catalog_path)

    matches: List[Dict[str, Any]] = []
    for row in rows:
        classification = row.get("classification")
        if not isinstance(classification, dict):
            continue

        ok = True

        for key, expected in equals.items():
            if _classification_value(classification, key) != expected:
                ok = False
                break
        if not ok:
            continue

        for key, expected in not_equals.items():
            if _classification_value(classification, key) == expected:
                ok = False
                break
        if not ok:
            continue

        for key in not_null_fields:
            if _classification_value(classification, key) is None:
                ok = False
                break
        if not ok:
            continue

        for key in null_fields:
            if _classification_value(classification, key) is not None:
                ok = False
                break
        if not ok:
            continue

        matches.append(row)
        if isinstance(limit, int) and limit > 0 and len(matches) >= limit:
            break

    return matches


def select_target_series(
    catalog_matches: Iterable[Dict[str, Any]],
    *,
    activity: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Selecciona una serie target desde un subconjunto ya filtrado.

    Regla:
    - Si `activity` viene informada, intenta match exacto por classification.activity.
    - Si no hay match exacto (o no se envía `activity`), retorna el primer elemento disponible.
    """

    matches = [row for row in catalog_matches if isinstance(row, dict)]
    if not matches:
        return None

    requested_activity = str(activity or "").strip().lower()
    if requested_activity:
        for row in matches:
            classification = row.get("classification")
            if not isinstance(classification, dict):
                continue
            row_activity = _classification_value(classification, "activity")
            row_activity_norm = str(row_activity or "").strip().lower()
            if row_activity_norm == requested_activity:
                return row

    return matches[0]


def select_target_series_by_classification(
    family_series: Iterable[Dict[str, Any]],
    *,
    eq: Optional[Dict[str, Any]] = None,
    fallback_to_first: bool = True,
) -> Optional[Dict[str, Any]]:
    """Selecciona serie target dentro de una familia usando filtros de classification.

    Ejemplo:
      eq={"activity": "mineria", "region": "metropolitana"}
    """

    matches = [row for row in family_series if isinstance(row, dict)]
    if not matches:
        return None

    filters = dict(eq or {})
    normalized_filters = {
        key: str(value).strip().lower() if value is not None else None
        for key, value in filters.items()
        if key
    }

    available_with_value: Dict[str, bool] = {}
    for key in normalized_filters.keys():
        available_with_value[key] = any(
            isinstance(row.get("classification"), dict)
            and _classification_value(row.get("classification"), key) is not None
            for row in matches
        )

    if normalized_filters:
        for row in matches:
            classification = row.get("classification")
            if not isinstance(classification, dict):
                continue

            ok = True
            for key, expected in normalized_filters.items():
                if expected is not None and not available_with_value.get(key, False):
                    continue
                current = _classification_value(classification, key)
                current_norm = str(current).strip().lower() if current is not None else None
                if current_norm != expected:
                    ok = False
                    break

            if ok:
                return row

    return matches[0] if fallback_to_first else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Busca series por campos de classification en catalog.json",
    )
    parser.add_argument(
        "--catalog",
        default=str(Path(__file__).with_name("catalog.json")),
        help="Ruta a catalog.json",
    )
    parser.add_argument(
        "--eq",
        action="append",
        default=[],
        help="Filtro igualdad: key=value (repetible)",
    )
    parser.add_argument(
        "--ne",
        action="append",
        default=[],
        help="Filtro desigualdad: key=value (repetible)",
    )
    parser.add_argument(
        "--not-null",
        dest="not_null",
        action="append",
        default=[],
        help="Campo que debe ser distinto de null (repetible)",
    )
    parser.add_argument(
        "--is-null",
        dest="is_null",
        action="append",
        default=[],
        help="Campo que debe ser null (repetible)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Máximo de resultados (0 = sin límite)",
    )
    parser.add_argument(
        "--ids-only",
        action="store_true",
        help="Imprime solo IDs de series",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    eq = _parse_key_values(args.eq)
    ne = _parse_key_values(args.ne)
    limit = args.limit if args.limit and args.limit > 0 else None

    matches = find_series_by_classification(
        args.catalog,
        eq=eq,
        ne=ne,
        require_not_null=args.not_null,
        require_null=args.is_null,
        limit=limit,
    )

    if args.ids_only:
        for row in matches:
            print(row.get("id"))
    else:
        print(json.dumps(matches, ensure_ascii=False, indent=2))

    print(f"\nTotal: {len(matches)}")


if __name__ == "__main__":
    main()
