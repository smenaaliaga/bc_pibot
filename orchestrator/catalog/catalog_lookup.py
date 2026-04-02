from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from orchestrator.normalizer.normalizer import ResolvedEntities

logger = logging.getLogger(__name__)


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
            has_explicit_empty_classification = isinstance(row_classification, dict) and not row_classification
            merged_classification: Dict[str, Any] = {}
            if isinstance(family_classification, dict):
                merged_classification.update(family_classification)
            if isinstance(row_classification, dict):
                merged_classification.update(row_classification)
            row["classification"] = merged_classification
            row["_classification_empty"] = has_explicit_empty_classification

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


def _to_flag_options(value: Any) -> Optional[set[int]]:
    if isinstance(value, (list, tuple, set)):
        options: set[int] = set()
        for item in value:
            flag = _to_flag(item)
            if flag is not None:
                options.add(flag)
        return options or None

    single = _to_flag(value)
    if single is None:
        return None
    return {single}


def _is_empty(value: Any) -> bool:
    return value in (None, "", [], {}, "none")


def _normalized_token(value: Any) -> Optional[str]:
    if value in (None, "", [], {}, ()): 
        return None
    token = str(value).strip().lower()
    if token in {"", "none", "null"}:
        return None
    return token


def find_family_by_classification(
    catalog_path: str | Path,
    *,
    indicator: Any,
    activity_value: Any = None,
    region_value: Any = None,
    investment_value: Any = None,
    calc_mode: Any = None,
    price: Any = None,
    seasonality: Any = None,
    frequency: Any = None,
    hist: Any = None,
) -> Optional[Dict[str, Any]]:
    payload = _read_catalog_payload(catalog_path)
    if not payload:
        return None

    requested_indicator = str(indicator or "").strip().lower() or None

    requested_has_activity = 0 if _is_empty(activity_value) else 1
    requested_has_region = 0 if _is_empty(region_value) else 1
    requested_has_investment = 0 if _is_empty(investment_value) else 1
    requested_activity_token = _normalized_token(activity_value)
    requested_region_token = _normalized_token(region_value)
    requested_investment_token = _normalized_token(investment_value)
    requested_calc_mode = str(calc_mode).strip().lower() if calc_mode not in (None, "") else None
    requested_price = str(price).strip().lower() if price not in (None, "") else None
    requested_seasonality = str(seasonality).strip().lower() if seasonality not in (None, "") else None
    requested_frequency = str(frequency).strip().lower() if frequency not in (None, "") else None
    requested_hist = _to_flag(hist)

    has_hist_dimension = any(
        isinstance(family_payload, dict)
        and isinstance(family_payload.get("classification"), dict)
        and _to_flag_options(family_payload.get("classification", {}).get("hist")) is not None
        for family_payload in payload.values()
    )

    candidates: List[Dict[str, Any]] = []
    for family_name, family_payload in payload.items():
        if not isinstance(family_payload, dict):
            continue

        family_classification = family_payload.get("classification")
        family_series = family_payload.get("series")
        if not isinstance(family_classification, dict) or not isinstance(family_series, list):
            continue

        indicator_value = str(family_classification.get("indicator") or "").strip().lower() or None

        has_activity_options = _to_flag_options(family_classification.get("has_activity"))
        has_region_options = _to_flag_options(family_classification.get("has_region"))
        has_investment_options = _to_flag_options(family_classification.get("has_investment"))
        family_activity_token = _normalized_token(family_classification.get("activity"))
        family_region_token = _normalized_token(family_classification.get("region"))
        family_investment_token = _normalized_token(family_classification.get("investment"))
        family_has_hist_key = "hist" in family_classification
        family_hist_options = _to_flag_options(family_classification.get("hist"))

        if has_activity_options is None or requested_has_activity not in has_activity_options:
            continue
        if has_region_options is None or requested_has_region not in has_region_options:
            continue
        if has_investment_options is None or requested_has_investment not in has_investment_options:
            continue

        if requested_has_activity == 1 and requested_activity_token is not None:
            if family_activity_token is not None and family_activity_token != requested_activity_token:
                continue
        if requested_has_region == 1 and requested_region_token is not None:
            if family_region_token is not None and family_region_token != requested_region_token:
                continue
        if requested_has_investment == 1 and requested_investment_token is not None:
            if family_investment_token is not None and family_investment_token != requested_investment_token:
                continue

        family_calc_mode = family_classification.get("calc_mode")
        if requested_calc_mode is not None:
            if isinstance(family_calc_mode, list):
                family_calc_modes = {str(v).strip().lower() for v in family_calc_mode}
                if requested_calc_mode not in family_calc_modes:
                    continue
            else:
                family_calc_mode_normalized = (
                    str(family_calc_mode).strip().lower() if family_calc_mode not in (None, "") else None
                )
                if family_calc_mode_normalized != requested_calc_mode:
                    continue

        family_price = family_classification.get("price")
        if requested_price is not None:
            if isinstance(family_price, list):
                family_prices = {str(v).strip().lower() for v in family_price}
                if requested_price not in family_prices:
                    continue
            else:
                family_price_normalized = (
                    str(family_price).strip().lower() if family_price not in (None, "") else None
                )
                if family_price_normalized != requested_price:
                    continue

        family_seasonality = family_classification.get("seasonality")
        if requested_seasonality is not None:
            if isinstance(family_seasonality, list):
                family_seasonalities = {str(v).strip().lower() for v in family_seasonality}
                if requested_seasonality not in family_seasonalities:
                    continue
            else:
                family_seasonality_normalized = (
                    str(family_seasonality).strip().lower() if family_seasonality not in (None, "") else None
                )
                if family_seasonality_normalized != requested_seasonality:
                    continue

        family_frequency = family_classification.get("frequency")
        if requested_frequency is not None:
            if isinstance(family_frequency, list):
                family_frequencies = {str(v).strip().lower() for v in family_frequency}
                if requested_frequency not in family_frequencies:
                    continue
            else:
                family_frequency_normalized = (
                    str(family_frequency).strip().lower() if family_frequency not in (None, "") else None
                )
                if family_frequency_normalized != requested_frequency:
                    continue

        if has_hist_dimension:
            if requested_hist is None:
                if family_has_hist_key:
                    continue
            elif family_hist_options is None or requested_hist not in family_hist_options:
                continue

        score = 0
        if requested_indicator is not None and indicator_value == requested_indicator:
            score += 100
        if requested_activity_token is not None and family_activity_token == requested_activity_token:
            score += 20
        if requested_region_token is not None and family_region_token == requested_region_token:
            score += 30
        if requested_investment_token is not None and family_investment_token == requested_investment_token:
            score += 20
        if family_calc_mode not in (None, ""):
            score += 1
        if family_price not in (None, ""):
            score += 1
        if family_seasonality not in (None, ""):
            score += 1
        if family_frequency not in (None, ""):
            score += 1
        if (
            requested_hist is not None
            and family_hist_options is not None
            and requested_hist in family_hist_options
        ):
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
        has_explicit_empty_classification = isinstance(row_classification, dict) and not row_classification
        merged_classification: Dict[str, Any] = {}
        if isinstance(family_classification, dict):
            merged_classification.update(family_classification)
        if isinstance(row_classification, dict):
            merged_classification.update(row_classification)
        row["classification"] = merged_classification
        row["_classification_empty"] = has_explicit_empty_classification

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
            if row.get("_classification_empty") is True:
                continue

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

    # Si se proporcionaron filtros de clasificación y ninguno matcheó,
    # no aplicar fallback silencioso al primer elemento.
    if normalized_filters:
        return None

    if not fallback_to_first:
        return None

    for row in matches:
        if row.get("_classification_empty") is not True:
            return row

    return None


# ---------------------------------------------------------------------------
# Búsqueda de serie de alto nivel (antes en _series_lookup.py)
# ---------------------------------------------------------------------------

@dataclass
class SeriesLookupResult:
    """Resultado de la búsqueda de serie en el catálogo."""

    family_dict: Optional[Dict[str, Any]] = None
    family_series: List[Dict[str, Any]] = None  # type: ignore[assignment]
    family_name: Optional[str] = None
    source_url: Optional[str] = None
    target_series: Optional[Dict[str, Any]] = None
    target_series_id: Optional[str] = None
    target_series_title: Optional[str] = None
    target_series_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.family_series is None:
            self.family_series = []


_EMPTY_CLS_VALUES = (None, "none", "", {}, [], ())


def _is_aggregate_request(ent: ResolvedEntities) -> bool:
    """Retorna True cuando no hay desglose de actividad/region/inversion."""
    return (
        ent.activity_cls_resolved in (*_EMPTY_CLS_VALUES, "general")
        and ent.region_cls in _EMPTY_CLS_VALUES
        and ent.investment_cls in _EMPTY_CLS_VALUES
    )


def _is_contribution_aggregate_request(ent: ResolvedEntities) -> bool:
    """Retorna True para consultas agregadas en modo contribucion."""
    return (
        str(ent.calc_mode_cls or "").strip().lower() == "contribution"
        and _is_aggregate_request(ent)
    )


def lookup_series(ent: ResolvedEntities) -> SeriesLookupResult:
    """Localiza la familia de series y selecciona la serie objetivo.

    Pasos:
      1. Calcula los filtros de familia (frecuencia, precio, estacionalidad, etc.).
      2. Busca la familia en el catálogo con ``find_family_by_classification``.
      3. Construye el diccionario de igualdad (``series_eq``) para seleccionar la
         serie concreta dentro de la familia.
      4. Retorna un ``SeriesLookupResult`` con toda la información necesaria.
    """
    from orchestrator.data._helpers import build_target_series_url

    result = SeriesLookupResult()

    # --- Filtros de familia ---------------------------------------------------
    family_frequency = None if ent.indicator_ent == "imacec" else ent.frequency_ent
    family_price = None if ent.indicator_ent == "imacec" else ent.price

    is_pib_aggregate = ent.indicator_ent == "pib" and _is_aggregate_request(ent)

    # Para PIB agregado (sin desglose) la familia base no tiene frequency
    # a nivel de clasificación; el resampleo se hace al cargar observaciones.
    # Para familias con desglose (regionales, etc.) la frecuencia SÍ importa.
    #
    # Las familias de volumen de PIB por actividad a nivel nacional (sin
    # región ni inversión) tampoco llevan frequency en la clasificación;
    # los datos trimestrales se re-muestrean a anual en load_observations.
    # Solo las familias de contribución y las regionales separan Q / A.
    _is_pib_no_freq = (
        ent.indicator_ent == "pib"
        and str(ent.calc_mode_cls or "").strip().lower() != "contribution"
        and ent.region_cls in _EMPTY_CLS_VALUES
        and ent.investment_cls in _EMPTY_CLS_VALUES
    )
    if _is_pib_no_freq:
        family_frequency = None

    family_calc_mode = ent.calc_mode_cls or "original"

    # Estacionalidad: para PIB agregado (sin desglose) sin contribución se
    # omite porque la familia base de PIB no tiene seasonality a nivel de
    # familia.  Pero si activity="general" (desglose por actividades), SÍ
    # se filtra para distinguir nsa vs sa.
    if (
        is_pib_aggregate
        and ent.calc_mode_cls != "contribution"
        and ent.activity_cls_resolved != "general"
    ):
        family_seasonality = None
    else:
        family_seasonality = ent.seasonality_ent

    # --- Buscar familia -------------------------------------------------------
    # "general" significa "dame TODAS las actividades/regiones/inversiones",
    # es decir has_activity/has_region/has_investment == 1.
    # Solo "none" y los vacíos implican ausencia (has_* == 0).

    _activity_fallback = (
        ent.activity_cls_resolved
        if ent.activity_cls_resolved not in _EMPTY_CLS_VALUES
        else None
    )
    _region_fallback = (
        ent.region_cls if ent.region_cls not in _EMPTY_CLS_VALUES else None
    )
    _investment_fallback = (
        ent.investment_cls if ent.investment_cls not in _EMPTY_CLS_VALUES else None
    )

    result.family_dict = find_family_by_classification(
        "orchestrator/catalog/catalog.json",
        indicator=ent.indicator_ent,
        activity_value=(
            ent.activity_ent[0] if ent.activity_ent
            else _activity_fallback
        ),
        region_value=(
            ent.region_ent[0] if ent.region_ent else _region_fallback
        ),
        investment_value=(
            ent.investment_ent[0] if ent.investment_ent
            else _investment_fallback
        ),
        calc_mode=family_calc_mode,
        price=family_price,
        seasonality=family_seasonality,
        frequency=family_frequency,
        hist=ent.hist,
    )

    if isinstance(result.family_dict, dict):
        result.family_series = family_to_series_rows(result.family_dict)
        result.source_url = result.family_dict.get("source_url")
        result.family_name = result.family_dict.get("family_name")
    else:
        result.family_series = []

    logger.info("[DATA_NODE] family_name=%s", result.family_name)
    logger.info("[DATA_NODE] family_source_url=%s", result.source_url)
    logger.info("[DATA_NODE] =========================================================")

    # --- Seleccionar serie objetivo -------------------------------------------
    series_eq: Dict[str, Any] = {
        "indicator": ent.indicator_ent,
        "seasonality": ent.seasonality_ent,
        "activity": ent.activity_ent[0] if ent.activity_ent else None,
        "region": ent.region_ent[0] if ent.region_ent else None,
        "investment": ent.investment_ent[0] if ent.investment_ent else None,
    }

    # Para contribuciones generales de PIB/IMACEC, ajustar la clave de actividad
    # al token que exista en la familia de series.
    if _is_contribution_aggregate_request(ent):
        indicator_norm = str(ent.indicator_ent or "").strip().lower()
        if indicator_norm in {"pib", "imacec"}:
            activity_tokens_in_family = {
                str(
                    ((row.get("classification") or {}).get("activity") or "")
                ).strip().lower()
                for row in result.family_series
                if isinstance(row, dict)
            }
            if indicator_norm in activity_tokens_in_family:
                series_eq["activity"] = indicator_norm
            else:
                series_eq.pop("activity", None)
                series_eq["indicator"] = indicator_norm

    if ent.activity_cls_resolved == "specific" and not ent.activity_ent:
        series_eq["activity"] = "__missing_specific_activity__"

    # Consultas agregadas (sin actividad especifica): no forzar filtro activity,
    # ya que para IMACEC/PIB la serie principal suele venir solo con indicator.
    if (
        str(ent.calc_mode_cls or "").strip().lower() != "contribution"
        and _is_aggregate_request(ent)
    ):
        series_eq.pop("activity", None)

    # Series históricas solo existen en nsa; no filtrar por seasonality.
    if ent.hist == 1:
        series_eq.pop("seasonality", None)

    result.target_series = select_target_series_by_classification(
        result.family_series,
        eq=series_eq,
        fallback_to_first=True,
    )

    if isinstance(result.target_series, dict):
        result.target_series_id = result.target_series.get("id")
        long_raw = result.target_series.get("long_title")
        display_raw = result.target_series.get("display_title")
        result.target_series_title = str(long_raw or display_raw or "").strip()

    result.target_series_url = build_target_series_url(
        source_url=result.source_url,
        series_id=result.target_series_id,
        period=ent.period_ent if isinstance(ent.period_ent, list) else None,
        frequency=ent.frequency_ent,
        calc_mode=ent.calc_mode_cls,
    )

    logger.info("[DATA_NODE] target_series_id=%s", result.target_series_id)
    logger.info("[DATA_NODE] target_series_title=%s", result.target_series_title)
    logger.info("[DATA_NODE] target_series_url=%s", result.target_series_url)
    logger.info("[DATA_NODE] =========================================================")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
