#!/usr/bin/env python3
"""
Script para generar archivos JSON de metadatos a partir de qa.csv.
Lee cada fila única (combinación de campos de clasificación) y crea un JSON
con la estructura de template.json incluyendo series parseadas y valores de clasificación.
"""
from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd


# Campos de clasificación que definen cada combinación única (para metadata.classification_fields)
CLASSIFICATION_FIELDS = [
    "activity_cls",
    "frequency",
    "calc_mode_cls",
    "region_cls",
    "investment_cls",
    "req_form_cls",
]

# Todos los campos que irán en classification (incluye postprocess)
ALL_CLASSIFICATION_KEYS = [
    "activity_cls",
    "frequency",
    "calc_mode_cls",
    "region_cls",
    "investment_cls",
    "req_form_cls",
    "indicator",
    "seasonality",
    "activity_value",
    "sub_activity_value",
    "region_value",
    "investment_value",
    "gasto_value",
    "price",
    "history",
]

# Campos de postprocess
POSTPROCESS_FIELDS = [
    "indicator",
    "seasonality",
    "activity_value",
    "sub_activity_value",
    "region_value",
    "gasto_value",
    "price",
    "history",
]

# Campos que definen una combinación única de salida
COMBINATION_FIELDS = [
    "activity_cls",
    "frequency",
    "calc_mode_cls",
    "region_cls",
    "investment_cls",
    "req_form_cls",
    "activity_value",
    "sub_activity_value",
    "region_value",
    "investment_value",
    "indicator",
    "seasonality",
    "gasto_value",
    "price",
    "history",
]

# Alias de columnas en el CSV para normalizar cabeceras
COLUMN_ALIASES = {
    "frecuency": "frequency",
    "price ": "price",
}

INDICATORS_FOR_METADATA = ["pib", "imacec"]


def coerce_text(v: Any) -> Optional[str]:
    """Convierte valor a string, retorna None si es NaN."""
    if pd.isna(v):
        return None
    return str(v).strip()


def build_key(classification: Dict[str, Any]) -> str:
    """Crea una clave determinística usando los campos de COMBINATION_FIELDS."""
    parts = []
    for field in COMBINATION_FIELDS:
        parts.append(str(classification.get(field, "none")))
    return "::".join(parts)


def normalize_column_names(columns: List[str]) -> List[str]:
    """Normaliza cabeceras: strip, lower y aplica alias conocidos."""
    normalized = []
    for col in columns:
        base = str(col).strip().lower()
        normalized.append(COLUMN_ALIASES.get(base, base))
    return normalized


def load_dataframe(input_path: str) -> pd.DataFrame:
    """Lee CSV/Excel tolerando encoding y separador usado en qa.csv."""
    if input_path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(input_path, dtype=str)
    else:
        try:
            df = pd.read_csv(input_path, dtype=str, sep=";", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, dtype=str, sep=";", encoding="latin1")

    df.columns = normalize_column_names(df.columns.tolist())
    return df


def parse_series(series_str: Optional[str]) -> Dict[str, Dict[str, str]]:
    """
    Parsea la columna 'series' que viene en formato:
    id1;title1;id2;title2;id3;title3
    Retorna dict con estructura:
    {
      "serie_1": {"id": "...", "title": "..."},
      "serie_2": {"id": "...", "title": "..."},
      ...
    }
    """
    series_dict = {}
    if not series_str:
        return series_dict

    parts = [p.strip() for p in str(series_str).split(";")]
    serie_num = 1
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            serie_id = parts[i]
            serie_title = parts[i + 1]
            series_dict[f"serie_{serie_num}"] = {
                "id": serie_id,
                "title": serie_title,
            }
            serie_num += 1

    return series_dict


def parse_sources(sources_str: Optional[str]) -> Dict[str, str]:
    """
    Parsea la columna 'sources_url' que viene en formato:
    url1;url2;url3
    Retorna dict con estructura:
    {
      "source_1": "url1",
      "source_2": "url2",
      ...
    }
    """
    sources_dict = {}
    if not sources_str:
        return sources_dict

    urls = [u.strip() for u in str(sources_str).split(";") if u.strip()]
    for idx, url in enumerate(urls, start=1):
        sources_dict[f"source_{idx}"] = url

    return sources_dict


def build_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Construye un registro de metadatos (sin el bloque metadata global)."""
    classification = {}
    for key in ALL_CLASSIFICATION_KEYS:
        val = coerce_text(row.get(key))
        classification[key] = val or "none"

    key = build_key(classification)

    serie_default = coerce_text(row.get("serie_default")) or "F032.IMC.IND.Z.Z.EP18.RB.Z.0.M"
    title_serie_default = (
        coerce_text(row.get("title_serie_default"))
        or "Imacec resto de bienes, serie original (índice 2018=100)"
    )

    series_str = coerce_text(row.get("series"))
    series_dict = parse_series(series_str)

    sources_str = coerce_text(row.get("sources_url"))
    sources_dict = parse_sources(sources_str)

    label = coerce_text(row.get("label")) or "pib_general"

    record = {
        "classification": classification,
        "key": key,
        "label": label,
        "serie_default": serie_default,
        "title_serie_default": title_serie_default,
        "series": series_dict,
        "sources_url": sources_dict,
    }

    return record


def generate_metadata_jsons(input_file: str, output_file: Optional[str] = None, output_file_collisions: Optional[str] = None):
    """
    Lee qa.csv y genera dos JSON:
    - metadata_q: registros sin colisión de clave.
    - metadata_q_c: registros con colisión de clave.
    Incluye métricas de QA y validaciones de conteo.
    """
    base_dir = os.path.dirname(__file__)
    input_path = input_file
    if not os.path.isabs(input_path):
        input_path = os.path.join(base_dir, input_file)

    if not os.path.exists(input_path):
        raise SystemExit(f"No existe archivo: {input_path}")

    out_path = output_file
    if out_path is None:
        out_path = os.path.join(base_dir, "metadata_q.json")
    else:
        if not os.path.isabs(out_path):
            out_path = os.path.join(base_dir, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out_path_coll = output_file_collisions
    if out_path_coll is None:
        out_path_coll = os.path.join(base_dir, "metadata_q_c.json")
    else:
        if not os.path.isabs(out_path_coll):
            out_path_coll = os.path.join(base_dir, out_path_coll)
    os.makedirs(os.path.dirname(out_path_coll), exist_ok=True)

    # Leer CSV/Excel con separador y encoding correctos
    df = load_dataframe(input_path)

    print(f"Leyendo {len(df)} filas de {input_path}")

    records_clean: List[Dict[str, Any]] = []
    records_collided: List[Dict[str, Any]] = []
    keys_seen = set()
    duplicate_keys: List[str] = []
    for _, row in df.iterrows():
        rec = build_record(row)
        k = rec["key"]
        if k in keys_seen:
            duplicate_keys.append(k)
            records_collided.append(rec)
        else:
            keys_seen.add(k)
            records_clean.append(rec)

    # Validación de conteo
    rows_read = len(df)
    records_built = len(records_clean) + len(records_collided)
    if rows_read != records_built:
        raise SystemExit(
            f"Conteo inconsistente: filas leídas={rows_read}, registros construidos={records_built}"
        )

    # Métricas QA
    field_none_counts: Dict[str, int] = {k: 0 for k in ALL_CLASSIFICATION_KEYS}
    for rec in records_clean:
        cls = rec["classification"]
        for k in ALL_CLASSIFICATION_KEYS:
            if cls.get(k) == "none":
                field_none_counts[k] += 1
    consolidated = {
        "metadata": {
            "description": "metadatos de las series del pib e imacec",
            "indicators": INDICATORS_FOR_METADATA,
            "classification_fields": CLASSIFICATION_FIELDS,
            "postprocess": POSTPROCESS_FIELDS,
        },
        "data": records_clean,
        "counts": {
            "rows": rows_read,
            "records": records_built,
            "unique_keys": len(keys_seen),
            "duplicate_records": len(records_collided),
            "duplicate_keys": len(duplicate_keys),
            "duplicate_keys_unique": len(set(duplicate_keys)),
        },
        "qa": {
            "duplicate_keys_sample": duplicate_keys[:20],
            "duplicate_keys_unique_sample": list(dict.fromkeys(duplicate_keys))[:20],
            "field_none_counts": field_none_counts,
        },
    }

    consolidated_collisions = {
        "metadata": consolidated["metadata"],
        "data": records_collided,
        "counts": {
            "rows": rows_read,
            "records": len(records_collided),
            "duplicate_keys_unique": len(set(duplicate_keys)),
        },
        "qa": {
            "duplicate_keys_unique_sample": list(dict.fromkeys(duplicate_keys))[:20],
        },
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(consolidated, fh, indent=2, ensure_ascii=False)

    with open(out_path_coll, "w", encoding="utf-8") as fh:
        json.dump(consolidated_collisions, fh, indent=2, ensure_ascii=False)

    print(f"Archivo sin colisiones: {out_path}")
    print(f"Archivo con colisiones: {out_path_coll}")
    print(f"Registros limpios: {len(records_clean)} | registros en colisión: {len(records_collided)} | filas CSV: {rows_read}")

    return out_path, out_path_coll


def main():
    parser = argparse.ArgumentParser(
        description="Generar archivos JSON de metadatos a partir de qa.csv"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="qa.csv",
        help="Archivo CSV de entrada (por defecto: qa.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Archivo de salida limpio (por defecto: metadata_q.json en la carpeta del script)",
    )
    parser.add_argument(
        "--output-collisions",
        "-oc",
        default=None,
        help="Archivo de salida con colisiones (por defecto: metadata_q_c.json en la carpeta del script)",
    )
    args = parser.parse_args()

    generate_metadata_jsons(args.input, args.output, args.output_collisions)


if __name__ == "__main__":
    main()
