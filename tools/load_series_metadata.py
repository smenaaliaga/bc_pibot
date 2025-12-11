"""
Carga metadata de series (serie_pibot_con_metadata.json) a Postgres.

Usa PG_DSN del entorno o un DSN pasado por variable.
Tabla destino: series_metadata
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import psycopg  # type: ignore
from psycopg import sql  # type: ignore
try:
    from psycopg.types.json import Json  # type: ignore
except Exception:
    Json = None  # type: ignore

try:
    # psycopg >=3
    from psycopg.extras import execute_values  # type: ignore
except Exception:
    execute_values = None  # type: ignore


FIELDS = [
    "COD_SERIE",
    "FREQ",
    "DESC_SERIE_ESP",
    "NKNAME_ESP",
    "CAP_ESP",
    "COD_CAPITULO",
    "COD_CUADRO",
    "DESC_CUAD_ESP",
    "URL",
    "METADATA_UNIDAD",
    "METADATA_FUENTE",
    "METADATA_REZAGO",
    "METADATA_BASE",
    "METADATA_METODOLOGIA",
    "METADATA_CONCEP_EST",
    "METADATA_RECOM_USO",
]


def load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of records")
    return data


def create_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS series_metadata (
                cod_serie TEXT PRIMARY KEY,
                freq TEXT,
                desc_serie_esp TEXT,
                nkname_esp TEXT,
                cap_esp TEXT,
                cod_capitulo TEXT,
                cod_cuadro TEXT,
                desc_cuad_esp TEXT,
                url TEXT,
                metadata_unidad TEXT,
                metadata_fuente TEXT,
                metadata_rezago TEXT,
                metadata_base TEXT,
                metadata_metodologia TEXT,
                metadata_concep_est TEXT,
                metadata_recom_uso TEXT,
                extra JSONB
            );
            CREATE INDEX IF NOT EXISTS idx_series_freq ON series_metadata(freq);
            CREATE INDEX IF NOT EXISTS idx_series_capitulo ON series_metadata(cod_capitulo);
            """
        )
    conn.commit()


def upsert_rows(conn, rows: List[Dict[str, Any]]) -> None:
    records = []
    for row in rows:
        code = row.get("COD_SERIE")
        if not code:
            continue
        rec = [row.get(f) for f in FIELDS]
        extra = {k: v for k, v in row.items() if k not in FIELDS}
        rec.append(extra or None)
        records.append(rec)
    if not records:
        return
    cols = [f.lower() for f in FIELDS] + ["extra"]
    insert_sql = sql.SQL(
        """
        INSERT INTO series_metadata ({cols})
        VALUES %s
        ON CONFLICT (cod_serie)
        DO UPDATE SET {updates}
        """
    ).format(
        cols=sql.SQL(",").join(sql.Identifier(c) for c in cols),
        updates=sql.SQL(",").join(sql.Composed([sql.Identifier(c), sql.SQL("= EXCLUDED."), sql.Identifier(c)]) for c in cols),
    )
    with conn.cursor() as cur:
        if execute_values:
            execute_values(cur, insert_sql.as_string(conn), records, page_size=1000)
        else:
            # Fallback for environments without psycopg.extras.execute_values
            row_placeholder = "(" + ",".join(["%s"] * len(cols)) + ")"
            sql_text = insert_sql.as_string(conn).replace("VALUES %s", f"VALUES {row_placeholder}")
            # Ensure JSONB is adapted when present
            adapted_records = []
            for rec in records:
                adapted = list(rec)
                if Json and isinstance(adapted[-1], dict):
                    adapted[-1] = Json(adapted[-1])
                adapted_records.append(adapted)
            cur.executemany(sql_text, adapted_records)
    conn.commit()


def main() -> None:
    dsn = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot")
    meta_path = Path(os.getenv("SERIES_METADATA_JSON", "series/series_pibot_con_metadata.json")).resolve()
    if not meta_path.exists():
        raise SystemExit(f"Metadata JSON not found: {meta_path}")
    rows = load_json(meta_path)
    with psycopg.connect(dsn) as conn:
        create_table(conn)
        upsert_rows(conn, rows)
    print(f"Loaded {len(rows)} records into series_metadata")


if __name__ == "__main__":
    main()
