"""
Purge series from series_metadata whose cod_serie matches F035.PIB.*.2013.*

Usage:
  PG_DSN=postgresql://postgres:postgres@localhost:5432/pibot python tools/purge_pib_2013_series.py
"""
import os
import psycopg  # type: ignore
import re


def main() -> None:
    dsn = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot")
    pattern = re.compile(r"^F035\.PIB\..*\.2013\..*")
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT cod_serie FROM series_metadata WHERE cod_serie LIKE 'F035.PIB.%'")
            codes = [r[0] for r in cur.fetchall() or []]
            targets = [c for c in codes if pattern.match(c or "")]
            if not targets:
                print("No matching series to purge.")
                return
            cur.execute(
                "DELETE FROM series_metadata WHERE cod_serie = ANY(%s)",
                (targets,),
            )
        conn.commit()
    print(f"Deleted {len(targets)} series matching F035.PIB.*.2013.*")


if __name__ == "__main__":
    main()
