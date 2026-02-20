"""Setup helper for PIBot Postgres tables.

All tables are created in docker/postgres/init.sql. This script is now a
lightweight helper that tells you how to apply that file.

Usage:
    python tools/setup_memory_tables.py
"""
from __future__ import annotations

def main(argv: Optional[list[str]] = None) -> int:
    print("All tables are created by docker/postgres/init.sql")
    print("Use docker compose or psql -f docker/postgres/init.sql to apply it.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
