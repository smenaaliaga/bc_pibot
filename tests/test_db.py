import os, psycopg
dsn = "postgresql://postgres:postgres@localhost:5432/pibot"
if not dsn:
    raise SystemExit("Falta PG_DSN en el entorno")

with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:
        # Reemplaza esta consulta por lo que necesites
        cur.execute("select thread_id, checkpoint_ns, checkpoint_id, metadata from checkpoints order by checkpoint_id desc limit 100")
        rows = cur.fetchall()
        for r in rows:
            print(r)

