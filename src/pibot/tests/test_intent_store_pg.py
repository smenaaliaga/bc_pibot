import os
import uuid
import pytest

from orchestrator.classifier.intent_store import PostgresIntentStore

try:
    import psycopg  # type: ignore
except Exception:  # pragma: no cover
    psycopg = None


@pytest.mark.skipif(psycopg is None, reason="psycopg not available")
def test_postgres_intent_store_persists_raw_payloads():
    dsn = os.getenv("PG_DSN")
    if not dsn:
        pytest.skip("PG_DSN not set")

    table = f"intents_test_{uuid.uuid4().hex[:8]}"
    store = PostgresIntentStore(dsn=dsn, table=table)

    try:
        intent_raw = {"intent": "ask_data", "source": "router"}
        predict_raw = {"entities": {"domain": ["IMACEC"]}}

        store.record(
            "sess-raw",
            "ask_data",
            0.9,
            spans=[{"text": "foo", "label": "O", "start": 0, "end": 3}],
            entities={"domain": "IMACEC"},
            intent_raw=intent_raw,
            predict_raw=predict_raw,
            turn_id=1,
        )

        rec = store.last("sess-raw")
        assert rec is not None
        assert rec.intent_raw == intent_raw
        assert rec.predict_raw == predict_raw
    finally:
        # cleanup
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()