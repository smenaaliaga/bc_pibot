"""Persistent intent store backends (Postgres/Redis) with in-memory fallback."""
from __future__ import annotations

import os
import json
import uuid
import time
import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional

from .intent_memory import IntentRecord, IntentMemory
from orchestrator.utils.pg_logging import throttled_pg_log

logger = logging.getLogger(__name__)

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None
try:
    from psycopg_pool import ConnectionPool  # type: ignore
except Exception:
    ConnectionPool = None

try:
    import redis  # type: ignore
except Exception:
    redis = None


def _json_safe(value: Any) -> Any:
    """Convert complex values (dates, dataclasses) into JSON-serializable types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return _json_safe(value.__dict__)
    return str(value)


class IntentStoreBase:
    def record(
        self,
        session_id: str,
        intent: str,
        score: float,
        spans: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[Dict[str, Any]] = None,
        intent_raw: Optional[Dict[str, Any]] = None,
        predict_raw: Optional[Dict[str, Any]] = None,
        turn_id: int = 0,
        model_version: Optional[str] = None,
    ) -> IntentRecord:
        raise NotImplementedError

    def last(self, session_id: str) -> Optional[IntentRecord]:
        raise NotImplementedError

    def history(self, session_id: str, k: int = 10) -> List[IntentRecord]:
        raise NotImplementedError


class InMemoryIntentStore(IntentMemory, IntentStoreBase):
    """Alias to reuse the in-process store."""

    pass


class PostgresIntentStore(IntentStoreBase):
    def __init__(self, dsn: str, table: str = "intents", ttl_days: Optional[int] = None):
        if not psycopg:
            raise RuntimeError("psycopg not available for PostgresIntentStore")
        self.dsn = dsn
        self.table = table
        self.ttl_days = ttl_days
        self._pool = None
        self._pool_tag = "direct"
        self._pg_log_state: Dict[str, Any] = {}
        self._pg_err_period = float(os.getenv("PG_ERROR_LOG_PERIOD", "60"))
        if ConnectionPool is not None:
            try:
                self._pool = ConnectionPool(
                    conninfo=self.dsn,
                    max_size=int(os.getenv("PG_POOL_SIZE", "5")),
                    open=False,
                )
                self._pool_tag = "pool"
                try:
                    self._pool.open()
                except Exception:
                    self._log_pg_error("Failed to open psycopg pool for intents; using direct connects", op="pool_open")
                    self._pool = None
                    self._pool_tag = "direct"
            except Exception:
                self._log_pg_error("Could not create psycopg pool for PostgresIntentStore; using direct connects")
        self._ensure_table()

    def _log_pg_error(
        self,
        msg: str,
        *,
        session_id: Optional[str] = None,
        op: Optional[str] = None,
        table: Optional[str] = None,
    ) -> None:
        throttled_pg_log(
            logger,
            self._pg_log_state,
            msg,
            session_id=session_id,
            op=op,
            table=table or self.table,
            pool=self._pool_tag,
            period=self._pg_err_period,
        )

    def _conn_ctx(self):
        if self._pool:
            try:
                if getattr(self._pool, "closed", False):
                    try:
                        self._pool.open()
                    except Exception:
                        self._log_pg_error("Pool was closed; reopening failed", op="pool_reopen")
                        self._pool = None
                        self._pool_tag = "direct"
                        return None
                import time as _t

                start = _t.time()
                ctx = self._pool.connection()
                elapsed = (_t.time() - start) * 1000.0
                logger.info("metric pg_pool_acquire_time_ms=%.2f", elapsed, extra={"op": "pool_connection", "table": self.table})
                return ctx
            except Exception:
                self._log_pg_error("Pool connection failed for PostgresIntentStore; falling back to direct connect")
                try:
                    self._pool.close()
                except Exception:
                    pass
                self._pool = None
                self._pool_tag = "direct"
        for _ in range(2):
            try:
                return psycopg.connect(self.dsn)
            except Exception:
                self._log_pg_error("Could not open Postgres connection for intents; retrying")
                time.sleep(0.2)
        return None

    def _ensure_table(self):
        try:
            conn_ctx = self._conn_ctx()
            if conn_ctx is None:
                return
            with conn_ctx as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.table} (
                            id UUID PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            turn_id INTEGER DEFAULT 0,
                            intent TEXT,
                            score DOUBLE PRECISION,
                            spans JSONB,
                            entities JSONB,
                            intent_raw JSONB,
                            predict_raw JSONB,
                            model_version TEXT,
                            ts TIMESTAMPTZ DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_intents_session_ts ON {self.table}(session_id, ts DESC);
                        """
                    )
                    try:
                        cur.execute(
                            f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS intent_raw JSONB"
                        )
                        cur.execute(
                            f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS predict_raw JSONB"
                        )
                    except Exception:
                        self._log_pg_error("Failed to add raw columns to intents table", op="schema", table=self.table)
                    # optional TTL cleanup if desired on init (lightweight)
                    if self.ttl_days:
                        cur.execute(
                            f"DELETE FROM {self.table} WHERE ts < NOW() - INTERVAL '{int(self.ttl_days)} days'"
                        )
        except Exception:
            self._log_pg_error("Failed to ensure intents table in Postgres")

    def record(
        self,
        session_id: str,
        intent: str,
        score: float,
        spans: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[Dict[str, Any]] = None,
        intent_raw: Optional[Dict[str, Any]] = None,
        predict_raw: Optional[Dict[str, Any]] = None,
        turn_id: int = 0,
        model_version: Optional[str] = None,
    ) -> IntentRecord:
        rec = IntentRecord(
            intent=intent or "",
            score=float(score or 0.0),
            spans=list(spans or []),
            entities=dict(entities or {}),
            intent_raw=dict(intent_raw or {}),
            predict_raw=dict(predict_raw or {}),
            turn_id=turn_id,
        )
        try:
            conn_ctx = self._conn_ctx()
            if conn_ctx is None:
                return rec
            with conn_ctx as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO {self.table}
                        (id, session_id, turn_id, intent, score, spans, entities, intent_raw, predict_raw, model_version, ts)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                        """,
                        (
                            uuid.uuid4(),
                            session_id,
                            turn_id,
                            rec.intent,
                            rec.score,
                            json.dumps(_json_safe(rec.spans)),
                            json.dumps(_json_safe(rec.entities)),
                            json.dumps(_json_safe(rec.intent_raw)),
                            json.dumps(_json_safe(rec.predict_raw)),
                            model_version or "",
                        ),
                    )
                    if self.ttl_days:
                        cur.execute(
                            f"DELETE FROM {self.table} WHERE ts < NOW() - INTERVAL '{int(self.ttl_days)} days'"
                        )
        except Exception:
            self._log_pg_error("Failed to record intent into Postgres")
        return rec

    def _rows_to_records(self, rows) -> List[IntentRecord]:
        out: List[IntentRecord] = []
        for r in rows or []:
            try:
                intent_raw = {}
                predict_raw = {}
                if len(r) >= 8:
                    intent, score, spans, entities, turn_id, ts, intent_raw, predict_raw = r[:8]
                else:
                    intent, score, spans, entities, turn_id, ts = r
                out.append(
                    IntentRecord(
                        intent=intent or "",
                        score=float(score or 0.0),
                        spans=spans or [],
                        entities=entities or {},
                        intent_raw=intent_raw or {},
                        predict_raw=predict_raw or {},
                        turn_id=turn_id or 0,
                        ts=ts.timestamp() if hasattr(ts, "timestamp") else time.time(),
                    )
                )
            except Exception:
                continue
        return out

    def last(self, session_id: str) -> Optional[IntentRecord]:
        try:
            conn_ctx = self._conn_ctx()
            if conn_ctx is None:
                return None
            with conn_ctx as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT intent, score, spans, entities, turn_id, ts, intent_raw, predict_raw
                        FROM {self.table}
                        WHERE session_id=%s
                        ORDER BY ts DESC
                        LIMIT 1
                        """,
                        (session_id,),
                    )
                    row = cur.fetchone()
                    return self._rows_to_records([row])[0] if row else None
        except Exception:
            self._log_pg_error("Failed to fetch last intent from Postgres")
            return None

    def history(self, session_id: str, k: int = 10) -> List[IntentRecord]:
        try:
            conn_ctx = self._conn_ctx()
            if conn_ctx is None:
                return []
            with conn_ctx as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT intent, score, spans, entities, turn_id, ts, intent_raw, predict_raw
                        FROM {self.table}
                        WHERE session_id=%s
                        ORDER BY ts DESC
                        LIMIT %s
                        """,
                        (session_id, k),
                    )
                    rows = cur.fetchall() or []
                    recs = self._rows_to_records(rows)
                    return list(reversed(recs))
        except Exception:
            self._log_pg_error("Failed to fetch intent history from Postgres")
            return []


class RedisIntentStore(IntentStoreBase):
    def __init__(self, url: Optional[str] = None, max_history: int = 50, ttl_seconds: Optional[int] = None):
        if not redis:
            raise RuntimeError("redis client not available for RedisIntentStore")
        self.max_history = max_history
        self.ttl = ttl_seconds
        if url:
            self._redis = redis.from_url(url, decode_responses=True)
        else:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "0"))
            self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"intent:{session_id}"

    def record(
        self,
        session_id: str,
        intent: str,
        score: float,
        spans: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[Dict[str, Any]] = None,
        intent_raw: Optional[Dict[str, Any]] = None,
        predict_raw: Optional[Dict[str, Any]] = None,
        turn_id: int = 0,
        model_version: Optional[str] = None,
    ) -> IntentRecord:
        rec = IntentRecord(
            intent=intent or "",
            score=float(score or 0.0),
            spans=list(spans or []),
            entities=dict(entities or {}),
            intent_raw=dict(intent_raw or {}),
            predict_raw=dict(predict_raw or {}),
            turn_id=turn_id,
        )
        try:
            payload = json.dumps(rec.__dict__, default=str)
            self._redis.rpush(self._key(session_id), payload)
            self._redis.ltrim(self._key(session_id), -self.max_history, -1)
            if self.ttl:
                try:
                    self._redis.expire(self._key(session_id), int(self.ttl))
                except Exception:
                    pass
        except Exception:
            logger.exception("Failed to record intent into Redis")
        return rec

    def _load(self, session_id: str, k: int) -> List[IntentRecord]:
        try:
            raw = self._redis.lrange(self._key(session_id), -k, -1) or []
            recs = []
            for item in raw:
                try:
                    obj = json.loads(item)
                    recs.append(
                        IntentRecord(
                            intent=obj.get("intent", ""),
                            score=float(obj.get("score", 0.0)),
                            spans=obj.get("spans") or [],
                            entities=obj.get("entities") or {},
                            intent_raw=obj.get("intent_raw") or {},
                            predict_raw=obj.get("predict_raw") or {},
                            turn_id=int(obj.get("turn_id", 0)),
                            ts=float(obj.get("ts", time.time())),
                        )
                    )
                except Exception:
                    continue
            return recs
        except Exception:
            logger.exception("Failed to load intents from Redis")
            return []

    def last(self, session_id: str) -> Optional[IntentRecord]:
        recs = self._load(session_id, 1)
        return recs[-1] if recs else None

    def history(self, session_id: str, k: int = 10) -> List[IntentRecord]:
        return self._load(session_id, k)


def create_intent_store() -> IntentStoreBase:
    """Factory: prefer Postgres, then Redis, then in-memory."""
    dsn = os.getenv("PG_DSN")
    if dsn and psycopg:
        try:
            ttl_env = os.getenv("INTENT_TTL_DAYS")
            ttl_days = int(ttl_env) if ttl_env and ttl_env.isdigit() else None
            return PostgresIntentStore(dsn=dsn, table=os.getenv("INTENT_TABLE", "intents"), ttl_days=ttl_days)
        except Exception:
            logger.exception("Falling back from PostgresIntentStore")
    redis_url = os.getenv("REDIS_URL")
    if redis_url and redis:
        try:
            ttl_env = os.getenv("INTENT_TTL_SECONDS")
            ttl_seconds = int(ttl_env) if ttl_env and ttl_env.isdigit() else None
            return RedisIntentStore(redis_url, ttl_seconds=ttl_seconds)
        except Exception:
            logger.exception("Falling back from RedisIntentStore")
    return InMemoryIntentStore()
