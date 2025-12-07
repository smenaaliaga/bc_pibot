"""MemoryInterface adapter (prefiere Postgres/Redis; fallback local)."""

from __future__ import annotations

import os
import uuid
import datetime
import logging
import time
from typing import Optional, List, Dict, Tuple, Any
from contextlib import nullcontext
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

from orchestrator.utils.pg_logging import throttled_pg_log

try:
    from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
except Exception:
    PostgresSaver = None

try:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore
except Exception:
    MemorySaver = None

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None
try:
    from psycopg.rows import dict_row  # type: ignore
except Exception:
    dict_row = None
try:
    from psycopg_pool import ConnectionPool  # type: ignore
except Exception:
    ConnectionPool = None
try:
    from memory_handler.response_diversity import ResponseDiversityManager  # type: ignore
except Exception:
    ResponseDiversityManager = None  # type: ignore

LANGGRAPH_AVAILABLE = bool(PostgresSaver or MemorySaver)


class MemoryAdapter:
    """Adapter that exposes a minimal MemoryInterface using LangGraph checkpointers."""

    def __init__(self, pg_dsn: Optional[str] = None):
        self.pg_dsn = pg_dsn or os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/pibot")
        self._require_pg = os.getenv("REQUIRE_PG_MEMORY", "0").lower() in ("1", "true", "yes", "on")
        self._cleanup = None
        self._fallback: List[Dict[str, str]] = []
        self._checkpoint_ns = os.getenv("LANGGRAPH_CHECKPOINT_NS", "memory")
        self._summary_every = int(os.getenv("MEMORY_SUMMARY_EVERY", "5"))
        self._max_turns_prompt = int(os.getenv("MEMORY_MAX_TURNS_PROMPT", "8"))
        self._local_facts: Dict[str, Dict[str, str]] = {}
        self._pool = None
        self._pool_dsn = None
        self._pg_log_state: Dict[str, Any] = {}
        self._pg_err_period = float(os.getenv("PG_ERROR_LOG_PERIOD", "60"))
        self._auto_setup = os.getenv("LANGGRAPH_AUTO_SETUP", "1").lower() not in ("0", "false", "no")
        layout_override = os.getenv("MEMORY_FACTS_LAYOUT", "").strip().lower()
        self._facts_layout_override = layout_override if layout_override in {"json", "kv"} else None
        self._facts_layout: Optional[str] = None
        # basic gauges/counters
        self._metrics: Dict[str, float] = {"memory_fallback_used": 0, "diversity_hits": 0}
        parsed = urlparse(self.pg_dsn)
        host = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port else ""
        self._pool_tag = f"{host}{port}" if host or port else ""
        # diversity manager config
        self._div_min_turns = int(os.getenv("DIVERSITY_MIN_TURNS", "3"))
        self._div_min_len = int(os.getenv("DIVERSITY_MIN_LEN", "20"))
        self._div_thresh = float(os.getenv("DIVERSITY_SIM_THRESHOLD", "0.6"))
        self._div_mgr = self._init_diversity_manager()
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available; MemoryAdapter usará fallback en proceso.")
            self.saver = None
        else:
            self.saver = self._init_saver()
        self._using_pg = bool(PostgresSaver and self.saver and isinstance(self.saver, PostgresSaver))
        if self._require_pg and not self._using_pg:
            raise RuntimeError("REQUIRE_PG_MEMORY habilitado pero PostgresSaver no está disponible o falló la inicialización.")
        logger.info(
            "[MEMORY_INIT] saver=%s using_pg=%s require_pg=%s dsn=%s",
            type(self.saver).__name__ if self.saver else "None",
            self._using_pg,
            self._require_pg,
            self.pg_dsn,
        )

    def _init_diversity_manager(self):
        if ResponseDiversityManager is None:
            return None
        try:
            return ResponseDiversityManager(
                min_turns=self._div_min_turns,
                min_len=self._div_min_len,
                similarity_threshold=self._div_thresh,
            )
        except Exception:
            logger.debug("No se pudo inicializar ResponseDiversityManager", exc_info=True)
            return None

    def _init_saver(self):
        if PostgresSaver and self.pg_dsn:
            conn_resource = None
            try:
                if ConnectionPool:
                    pool = self._conn_pool()
                    if pool is None:
                        raise RuntimeError("No se pudo crear ConnectionPool para PostgresSaver")
                    conn_resource = pool
                elif psycopg:
                    conn_kwargs = {"autocommit": True, "prepare_threshold": 0}
                    if dict_row:
                        conn_kwargs["row_factory"] = dict_row
                    conn_resource = psycopg.connect(self.pg_dsn, **conn_kwargs)
                if conn_resource:
                    saver = PostgresSaver(conn_resource)
                    if self._auto_setup and hasattr(saver, "setup"):
                        try:
                            saver.setup()
                        except Exception:
                            logger.debug("LangGraph PostgresSaver setup failed", exc_info=True)
                    return saver
                raise RuntimeError("PostgresSaver requiere psycopg o ConnectionPool disponible")
            except Exception as e:
                logger.warning("PostgresSaver init failed: %s", e)
                if self._require_pg:
                    raise
        if MemorySaver:
            return MemorySaver()
        return None

    def _conn_pool(self) -> Optional[Any]:
        if not psycopg or not ConnectionPool:
            return None
        if self._pool is None or self._pool_dsn != self.pg_dsn:
            try:
                pool_kwargs: Dict[str, Any] = {"autocommit": True, "prepare_threshold": 0}
                if dict_row:
                    pool_kwargs["row_factory"] = dict_row
                self._pool = ConnectionPool(self.pg_dsn, kwargs=pool_kwargs)
                self._pool_dsn = self.pg_dsn
            except Exception as e:
                self._log_pg_error("Error creando pool: %s" % e, op="pool", table="psycopg_pool")
                self._pool = None
        return self._pool

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
            table=table,
            pool=self._pool_tag,
            period=self._pg_err_period,
        )

    def _ensure_facts_layout(self) -> Optional[str]:
        if self._facts_layout:
            return self._facts_layout
        layout = self._facts_layout_override or self._detect_facts_layout()
        if layout is None:
            layout = "json"
        if self._ensure_session_facts_table(layout):
            self._facts_layout = layout
            return layout
        return None

    def _detect_facts_layout(self) -> Optional[str]:
        pool = self._conn_pool()
        if not pool:
            return None
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'session_facts'
                        """
                    )
                    columns: set[str] = set()
                    for row in cur.fetchall():
                        if not row:
                            continue
                        col_name = None
                        if isinstance(row, dict):
                            col_name = row.get("column_name")
                        else:
                            try:
                                col_name = row[0]
                            except Exception:
                                col_name = None
                        if col_name:
                            columns.add(str(col_name).lower())
            if not columns:
                return None
            if "facts" in columns:
                return "json"
            if {"fact_key", "fact_value"}.issubset(columns):
                return "kv"
        except Exception as e:
            self._log_pg_error("No se pudo detectar esquema de session_facts: %s" % e, op="detect_schema", table="session_facts")
        return None

    def _ensure_session_facts_table(self, layout: str) -> bool:
        pool = self._conn_pool()
        if not pool:
            return False
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    if layout == "json":
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS session_facts (
                                session_id TEXT PRIMARY KEY,
                                facts JSONB,
                                updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
                            )
                            """
                        )
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_session_facts_updated ON session_facts(updated_at DESC)"
                        )
                    else:
                        cur.execute(
                            """
                            CREATE TABLE IF NOT EXISTS session_facts (
                                session_id TEXT NOT NULL,
                                fact_key TEXT NOT NULL,
                                fact_value TEXT,
                                ts TIMESTAMPTZ DEFAULT NOW(),
                                PRIMARY KEY (session_id, fact_key)
                            )
                            """
                        )
                        cur.execute(
                            "CREATE INDEX IF NOT EXISTS idx_session_facts_session ON session_facts(session_id)"
                        )
                conn.commit()
            return True
        except Exception as e:
            self._log_pg_error("No se pudo asegurar tabla session_facts: %s" % e, op="ensure_table", table="session_facts")
        return False

    def _set_facts_json(self, pool: Any, session_id: str, facts: Dict[str, str]) -> None:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_facts(session_id, facts, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (session_id) DO UPDATE
                        SET facts = EXCLUDED.facts,
                            updated_at = NOW()
                    """,
                    (session_id, facts),
                )
            conn.commit()

    def _set_facts_kv(self, pool: Any, session_id: str, facts: Dict[str, str]) -> None:
        rows = [(session_id, key, str(value)) for key, value in facts.items()]
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM session_facts WHERE session_id=%s", (session_id,))
                if rows:
                    cur.executemany(
                        """
                        INSERT INTO session_facts(session_id, fact_key, fact_value, ts)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (session_id, fact_key) DO UPDATE
                            SET fact_value = EXCLUDED.fact_value,
                                ts = NOW()
                        """,
                        rows,
                    )
            conn.commit()

    def _read_facts_json(self, pool: Any, session_id: str) -> Dict[str, str]:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT facts FROM session_facts WHERE session_id=%s", (session_id,))
                row = cur.fetchone()
                if row and row[0]:
                    return dict(row[0])
        return {}

    def _read_facts_kv(self, pool: Any, session_id: str) -> Dict[str, str]:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT fact_key, fact_value FROM session_facts WHERE session_id=%s",
                    (session_id,),
                )
                rows = cur.fetchall()
                if rows:
                    return {str(key): ("" if value is None else str(value)) for key, value in rows}
        return {}

    # --- Facts API ---------------------------------------------------------
    def set_facts(self, session_id: str, facts: Dict[str, str]) -> None:
        if not session_id or not facts:
            return
        self._local_facts.setdefault(session_id, {}).update({k: str(v) for k, v in facts.items()})
        pool = self._conn_pool()
        if not pool:
            return
        layout = self._ensure_facts_layout()
        if not layout:
            return
        try:
            if layout == "kv":
                self._set_facts_kv(pool, session_id, facts)
            else:
                self._set_facts_json(pool, session_id, facts)
        except Exception as e:
            self._log_pg_error("Error guardando facts: %s" % e, session_id=session_id, op="facts_write", table="session_facts")

    def get_facts(self, session_id: str) -> Dict[str, str]:
        if not session_id:
            return {}
        if session_id in self._local_facts:
            return self._local_facts[session_id]
        pool = self._conn_pool()
        if not pool:
            return {}
        layout = self._ensure_facts_layout()
        if not layout:
            return {}
        try:
            if layout == "kv":
                return self._read_facts_kv(pool, session_id)
            return self._read_facts_json(pool, session_id)
        except Exception as e:
            self._log_pg_error("Error leyendo facts: %s" % e, session_id=session_id, op="facts_read", table="session_facts")
        return {}

    # --- Turns API (minimal) ----------------------------------------------
    def on_user_turn(self, session_id: str, message: str) -> None:
        """Append user message to fallback memory if no saver is available."""
        if not session_id:
            session_id = uuid.uuid4().hex
        self._fallback.append({"role": "user", "content": message, "ts": datetime.datetime.utcnow().isoformat()})
        if len(self._fallback) > 200:
            self._fallback = self._fallback[-200:]

    def on_assistant_turn(self, session_id: str, message: str) -> None:
        if not session_id:
            session_id = uuid.uuid4().hex
        self._fallback.append({"role": "assistant", "content": message, "ts": datetime.datetime.utcnow().isoformat()})
        if len(self._fallback) > 200:
            self._fallback = self._fallback[-200:]

    def get_history_for_llm(self, session_id: str) -> List[Dict[str, str]]:
        if not LANGGRAPH_AVAILABLE or not self.saver:
            return [{"role": m["role"], "content": m["content"]} for m in self._fallback[-self._max_turns_prompt :]]
        try:
            with nullcontext():
                if hasattr(self.saver, "list") and callable(getattr(self.saver, "list", None)):
                    turns = list(self.saver.list(config={"configurable": {"thread_id": session_id}}))
                    turns = turns[-self._max_turns_prompt :]
                    return [{"role": t.get("role", ""), "content": t.get("content", "")} for t in turns]
        except Exception as e:
            logger.debug("get_history_for_llm failed: %s", e)
            self._metrics["memory_fallback_used"] += 1
        return [{"role": m["role"], "content": m["content"]} for m in self._fallback[-self._max_turns_prompt :]]

    def get_backend_status(self) -> Dict[str, Any]:
        return {
            "using_pg": self._using_pg,
            "require_pg": self._require_pg,
            "saver": type(self.saver).__name__ if self.saver else None,
            "fallback_len": len(self._fallback),
            "metrics": self._metrics,
        }


# Compatibilidad
LangChainMemoryAdapter = MemoryAdapter
