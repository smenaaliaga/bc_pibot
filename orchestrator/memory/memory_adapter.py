"""MemoryInterface adapter (prefiere Postgres/Redis; fallback local)."""

from __future__ import annotations

import atexit
import os
import uuid
import datetime
import logging
import json
from typing import Optional, List, Dict, Tuple, Any
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
    from psycopg.types.json import Json  # type: ignore[attr-defined]
except Exception:
    Json = None  # type: ignore
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
        self._fallback_turns: Dict[str, List[Dict[str, Any]]] = {}
        self._max_fallback_turns = int(os.getenv("MEMORY_MAX_TURNS_STORE", "200"))
        self._fallback_checkpoints: Dict[str, List[Dict[str, Any]]] = {}
        self._max_fallback_checkpoints = int(os.getenv("MEMORY_MAX_CHECKPOINTS", "10"))
        self._checkpoint_ns = os.getenv("LANGGRAPH_CHECKPOINT_NS", "memory")
        self._summary_every = int(os.getenv("MEMORY_SUMMARY_EVERY", "5"))
        self._max_turns_prompt = int(os.getenv("MEMORY_MAX_TURNS_PROMPT", "8"))
        self._local_facts: Dict[str, Dict[str, str]] = {}
        self._pool = None
        self._pool_dsn = None
        self._pool_open = False
        self._pool_cleanup_registered = False
        self._turns_table_ready = False
        self._pg_log_state: Dict[str, Any] = {}
        self._pg_err_period = float(os.getenv("PG_ERROR_LOG_PERIOD", "60"))
        self._auto_setup = os.getenv("LANGGRAPH_AUTO_SETUP", "1").lower() not in ("0", "false", "no")
        layout_override = os.getenv("MEMORY_FACTS_LAYOUT", "").strip().lower()
        self._facts_layout_override = layout_override if layout_override in {"json", "kv"} else None
        self._facts_layout: Optional[str] = None
        # basic gauges/counters
        self._metrics: Dict[str, float] = {
            "memory_fallback_used": 0,
            "diversity_hits": 0,
            "turns_fallback_used": 0,
        }
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
                self._pool = ConnectionPool(self.pg_dsn, kwargs=pool_kwargs, open=False)
                self._pool_dsn = self.pg_dsn
                self._pool.open()
                self._pool_open = True
                if not self._pool_cleanup_registered:
                    try:
                        atexit.register(self._close_pool)
                        self._pool_cleanup_registered = True
                    except Exception:
                        logger.debug("No se pudo registrar cleanup para ConnectionPool", exc_info=True)
            except Exception as e:
                self._log_pg_error("Error creando pool: %s" % e, op="pool", table="psycopg_pool")
                self._pool = None
                self._pool_open = False
        elif not self._pool_open:
            try:
                self._pool.open()
                self._pool_open = True
            except Exception as e:
                self._log_pg_error("Error reabriendo pool: %s" % e, op="pool", table="psycopg_pool")
                self._pool_open = False
        return self._pool

    def _close_pool(self) -> None:
        pool = self._pool
        if not pool:
            return
        try:
            pool.close()
            wait_close = getattr(pool, "wait_close", None)
            if callable(wait_close):
                try:
                    wait_close()
                except Exception:
                    logger.debug("wait_close falló para ConnectionPool", exc_info=True)
        except Exception:
            logger.debug("Error cerrando ConnectionPool", exc_info=True)
        finally:
            self._pool = None
            self._pool_open = False

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

    def _normalize_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not metadata:
            return {}
        normalized: Dict[str, Any] = {}
        for key, value in metadata.items():
            key_str = str(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key_str] = value
            else:
                normalized[key_str] = str(value)
        return normalized

    def _append_fallback_turn(self, session_id: str, payload: Dict[str, Any]) -> None:
        bucket = self._fallback_turns.setdefault(session_id, [])
        bucket.append(payload)
        if len(bucket) > self._max_fallback_turns:
            self._fallback_turns[session_id] = bucket[-self._max_fallback_turns :]

    def _append_checkpoint_cache(self, session_id: str, payload: Dict[str, Any]) -> None:
        bucket = self._fallback_checkpoints.setdefault(session_id, [])
        bucket.append(payload)
        if len(bucket) > self._max_fallback_checkpoints:
            self._fallback_checkpoints[session_id] = bucket[-self._max_fallback_checkpoints :]

    def _saver_config(self, session_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": session_id, "checkpoint_ns": self._checkpoint_ns}}

    def _json_param(self, value: Dict[str, Any]) -> Any:
        value = value or {}
        if Json is not None:
            try:
                return Json(value)
            except Exception:
                logger.debug("Json adapter failed; falling back to dumps", exc_info=True)
        try:
            return json.dumps(value)
        except Exception:
            return "{}"

    def _ensure_session_turns_table(self) -> bool:
        if self._turns_table_ready:
            return True
        pool = self._conn_pool()
        if not pool:
            return False
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS session_turns (
                            session_id TEXT NOT NULL,
                            turn_id BIGSERIAL PRIMARY KEY,
                            role TEXT NOT NULL,
                            content TEXT,
                            metadata JSONB,
                            ts TIMESTAMPTZ DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_session_turns_session_ts ON session_turns(session_id, ts DESC)"
                    )
                conn.commit()
            self._turns_table_ready = True
            return True
        except Exception as e:
            self._log_pg_error(
                "No se pudo asegurar tabla session_turns: %s" % e,
                op="ensure_table",
                table="session_turns",
            )
        return False

    def _persist_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any],
        ts: datetime.datetime,
    ) -> bool:
        pool = self._conn_pool()
        if not pool:
            return False
        if not self._ensure_session_turns_table():
            return False
        metadata_payload = metadata or {}
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO session_turns(session_id, role, content, metadata, ts)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (session_id, role, content, self._json_param(metadata_payload), ts),
                    )
                conn.commit()
            return True
        except Exception as e:
            self._log_pg_error(
                "Error guardando turno: %s" % e,
                session_id=session_id,
                op="turn_write",
                table="session_turns",
            )
        return False

    def _parse_ts_filter(self, since_ts: Optional[Any]) -> Optional[datetime.datetime]:
        if since_ts is None:
            return None
        if isinstance(since_ts, datetime.datetime):
            return since_ts
        if isinstance(since_ts, (int, float)):
            return datetime.datetime.fromtimestamp(since_ts, datetime.timezone.utc)
        if isinstance(since_ts, str):
            try:
                return datetime.datetime.fromisoformat(since_ts)
            except ValueError:
                return None
        return None

    def _read_turns_pg(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
        since_ts: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        pool = self._conn_pool()
        if not pool:
            return []
        if not self._ensure_session_turns_table():
            return []
        ts_filter = self._parse_ts_filter(since_ts)
        query = [
            "SELECT role, content, metadata, ts FROM session_turns WHERE session_id=%s",
        ]
        params: List[Any] = [session_id]
        if ts_filter is not None:
            query.append("AND ts >= %s")
            params.append(ts_filter)
        query.append("ORDER BY ts DESC")
        if limit:
            query.append("LIMIT %s")
            params.append(limit)
        sql = " ".join(query)
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, tuple(params))
                    rows = cur.fetchall() or []
        except Exception as e:
            self._log_pg_error(
                "Error leyendo turnos: %s" % e,
                session_id=session_id,
                op="turn_read",
                table="session_turns",
            )
            return []
        turns: List[Dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                role = row.get("role", "")
                content = row.get("content", "")
                metadata = row.get("metadata") or {}
                ts_value = row.get("ts")
            else:
                try:
                    role, content, metadata, ts_value = row
                except Exception:
                    continue
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            turns.append(
                {
                    "role": str(role or ""),
                    "content": "" if content is None else str(content),
                    "metadata": metadata or {},
                    "ts": ts_value.isoformat() if hasattr(ts_value, "isoformat") else ts_value,
                }
            )
        return list(reversed(turns))

    def _fallback_turn_rows(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        bucket = self._fallback_turns.get(session_id, [])
        if not bucket:
            return []
        subset = bucket[-limit:] if limit else list(bucket)
        return [dict(item) for item in subset]

    def _record_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not session_id:
            session_id = uuid.uuid4().hex
        if not role or not content:
            return
        ts = datetime.datetime.now(datetime.timezone.utc)
        normalized_metadata = self._normalize_metadata(metadata)
        payload = {
            "role": role,
            "content": str(content),
            "metadata": normalized_metadata,
            "ts": ts.isoformat(),
        }
        self._append_fallback_turn(session_id, payload)
        if not self._persist_turn(session_id, role, str(content), normalized_metadata, ts):
            self._metrics["turns_fallback_used"] = self._metrics.get("turns_fallback_used", 0) + 1

    def save_checkpoint(
        self,
        session_id: str,
        checkpoint: Dict[str, Any],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not session_id or not checkpoint:
            return False
        normalized_metadata = self._normalize_metadata(metadata)
        entry = {
            "checkpoint": checkpoint,
            "metadata": {
                **normalized_metadata,
                "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        }
        self._append_checkpoint_cache(session_id, entry)
        saver = getattr(self, "saver", None)
        if saver and hasattr(saver, "put"):
            config = self._saver_config(session_id)
            payload = {
                "id": '_'.join([session_id, uuid.uuid4().hex]),
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "values": checkpoint,
                "channel_values": {},
            }
            try:
                channel_versions: Dict[str, Any] = {}
                saver.put(config, payload, normalized_metadata, channel_versions)  # type: ignore[arg-type]
            except Exception:
                logger.debug("save_checkpoint fall back to cache", exc_info=True)
        return True

    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None
        saver = getattr(self, "saver", None)
        if saver and hasattr(saver, "get"):
            config = self._saver_config(session_id)
            try:
                data = saver.get(config)  # type: ignore[call-arg]
            except Exception:
                logger.debug("load_checkpoint saver.get failed", exc_info=True)
            else:
                if isinstance(data, dict):
                    checkpoint_payload = data.get("checkpoint") or data.get("values") or data
                    metadata = data.get("metadata") or {}
                    return {"checkpoint": checkpoint_payload, "metadata": metadata}
        cache = self._fallback_checkpoints.get(session_id)
        if cache:
            return cache[-1]
        return None

    def get_recent_turns(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
        since_ts: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        if not session_id:
            return []
        turns = self._read_turns_pg(session_id, limit=limit, since_ts=since_ts)
        if turns:
            return turns if not limit else turns[-limit:]
        return self._fallback_turn_rows(session_id, limit=limit)

    def get_window_for_llm(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        limit = max_turns or self._max_turns_prompt
        turns = self.get_recent_turns(session_id, limit=limit)
        if not turns:
            return []
        window = []
        for turn in turns[-limit:]:
            role = str(turn.get("role", ""))
            content = str(turn.get("content", ""))
            if role and content:
                window.append({"role": role, "content": content})
        return window[-limit:]

    # --- Facts API ---------------------------------------------------------
    def set_facts(self, session_id: str, facts: Dict[str, str]) -> None:
        if not session_id or not facts:
            return
        normalized = {str(k): self._serialize_fact_value(v) for k, v in facts.items()}
        self._local_facts.setdefault(session_id, {}).update(normalized)
        pool = self._conn_pool()
        if not pool:
            return
        layout = self._ensure_facts_layout()
        if not layout:
            return
        try:
            if layout == "kv":
                self._set_facts_kv(pool, session_id, normalized)
            else:
                self._set_facts_json(pool, session_id, normalized)
        except Exception as e:
            self._log_pg_error("Error guardando facts: %s" % e, session_id=session_id, op="facts_write", table="session_facts")

    @staticmethod
    def _serialize_fact_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

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
    def on_user_turn(self, session_id: str, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record the latest user message across backends."""
        if not message:
            return
        self._record_turn(session_id, "user", message, metadata=metadata)

    def on_assistant_turn(
        self,
        session_id: str,
        message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not message:
            return
        self._record_turn(session_id, "assistant", message, metadata=metadata)

    def get_history_for_llm(self, session_id: str) -> List[Dict[str, str]]:
        try:
            window = self.get_window_for_llm(session_id, max_turns=self._max_turns_prompt)
            if window:
                return window
        except Exception as e:
            logger.debug("get_history_for_llm failed: %s", e)
            self._metrics["memory_fallback_used"] += 1
        return []

    def get_backend_status(self) -> Dict[str, Any]:
        return {
            "using_pg": self._using_pg,
            "require_pg": self._require_pg,
            "saver": type(self.saver).__name__ if self.saver else None,
            "fallback_sessions": len(self._fallback_turns),
            "checkpoint_sessions": len(self._fallback_checkpoints),
            "metrics": self._metrics,
        }


# Compatibilidad
LangChainMemoryAdapter = MemoryAdapter
