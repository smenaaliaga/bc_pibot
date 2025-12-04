"""LangGraph-backed MemoryInterface adapter (with fallbacks)."""

from __future__ import annotations

import os
import uuid
import datetime
import logging
import time
from typing import Optional, List, Dict, Tuple, Any
from contextlib import nullcontext

logger = logging.getLogger(__name__)

from .pg_logging import throttled_pg_log

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
    from psycopg_pool import ConnectionPool  # type: ignore
except Exception:
    ConnectionPool = None
try:
    from memory_handler.response_diversity import ResponseDiversityManager  # type: ignore
except Exception:
    ResponseDiversityManager = None  # type: ignore

LANGGRAPH_AVAILABLE = bool(PostgresSaver or MemorySaver)


class LangChainMemoryAdapter:
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
        # basic gauges/counters
        self._metrics: Dict[str, float] = {"memory_fallback_used": 0, "diversity_hits": 0}
        self._pool_tag = ""
        # diversity manager config
        self._div_min_turns = int(os.getenv("DIVERSITY_MIN_TURNS", "3"))
        self._div_min_len = int(os.getenv("DIVERSITY_MIN_LEN", "20"))
        self._div_thresh = float(os.getenv("DIVERSITY_SIM_THRESHOLD", "0.6"))
        self._div_mgr = self._init_diversity_manager()
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available; LangChainMemoryAdapter will use in-process fallback.")
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
        self._init_pg_sidecars()

    # --- Postgres side storage for quick history/facts ---
    def _init_pg_sidecars(self):
        if not psycopg or not self.pg_dsn:
            return
        # init connection pool if possible
        if ConnectionPool is not None:
            try:
                self._pool = ConnectionPool(
                    conninfo=self.pg_dsn,
                    max_size=int(os.getenv("PG_POOL_SIZE", "5")),
                    open=False,
                )
                try:
                    # Explicitly open the pool since we set open=False
                    self._pool.open()
                except Exception:
                    self._log_pg_error("Failed to open psycopg pool; using direct connects", op="pool_open")
                    self._pool = None
                self._pool_dsn = self.pg_dsn
            except Exception:
                self._log_pg_error("Could not create psycopg pool; falling back to direct connect", op="pool_init")
                self._pool = None
                self._pool_dsn = None
            if self._pool_dsn:
                try:
                    import hashlib

                    self._pool_tag = hashlib.sha1(self._pool_dsn.encode()).hexdigest()[:8]
                except Exception:
                    self._pool_tag = ""
        if self._require_pg and not self._pool_dsn:
            raise RuntimeError("REQUIRE_PG_MEMORY habilitado pero no se pudo abrir pool de Postgres para memoria.")
        try:
            with psycopg.connect(self.pg_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS short_term_turns (
                            id UUID PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            content TEXT,
                            ts TIMESTAMPTZ DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_short_term_turns_session_ts
                            ON short_term_turns(session_id, ts DESC);
                        """
                    )
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS session_facts (
                            session_id TEXT NOT NULL,
                            fact_key TEXT NOT NULL,
                            fact_value TEXT,
                            ts TIMESTAMPTZ DEFAULT NOW(),
                            PRIMARY KEY (session_id, fact_key)
                        );
                        CREATE INDEX IF NOT EXISTS idx_session_facts_session ON session_facts(session_id);
                        """
                    )
                # TTL cleanup via trigger-free simple cron approach: we store ts and will delete older rows manually elsewhere if needed
        except Exception:
            self._log_pg_error("Could not init Postgres sidecar tables for memory", op="init_sidecars")
            return

    def _pg_conn_ctx(self):
        if not psycopg or not self.pg_dsn:
            return None
        # use pool if available (returns a context manager)
        if self._pool:
            try:
                if getattr(self._pool, "closed", False):
                    try:
                        self._pool.open()
                    except Exception:
                        self._log_pg_error("Pool was closed; reopening failed", op="pool_reopen", table="short_term_turns")
                        self._pool = None
                        self._pool_dsn = None
                        return None
                import time as _t

                start = _t.time()
                ctx = self._pool.connection()
                elapsed = (_t.time() - start) * 1000.0
                logger.info(
                    "metric pg_pool_acquire_time_ms=%.2f",
                    elapsed,
                    extra={"op": "pool_connection", "pool": self._pool_tag},
                )
                return ctx
            except Exception:
                self._log_pg_error("Pool connection failed; dropping pool", op="pool_connection")
                try:
                    self._pool.close()
                except Exception:
                    pass
                self._pool = None
        # fallback direct connection with simple retry wrapped as context manager
        for _ in range(2):
            try:
                return nullcontext(psycopg.connect(self.pg_dsn))
            except Exception:
                self._log_pg_error("Could not open Postgres connection for memory; retrying", op="connect")
                time.sleep(0.2)
        return None

    def _init_saver(self):
        # Prefer PostgresSaver if available, otherwise MemorySaver
        if PostgresSaver:
            try:
                candidate = PostgresSaver.from_conn_string(self.pg_dsn)
                if hasattr(candidate, "__enter__"):
                    entered = candidate.__enter__()
                    self._cleanup = candidate.__exit__
                    saver = entered
                else:
                    saver = candidate
                self._ensure_tables(saver)
                return saver
            except Exception:
                logger.exception("PostgresSaver init failed; trying MemorySaver fallback")
        if MemorySaver:
            try:
                saver = MemorySaver()
                return saver
            except Exception:
                logger.exception("MemorySaver init failed")
        return None

    def __del__(self):
        try:
            if self._cleanup:
                self._cleanup(None, None, None)
        except Exception:
            pass
        try:
            if self._pool:
                self._pool.close()
        except Exception:
            pass

    def _init_diversity_manager(self):
        if not ResponseDiversityManager:
            return None
        embeddings_obj = None
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore

            openai_model = os.getenv("OPENAI_EMBEDDINGS_MODEL") or os.getenv("OPENAI_MODEL")
            if openai_model:
                embeddings_obj = OpenAIEmbeddings(model=openai_model)
        except Exception:
            embeddings_obj = None

        try:
            return ResponseDiversityManager(
                embeddings=embeddings_obj,
                llm=None,
                similarity_threshold=self._div_thresh,
                max_history=50,
            )
        except Exception:
            return None

    def _ensure_tables(self, saver) -> None:
        try:
            if hasattr(saver, "setup"):
                saver.setup()
            elif hasattr(saver, "create_tables"):
                saver.create_tables()
            elif hasattr(saver, "ensure_tables"):
                saver.ensure_tables()
        except Exception:
            logger.exception("ensure_tables failed for PostgresSaver")

    # --- Recent facts (lightweight extraction) ---
    def _extract_facts(self, text: str) -> Dict[str, str]:
        facts: Dict[str, str] = {}
        low = (text or "").lower()
        # nombre
        import re
        m = re.search(r"(mi nombre es|me llamo)\s+([a-záéíóúñü]+)", low)
        if m:
            facts["nombre"] = m.group(2).strip()
        m = re.search(r"estudi[ée]\s+([a-záéíóúñü\s]+)", low)
        if m:
            facts["estudios"] = m.group(1).strip()
        return facts

    def _store_facts_pg(self, session_id: str, facts: Dict[str, str]) -> None:
        if not facts:
            return
        conn = self._pg_conn_ctx()
        if not conn:
            return
        try:
            with conn as c, c.cursor() as cur:
                for k, v in facts.items():
                    cur.execute(
                        """
                        INSERT INTO session_facts(session_id, fact_key, fact_value, ts)
                        VALUES (%s,%s,%s,NOW())
                        ON CONFLICT (session_id, fact_key) DO UPDATE SET fact_value=EXCLUDED.fact_value, ts=EXCLUDED.ts
                        """,
                        (session_id, k, v),
                    )
        except Exception:
            logger.exception("Failed to store facts in Postgres")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _store_turn_pg(self, session_id: str, role: str, text: str) -> None:
        conn = self._pg_conn_ctx()
        if not conn:
            return
        try:
            with conn as c, c.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO short_term_turns(id, session_id, role, content, ts)
                    VALUES (%s,%s,%s,%s,NOW())
                    """,
                    (uuid.uuid4(), session_id, role, text),
                )
                # optional TTL cleanup
                ttl_days = os.getenv("SHORT_TERM_TURNS_TTL_DAYS")
                if ttl_days and ttl_days.isdigit():
                    cur.execute(
                        f"DELETE FROM short_term_turns WHERE ts < NOW() - INTERVAL '{int(ttl_days)} days'"
                    )
        except Exception:
            self._log_pg_error("Failed to store turn in Postgres", op="store_turn", table="short_term_turns")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def get_recent_history(self, session_id: str, max_turns: Optional[int] = None) -> List[Dict[str, str]]:
        max_turns = max_turns or self._max_turns_prompt
        conn = self._pg_conn_ctx()
        if conn:
            try:
                with conn as c, c.cursor() as cur:
                    cur.execute(
                        """
                        SELECT role, content FROM short_term_turns
                        WHERE session_id=%s
                        ORDER BY ts DESC
                        LIMIT %s
                        """,
                        (session_id, max_turns),
                    )
                    rows = cur.fetchall() or []
                    rows = list(reversed(rows))
                    return [{"role": r[0], "content": r[1]} for r in rows]
            except Exception:
                self._log_pg_error("Failed to load recent history from Postgres", op="load_history", table="short_term_turns")
                # simple backoff: drop pool to allow recreation on next call
                try:
                    if self._pool:
                        self._pool.close()
                        self._pool = None
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        # fallback to checkpointer
        msgs = self._load_messages(session_id)
        return msgs[-max_turns:]

    def get_facts(self, session_id: str) -> Dict[str, str]:
        conn = self._pg_conn_ctx()
        if conn:
            try:
                with conn as c, c.cursor() as cur:
                    cur.execute(
                        """
                        SELECT fact_key, fact_value FROM session_facts
                        WHERE session_id=%s
                        """,
                        (session_id,),
                    )
                    rows = cur.fetchall() or []
                    return {r[0]: r[1] for r in rows}
            except Exception:
                self._log_pg_error("Failed to load facts from Postgres", op="load_facts", table="session_facts")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        return self._local_facts.get(session_id, {})

    def _config(self, session_id: str) -> Dict[str, Dict[str, str]]:
        return {"configurable": {"thread_id": session_id, "checkpoint_ns": self._checkpoint_ns}}

    def _load_messages(self, session_id: str) -> List[Dict[str, str]]:
        try:
            if self.saver is None:
                return list(self._fallback)
            chk = self.saver.get(config=self._config(session_id)) or {}
            return (chk.get("channel_values", {}) or {}).get("messages", []) or []
        except Exception:
            self._log_pg_error("load_messages failed (PostgresSaver)", op="load_messages", table="checkpoints")
            try:
                self._metrics["memory_fallback_used"] += 1
                return list(self._fallback)
            except Exception:
                return []

    def _save_messages(self, session_id: str, messages: List[Dict[str, str]]) -> None:
        try:
            if self.saver is None:
                self._fallback = messages[-50:]
                return
            now = datetime.datetime.now(datetime.timezone.utc)
            ts = now.isoformat()
            version_str = str(int(now.timestamp() * 1_000))
            # maintain a running summary every N turns to avoid bloat
            summary = ""
            if len(messages) % self._summary_every == 0:
                try:
                    import re
                    recent = messages[-self._summary_every :]
                    sentences = []
                    for m in recent:
                        s = re.split(r"(?<=[.!?])\\s+", m.get("content", "").strip())
                        if s:
                            sentences.append(f"{m.get('role')}: {s[0]}")
                    summary = " | ".join(sentences)[-2000:]
                except Exception:
                    summary = ""
            checkpoint = {
                "v": 1,
                "id": uuid.uuid4().hex,
                "ts": ts,
                "channel_values": {"messages": messages[-self._max_turns_prompt:], "summary": summary},
                "channel_versions": {"messages": version_str, "summary": version_str},
                "versions_seen": {},
                "updated_channels": ["messages", "summary"],
            }
            self.saver.put(
                config=self._config(session_id),
                checkpoint=checkpoint,
                metadata={"source": "pibot", "step": len(messages), "writes": {}},
                new_versions={"messages": version_str},
            )
        except Exception:
            self._log_pg_error("save_messages failed (PostgresSaver)", op="save_messages", table="checkpoints")
            try:
                self._fallback = messages[-50:]
                self._metrics["memory_fallback_used"] += 1
            except Exception:
                pass

    def on_user_turn(self, session_id: str, text: str) -> None:
        msgs = self._load_messages(session_id)
        msgs.append({"role": "user", "content": text})
        self._save_messages(session_id, msgs)
        self._store_turn_pg(session_id, "user", text)
        facts = self._extract_facts(text)
        if facts:
            self._local_facts.setdefault(session_id, {}).update(facts)
            self._store_facts_pg(session_id, facts)

    def on_assistant_turn(self, session_id: str, text: str) -> None:
        msgs = self._load_messages(session_id)
        msgs.append({"role": "assistant", "content": text})
        self._save_messages(session_id, msgs)

    def diversity_check(self, session_id: str, candidate: str) -> Tuple[bool, float]:
        """Simple redundancy check using last 10 assistant messages from memory."""
        try:
            history = self._load_messages(session_id)
            texts = [m.get("content", "") for m in history if m.get("role") == "assistant"]
            min_turns = self._div_min_turns
            min_len = self._div_min_len
            sim_threshold = self._div_thresh
            # Avoid triggering redundancy on the first replies or very short text
            if not texts or len(texts) < min_turns:
                return False, 0.0
            cand_raw = (candidate or "").strip()
            if not cand_raw or len(cand_raw) < min_len:
                return False, 0.0

            # If the diversity manager has no history (e.g. on cold start) seed it from stored turns
            if self._div_mgr and getattr(self._div_mgr, "_assistant_texts", []) == []:
                try:
                    for prev in texts[-10:]:
                        self._div_mgr.register_response(prev)
                except Exception:
                    pass

            if self._div_mgr:
                try:
                    is_red, sim = self._div_mgr.check_redundancy(cand_raw)
                    if is_red:
                        self._metrics["diversity_hits"] += 1
                        return True, float(sim)
                except Exception:
                    pass

            cand = cand_raw.lower()

            def tokset(s: str) -> set:
                import re
                tokens = re.findall(r"[a-zケゼヴИカヵ]+", s.lower())
                stop = {'el','la','los','las','de','del','en','y','o','u','que','es','un','una','por','para','a','al','su','sus','con','se','lo','como','sobre'}
                return {t for t in tokens if t not in stop}

            cand_set = tokset(cand)
            if not cand_set:
                return False, 0.0
            best_sim = 0.0
            for prev in reversed(texts[-10:]):
                p = (prev or '').strip().lower()
                if not p:
                    continue
                if p in cand or cand in p:
                    return True, 1.0
                p_set = tokset(p)
                if not p_set:
                    continue
                inter = len(cand_set & p_set)
                union = len(cand_set | p_set) or 1
                sim = inter / union
                if sim > best_sim:
                    best_sim = sim
            return (best_sim >= sim_threshold), float(best_sim)
        except Exception:
            self._log_pg_error("diversity_check failed (PostgresSaver)", op="diversity_check")
            return False, 0.0

    def diversity_maybe_rewrite(self, session_id: str, candidate: str, user_msg: str) -> str:
        return candidate

    def diversity_register(self, session_id: str, text: str) -> None:
        self.on_assistant_turn(session_id, text)
        try:
            if self._div_mgr:
                self._div_mgr.register_response(text)
        except Exception:
            pass

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
