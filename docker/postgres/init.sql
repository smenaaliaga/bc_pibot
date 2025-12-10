\set ON_ERROR_STOP on
SELECT 'CREATE DATABASE pibot'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'pibot')\gexec
\connect pibot

-- Extensión pgvector para RAG
CREATE EXTENSION IF NOT EXISTS vector;

-- LangGraph checkpointer
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, key)
);
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    value JSONB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);
CREATE TABLE IF NOT EXISTS checkpoint_migrations (v INTEGER PRIMARY KEY);
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id, checkpoint_ns, checkpoint_id);

-- Short-term sidecar storage
CREATE TABLE IF NOT EXISTS short_term_turns (
    id UUID PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    ts TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_short_term_turns_session_ts ON short_term_turns(session_id, ts DESC);

CREATE TABLE IF NOT EXISTS session_facts (
    session_id TEXT PRIMARY KEY,
    facts JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_session_facts_updated ON session_facts(updated_at DESC);

-- Conversation turn history for MemoryAdapter windowing
CREATE TABLE IF NOT EXISTS session_turns (
    session_id TEXT NOT NULL,
    turn_id BIGSERIAL PRIMARY KEY,
    role TEXT NOT NULL,
    content TEXT,
    metadata JSONB,
    ts TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_session_turns_session_ts ON session_turns(session_id, ts DESC);

-- Intent store
CREATE TABLE IF NOT EXISTS intents (
    id UUID PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id INTEGER DEFAULT 0,
    intent TEXT,
    score DOUBLE PRECISION,
    spans JSONB,
    entities JSONB,
    model_version TEXT,
    ts TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_intents_session_ts ON intents(session_id, ts DESC);

-- Series metadata catalog powering search helpers
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
