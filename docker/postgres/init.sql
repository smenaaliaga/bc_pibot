-- Inicialización de la base de datos y tablas de memoria/diversidad
CREATE DATABASE pibot;
\c pibot;

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
    session_id TEXT NOT NULL,
    fact_key TEXT NOT NULL,
    fact_value TEXT,
    ts TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (session_id, fact_key)
);
CREATE INDEX IF NOT EXISTS idx_session_facts_session ON session_facts(session_id);

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
