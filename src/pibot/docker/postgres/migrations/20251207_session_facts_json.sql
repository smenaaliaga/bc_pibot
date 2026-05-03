-- Migration: consolidate legacy key/value session_facts rows into JSON blobs.
-- Run once after pulling the LangGraph memory refresh changes.
DO $$
DECLARE
    has_fact_key BOOLEAN;
    has_facts_json BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'session_facts'
          AND column_name = 'fact_key'
    ) INTO has_fact_key;

    SELECT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'session_facts'
          AND column_name = 'facts'
    ) INTO has_facts_json;

    IF has_facts_json THEN
        RAISE NOTICE 'session_facts already stores JSON blobs; skipping migration.';
        RETURN;
    END IF;

    IF NOT has_fact_key THEN
        RAISE NOTICE 'session_facts table not found or already rebuilt; skipping migration.';
        RETURN;
    END IF;

    CREATE TABLE IF NOT EXISTS session_facts_v2 (
        session_id TEXT PRIMARY KEY,
        facts JSONB,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    INSERT INTO session_facts_v2 (session_id, facts, updated_at)
    SELECT
        session_id,
        jsonb_object_agg(fact_key, COALESCE(fact_value, '')) FILTER (WHERE fact_key IS NOT NULL),
        COALESCE(MAX(ts), NOW())
    FROM session_facts
    GROUP BY session_id
    ON CONFLICT (session_id) DO UPDATE
        SET facts = EXCLUDED.facts,
            updated_at = EXCLUDED.updated_at;

    DROP TABLE session_facts;
    ALTER TABLE session_facts_v2 RENAME TO session_facts;
    CREATE INDEX IF NOT EXISTS idx_session_facts_updated ON session_facts(updated_at DESC);
END $$;
