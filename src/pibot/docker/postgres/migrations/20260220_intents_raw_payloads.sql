-- Migration: add raw API payloads to intents.
BEGIN;
ALTER TABLE intents ADD COLUMN IF NOT EXISTS intent_raw JSONB;
ALTER TABLE intents ADD COLUMN IF NOT EXISTS predict_raw JSONB;
COMMIT;
