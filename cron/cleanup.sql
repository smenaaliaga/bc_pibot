-- Limpieza periódica de tablas de memoria y diversidad.
-- Ajusta los intervalos según tus necesidades.

-- Purga turnos de memoria corta (checkpointer y sidecar)
DELETE FROM checkpoints
WHERE (checkpoint->>'ts')::timestamptz < NOW() - INTERVAL '30 days';

DELETE FROM checkpoint_blobs
WHERE checkpoint_id NOT IN (SELECT checkpoint_id FROM checkpoints);

DELETE FROM checkpoint_writes
WHERE checkpoint_id NOT IN (SELECT checkpoint_id FROM checkpoints);

DELETE FROM short_term_turns
WHERE ts < NOW() - INTERVAL '30 days';

-- Purga hechos de sesión e intents antiguos
DELETE FROM session_facts
WHERE ts < NOW() - INTERVAL '90 days';

DELETE FROM intents
WHERE ts < NOW() - INTERVAL '90 days';

VACUUM (ANALYZE) checkpoints;
VACUUM (ANALYZE) checkpoint_blobs;
VACUUM (ANALYZE) checkpoint_writes;
VACUUM (ANALYZE) short_term_turns;
VACUUM (ANALYZE) session_facts;
VACUUM (ANALYZE) intents;
