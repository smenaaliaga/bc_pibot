-- Crear base (ejecuta esto solo si tu usuario puede crearla; si ya existe, omite)
-- CREATE DATABASE mi_bd;
-- Conéctate a la base en DataGrip antes de seguir (mi_bd).

BEGIN;

-- Extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla principal de embeddings
DROP TABLE IF EXISTS series_embeddings CASCADE;
CREATE TABLE series_embeddings (
    cod_serie   TEXT PRIMARY KEY,
    nkname_esp  TEXT NOT NULL,
    embedding   vector(3072)  -- dimensión para text-embedding-3-large
);

-- Índice aproximado (requiere haber insertado suficientes filas antes de ser realmente útil).
-- Para búsquedas por distancia L2
CREATE INDEX IF NOT EXISTS series_embeddings_embedding_ivfflat
    ON series_embeddings USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100);

-- (Opcional) Índice HNSW (si tu versión de pgvector >= 0.5.0)
-- CREATE INDEX IF NOT EXISTS series_embeddings_embedding_hnsw
--     ON series_embeddings USING hnsw (embedding vector_l2_ops)
--     WITH (m = 16, ef_construction = 64);

COMMIT;

-- Consulta ejemplo de similitud (sustituye [...] por tu vector JSON/array):
-- SELECT cod_serie, nkname_esp,
--        embedding <-> '[0.12, -0.34, ...]' AS distancia
-- FROM series_embeddings
-- ORDER BY embedding <-> '[0.12, -0.34, ...]'
-- LIMIT 5;

-- Recomendación: tras insertar muchos registros, recrea el índice ivfflat
-- y ejecuta ANALYZE:
-- ANALYZE series_embeddings;