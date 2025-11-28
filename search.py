# -*- coding: utf-8 -*-
"""
search.py
---------
Capa de búsqueda vectorial para series del PIBot.

Incluye:

1) Inserción en pgvector (PostgreSQL + extensión pgvector) a partir del Excel
   `series_pibot_con_metadata.xlsx`:
       - insert_pgvector_series(excel_path: str, table_name: str = "series_embeddings")

2) Inserción en Azure AI Search como base de datos vectorial:
       - insert_ai_search_azure(excel_path: str)

3) Búsqueda vectorial en pgvector a partir del nombre aproximado de la serie:
       - search_serie_pg_vector(nombre_serie: str, top_k: int = 1)

4) Búsqueda vectorial en Azure AI Search a partir del nombre aproximado de la serie:
       - search_azure_ai_searchr(nombre_serie: str, top_k: int = 1)

Supuestos de diseño:

- Cada fila del Excel `series_pibot_con_metadata.xlsx` representa UNA serie:
    * COD_SERIE      → identificador único de la serie (string BCCh).
    * NKNAME_ESP     → nombre detallado de la serie en español.
    * Otros metadatos (región, dominio, etc.) NO se indexan aquí; se recuperan luego
      desde un catálogo JSON externo usando COD_SERIE.

- Para los embeddings se usa OpenAI (o Azure OpenAI) vía `openai.Embeddings`.
  El modelo de embeddings se toma de config.py:
      settings.openai_embeddings_model
  o por defecto: "text-embedding-3-large".

- Para pgvector:
    * Debe existir una base PostgreSQL con la extensión pgvector instalada.
    * Se usará una tabla (por defecto `series_embeddings`) con:
        cod_serie TEXT PRIMARY KEY
        nkname_esp TEXT
        embedding  vector(<dimensión del modelo>)

- Para Azure AI Search:
    * Debe existir un índice para series (p.ej. `pib-series`) con:
        - id            : Edm.String, key
        - cod_serie     : Edm.String (filterable)
        - nkname_esp    : Edm.String, searchable
        - content       : Edm.String, searchable (opcional, se puede igualar a NKNAME_ESP)
        - embedding     : Collection(Edm.Single), vector config
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

# pgvector + PostgreSQL
import psycopg2
try:
    import pgvector  # valida presencia del paquete base
except ImportError:
    pgvector = None  # type: ignore

# Intentar ambas variantes de registro (psycopg2 y psycopg3)
try:
    from pgvector.psycopg2 import register_vector as _register_vector_psycopg2  # type: ignore
except Exception:
    _register_vector_psycopg2 = None  # type: ignore

try:
    from pgvector.psycopg import register_vector as _register_vector_psycopg3  # type: ignore
except Exception:
    _register_vector_psycopg3 = None  # type: ignore


def register_vector(conn):  # wrapper unificado
    """Registra el tipo vector en la conexión (psycopg2 o psycopg3)."""
    if _register_vector_psycopg2 is not None:
        return _register_vector_psycopg2(conn)
    if _register_vector_psycopg3 is not None:
        return _register_vector_psycopg3(conn)
    raise RuntimeError(
        "pgvector.register_vector no disponible. Instala/actualiza: pip install pgvector psycopg2-binary"
    )

# Azure AI Search
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import Vector
except ImportError:
    AzureKeyCredential = None  # type: ignore
    SearchClient = None        # type: ignore
    Vector = None              # type: ignore

from config import get_settings, LOG_LEVEL
# Logger: usar local si existe; fallback minimal si no
try:
    from logger import get_logger, Phase  # type: ignore
except Exception:
    import logging, contextlib, time as _time, os as _os, datetime as _dt

    def get_logger(name: str, level: str = "INFO") -> logging.Logger:
        logger = logging.getLogger(name)
        if logger.handlers:
            return logger
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        root = _os.path.abspath(_os.path.dirname(__file__))
        log_dir = _os.path.join(root, "logs")
        _os.makedirs(log_dir, exist_ok=True)
        fname = _os.path.join(log_dir, f"pibot_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = logging.FileHandler(fname, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        return logger

    class Phase(contextlib.ContextDecorator):
        def __init__(self, logger: logging.Logger, name: str, extra: Optional[dict] = None):
            self.logger = logger
            self.name = name
            self.extra = extra or {}
            self._t0: Optional[float] = None

        def __enter__(self):
            self._t0 = _time.perf_counter()
            self.logger.info(f"[FASE] start: {self.name} | {self.extra}")
            return self

        def __exit__(self, exc_type, exc, tb):
            t1 = _time.perf_counter()
            if exc:
                self.logger.error(f"[FASE] error: {self.name} ({t1 - (self._t0 or t1):.3f}s) | error={exc}")
            else:
                self.logger.info(f"[FASE] end: {self.name} ({t1 - (self._t0 or t1):.3f}s)")
            return False

logger = get_logger(__name__, level=LOG_LEVEL)
# Reusar logger del orquestador si disponible para lograr archivo único por ejecución
try:
    import orchestrator as _orch_mod
    if hasattr(_orch_mod, 'logger'):
        logger = _orch_mod.logger  # type: ignore[assignment]
        path_used = getattr(logger, '_session_log_path', None)
        logger.info(f"[LOG_REUSE] search.py reutiliza logger del orquestador (archivo único) path={path_used}")
except Exception as _e_log_reuse:
    # Mantener logger local, pero registrar el fallo
    try:
        path_used = None
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                path_used = getattr(h, 'baseFilename', None)
                break
        logger.info(f"[LOG_LOCAL] usando logger local path={path_used}")
    except Exception:
        logger.info(f"[LOG_REUSE] No se pudo reutilizar logger del orquestador: {_e_log_reuse}")

settings = get_settings()

# ---------------------------------------------------------------------------
# Config de embeddings
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL = (
    os.getenv("OPENAI_EMBED_MODEL")
    or os.getenv("OPENAI_EMBEDDINGS_MODEL")
    or getattr(settings, "openai_embeddings_model", "text-embedding-3-large")
)
_client = OpenAI(api_key=settings.openai_api_key)


def _embed_text(text: str) -> List[float]:
    """
    Genera embedding para un texto usando el modelo configurado.
    """
    text = (text or "").strip()
    if not text:
        return []
    resp = _client.embeddings.create(
        model=_EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


# Exponer función pública para obtener embedding de un texto (para tester)
def get_query_embedding(text: str) -> List[float]:
    return _embed_text(text)


# ---------------------------------------------------------------------------
# Dataclass original (si quieres reutilizarla para futuras funciones)
# ---------------------------------------------------------------------------

@dataclass
class VectorSearchInput:
    query_text: str
    data_domain: str
    is_generic: bool
    imacec: Optional[Dict[str, Any]] = None
    pibe: Optional[Dict[str, Any]] = None
    default_key: Optional[str] = None
    default_series_metadata: Optional[Dict[str, Any]] = None
    top_k: int = 5


def semantic_search_series(params: VectorSearchInput) -> List[Dict[str, Any]]:
    """
    Stub genérico, por si quieres más adelante un router que decida entre pgvector
    y Azure AI Search. Por ahora solo imprime el input.
    """
    logger.info(f"[semantic_search_series] Stub llamado con params={params}")
    return []


# ---------------------------------------------------------------------------
# 1) Inserción en pgvector a partir del Excel
# ---------------------------------------------------------------------------

def _get_pg_dsn() -> str:
    """
    Obtiene el DSN de PostgreSQL/pgvector desde settings o variables de entorno.
    """
    dsn = getattr(settings, "pg_dsn", None) or os.getenv("PG_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "No se encontró DSN para PostgreSQL. Configura settings.pg_dsn o la env var PG_DSN / DATABASE_URL."
        )
    return dsn


def insert_pgvector_series(
    excel_path: str,
    table_name: str = "series_embeddings",
    batch_size: int = 100,
) -> None:
    """
    Inserta TODAS las series del Excel `series_pibot_con_metadata.xlsx` en una tabla
    pgvector como base de datos vectorial.

    - Usa únicamente:
        * COD_SERIE     → cod_serie
        * NKNAME_ESP    → nkname_esp / texto a embeddear

    El resto de metadatos NO se indexan aquí, porque se recuperarán desde un JSON
    externo cuando se tenga el COD_SERIE.

    La tabla pgvector se crea automáticamente si no existe, con la dimensión
    adecuada a partir del primer embedding.

    Esquema propuesto:
        CREATE TABLE series_embeddings (
            cod_serie  TEXT PRIMARY KEY,
            nkname_esp TEXT,
            embedding  vector(<dim>)
        );
    """
    if pgvector is None:
        raise RuntimeError(
            "El módulo 'pgvector' no está instalado. Ejecuta: pip install pgvector psycopg2-binary"
        )

    logger.info(f"[insert_pgvector_series] Leyendo Excel desde: {excel_path}")
    df = pd.read_excel(excel_path)

    # Validaciones mínimas
    if "COD_SERIE" not in df.columns or "NKNAME_ESP" not in df.columns:
        raise ValueError(
            "El Excel debe contener las columnas 'COD_SERIE' y 'NKNAME_ESP'."
        )

    # Nos quedamos solo con las columnas mínimas para el vector store
    df = df[["COD_SERIE", "NKNAME_ESP"]].dropna(subset=["COD_SERIE", "NKNAME_ESP"])

    dsn = _get_pg_dsn()
    logger.info(f"[insert_pgvector_series] Conectando a PostgreSQL con DSN={dsn!r}")

    with Phase(logger, "FASE_PGV_1: conexión pgvector", {"dsn": dsn}):
        conn = psycopg2.connect(dsn)
        conn.autocommit = False
        cur = conn.cursor()
        # Asegurar extensión vector ANTES de registrar el tipo en la conexión
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
        except Exception:
            conn.rollback()
        # Registrar tipo vector en la conexión (psycopg2/psycopg3)
        register_vector(conn)

    try:
        # Obtener primer embedding para dimensionar la tabla
        logger.info("[insert_pgvector_series] Calculando embedding de ejemplo para dimensionar la tabla...")
        first_row = df.iloc[0]
        example_emb = _embed_text(str(first_row["NKNAME_ESP"]))
        if not example_emb:
            raise RuntimeError("No se pudo calcular embedding de ejemplo.")
        dim = len(example_emb)

        with Phase(
            logger,
            "FASE_PGV_2: crear tabla series_embeddings (si no existe)",
            {"table_name": table_name, "dim": dim},
        ):
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    cod_serie  TEXT PRIMARY KEY,
                    nkname_esp TEXT,
                    embedding  vector({dim})
                );
                """
            )
            conn.commit()

        # Inserción por lotes
        total = len(df)
        logger.info(f"[insert_pgvector_series] Total de series a insertar: {total}")

        with Phase(
            logger,
            "FASE_PGV_3: inserción de embeddings en pgvector",
            {"table_name": table_name, "total_series": total},
        ):
            batch: List[tuple] = []
            for i, row in df.iterrows():
                cod = str(row["COD_SERIE"]).strip()
                nkname = str(row["NKNAME_ESP"]).strip()
                if not cod or not nkname:
                    continue

                emb = _embed_text(nkname)
                if not emb:
                    logger.warning(
                        f"[insert_pgvector_series] Embedding vacío para cod_serie={cod!r}; se omite."
                    )
                    continue

                batch.append((cod, nkname, emb))

                if len(batch) >= batch_size:
                    _insert_pg_batch(cur, conn, table_name, batch)
                    logger.info(
                        f"[insert_pgvector_series] Insertadas {i+1}/{total} filas (último batch)."
                    )
                    batch = []

            # último batch
            if batch:
                _insert_pg_batch(cur, conn, table_name, batch)
                logger.info(
                    f"[insert_pgvector_series] Inserción completada. Total filas: {total}"
                )

    finally:
        cur.close()
        conn.close()
        logger.info("[insert_pgvector_series] Conexión a PostgreSQL cerrada.")


def _insert_pg_batch(
    cur: "psycopg2.extensions.cursor",
    conn: "psycopg2.extensions.connection",
    table_name: str,
    batch: List[tuple],
) -> None:
    """
    Inserta un batch de (cod_serie, nkname_esp, embedding) con upsert.
    """
    sql = f"""
        INSERT INTO {table_name} (cod_serie, nkname_esp, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (cod_serie)
        DO UPDATE SET
            nkname_esp = EXCLUDED.nkname_esp,
            embedding  = EXCLUDED.embedding;
    """
    cur.executemany(sql, batch)
    conn.commit()


# ---------------------------------------------------------------------------
# 2) Inserción en Azure AI Search a partir del Excel
# ---------------------------------------------------------------------------

def _get_search_client_series() -> Any:
    """
    Crea un SearchClient para el índice de series en Azure AI Search.

    Requiere en config.py o variables de entorno:
        settings.azure_search_endpoint_series  o  AZURE_SEARCH_ENDPOINT_SERIES
        settings.azure_search_key              o  AZURE_SEARCH_KEY
        settings.azure_search_index_series     o  AZURE_SEARCH_INDEX_SERIES
    """
    if SearchClient is None or AzureKeyCredential is None:
        raise RuntimeError(
            "azure-search-documents no está instalado. Ejecuta: pip install azure-search-documents"
        )

    endpoint = getattr(settings, "azure_search_endpoint_series", None) or os.getenv(
        "AZURE_SEARCH_ENDPOINT_SERIES"
    )
    key = getattr(settings, "azure_search_key", None) or os.getenv("AZURE_SEARCH_KEY")
    index_name = getattr(settings, "azure_search_index_series", None) or os.getenv(
        "AZURE_SEARCH_INDEX_SERIES"
    )

    if not endpoint or not key or not index_name:
        raise RuntimeError(
            "Faltan configuraciones para Azure AI Search. "
            "Define endpoint, key e index_name en config.py o variables de entorno."
        )

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(key),
    )


def insert_ai_search_azure(
    excel_path: str,
    batch_size: int = 100,
) -> None:
    """
    Inserta todas las series del Excel en el índice de Azure AI Search como base
    vectorial.

    Se indexan solo:
        - id          → COD_SERIE
        - cod_serie   → COD_SERIE
        - nkname_esp  → NKNAME_ESP
        - content     → NKNAME_ESP (texto base)
        - embedding   → embedding de NKNAME_ESP (Collection(Edm.Single))

    Los metadatos (url de cuadro, dimensiones SDMX, etc.) se obtendrán luego
    desde un catálogo JSON externo usando cod_serie como clave.
    """
    logger.info(f"[insert_ai_search_azure] Leyendo Excel desde: {excel_path}")
    df = pd.read_excel(excel_path)

    if "COD_SERIE" not in df.columns or "NKNAME_ESP" not in df.columns:
        raise ValueError(
            "El Excel debe contener las columnas 'COD_SERIE' y 'NKNAME_ESP'."
        )

    df = df[["COD_SERIE", "NKNAME_ESP"]].dropna(subset=["COD_SERIE", "NKNAME_ESP"])

    client = _get_search_client_series()
    total = len(df)
    logger.info(
        f"[insert_ai_search_azure] Conectado a Azure AI Search. Total de series a insertar: {total}"
    )

    with Phase(
        logger,
        "FASE_AIS_1: inserción de documentos en Azure AI Search",
        {"total_series": total},
    ):
        batch_docs: List[Dict[str, Any]] = []

        for i, row in df.iterrows():
            cod = str(row["COD_SERIE"]).strip()
            nkname = str(row["NKNAME_ESP"]).strip()
            if not cod or not nkname:
                continue

            emb = _embed_text(nkname)
            if not emb:
                logger.warning(
                    f"[insert_ai_search_azure] Embedding vacío para cod_serie={cod!r}; se omite."
                )
                continue

            doc = {
                "id": cod,
                "cod_serie": cod,
                "nkname_esp": nkname,
                "content": nkname,
                "embedding": emb,
            }
            batch_docs.append(doc)

            if len(batch_docs) >= batch_size:
                result = client.upload_documents(documents=batch_docs)
                logger.info(
                    f"[insert_ai_search_azure] Subido batch de {len(batch_docs)} docs. "
                    f"Progreso: {i+1}/{total}"
                )
                batch_docs = []

        if batch_docs:
            result = client.upload_documents(documents=batch_docs)
            logger.info(
                f"[insert_ai_search_azure] Inserción completada. Total filas: {total}"
            )


# ---------------------------------------------------------------------------
# 3) Búsqueda vectorial en pgvector
# ---------------------------------------------------------------------------

def search_serie_pg_vector(nombre_serie: str, top_k: int = 1) -> Optional[List[Dict[str, Any]]]:
    """Realiza búsqueda vectorial en pgvector para nombre aproximado de la serie.

    Retorna lista de dicts ordenados por similitud DESC con claves:
      - cod_serie
      - nkname_esp
      - similarity (float)
    Si top_k == 1 se mantiene compatibilidad retornando lista con un elemento.
    """
    logger.info(f"[VECTOR_SEARCH] start | provider=pgvector | top_k={top_k} | query={nombre_serie!r}")
    if pgvector is None:
        logger.error("[VECTOR_SEARCH] pgvector no instalado")
        return None
    nombre_serie = (nombre_serie or '').strip()
    if not nombre_serie:
        return None
    dsn = None
    try:
        dsn = _get_pg_dsn()
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH] DSN error: {e}")
        return None
    try:
        emb = _embed_text(nombre_serie)
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH] embedding error: {e}")
        return None
    if not emb:
        return None
    try:
        conn = psycopg2.connect(dsn)
        register_vector(conn)
        cur = conn.cursor()
        # Parametrizar como literal de vector para evitar error "operator does not exist: vector <-> numeric[]"
        vec_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"
        sql = """
        SELECT cod_serie, nkname_esp, embedding <-> %s::vector AS distance
        FROM series_embeddings
        ORDER BY distance ASC
        LIMIT %s;
        """
        cur.execute(sql, (vec_literal, top_k))
        rows = cur.fetchall()
        cur.close(); conn.close()
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH] query error: {e}")
        try:
            cur.close(); conn.close()
        except Exception:
            pass
        return None
    out: List[Dict[str, Any]] = []
    for r in rows:
        cod, nk, dist = r
        similarity = 1.0 - float(dist) if dist is not None else None
        out.append({"cod_serie": cod, "nkname_esp": nk, "similarity": similarity})
    logger.info(f"[VECTOR_SEARCH] end | results={len(out)} | codes={[m['cod_serie'] for m in out]}")
    return out


# ---------------------------------------------------------------------------
# 4) Búsqueda vectorial en Azure AI Search
# ---------------------------------------------------------------------------

def search_azure_ai_searchr(
    nombre_serie: str,
    top_k: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Realiza una búsqueda vectorial en Azure AI Search a partir de `nombre_serie`
    (ej.: "Imacec empalmado, serie original (índice 2018=100)") y retorna el
    documento con mayor similitud según el vector `embedding`.

    Retorna:
        - Si top_k == 1:
            {"cod_serie": ..., "nkname_esp": ..., "score": ...}
        - Si top_k > 1:
            {"results": [ {...}, {...}, ... ]}
    """
    emb = _embed_text(nombre_serie)
    if not emb:
        logger.warning("[search_azure_ai_searchr] Embedding vacío; no se puede buscar.")
        return None

    client = _get_search_client_series()
    logger.info(
        f"[search_azure_ai_searchr] Buscando en Azure AI Search | nombre_serie='{nombre_serie}'"
    )

    with Phase(
        logger,
        "FASE_AIS_SEARCH_1: búsqueda vectorial en Azure AI Search",
        {"top_k": top_k},
    ):
        # Nuevo modelo de vector search: vectors=[Vector(...)]
        # - search_text="" para no usar BM25, solo vector
        # - fields="embedding" debe coincidir con el campo vectorial del índice
        vector = Vector(value=emb, k=top_k, fields="embedding")
        results_iter = client.search(
            search_text="",
            vectors=[vector],
            select=["id", "cod_serie", "nkname_esp"],
        )

        results: List[Dict[str, Any]] = []
        for r in results_iter:
            doc = {
                "cod_serie": r.get("cod_serie", r.get("id")),
                "nkname_esp": r.get("nkname_esp"),
                "score": float(r["@search.score"]),
            }
            results.append(doc)

    if not results:
        logger.info("[search_azure_ai_searchr] Sin resultados.")
        return None

    if top_k == 1:
        best = results[0]
        logger.info(
            f"[search_azure_ai_searchr] Mejor match: cod_serie={best['cod_serie']!r}, "
            f"nkname_esp={best['nkname_esp']!r}, score={best['score']:.6f}"
        )
        return best

    logger.info(
        f"[search_azure_ai_searchr] Devueltos {len(results)} resultados (top_k={top_k})."
    )
    return {"results": results}
