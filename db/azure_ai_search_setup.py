# -*- coding: utf-8 -*-
"""azure_ai_search_setup.py
---------------------------------
Script de utilería para preparar el entorno de Azure AI Search desde Python,
como análogo al script SQL `pgvector.sql` usado para PostgreSQL/pgvector.

Este script NO crea recursos de Azure (servicio de búsqueda, índice, etc.),
pero centraliza la lógica para:

- Validar variables de entorno requeridas para Azure AI Search y embeddings.
- Mostrar la configuración efectiva que usará `search.py`.
- Probar conectividad básica al servicio de búsqueda (si `search.py` expone
  funciones de healthcheck o creación de índice).

La idea es ejecutar este script antes de usar `tester_ai_search.py` para
asegurar que el "entorno lógico" de Azure AI Search está listo.

Uso sugerido:
    python db/azure_ai_search_setup.py

Requisitos:
    - Variables de entorno/`config.py` adecuadamente configuradas para:
        * Endpoint y API key de Azure AI Search
        * Nombre del índice a utilizar
        * OpenAI / Azure OpenAI para generación de embeddings (si aplica)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass con la configuración mínima esperada para Azure AI Search
# ---------------------------------------------------------------------------

@dataclass
class AzureSearchConfig:
    endpoint: str
    api_key: str
    index_name: str
    embedding_model: Optional[str] = None


def load_azure_search_config() -> AzureSearchConfig:
    """Carga la configuración de Azure AI Search desde variables de entorno.

    Ajusta los nombres de las variables para que coincidan con los usados
    realmente en `config.py` y `search.py`.
    """

    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL") or os.getenv(
        "AZURE_OPENAI_EMBEDDING_MODEL",
    )

    missing = []
    if not endpoint:
        missing.append("AZURE_SEARCH_ENDPOINT")
    if not api_key:
        missing.append("AZURE_SEARCH_API_KEY")
    if not index_name:
        missing.append("AZURE_SEARCH_INDEX_NAME")

    if missing:
        raise RuntimeError(
            "Faltan variables de entorno para Azure AI Search: "
            + ", ".join(missing)
        )

    return AzureSearchConfig(
        endpoint=endpoint,
        api_key=api_key,
        index_name=index_name,
        embedding_model=embedding_model,
    )


# ---------------------------------------------------------------------------
# Healthcheck opcional usando utilidades de search.py (si existen)
# ---------------------------------------------------------------------------


def try_healthcheck_with_search_module(cfg: AzureSearchConfig) -> None:
    """Intenta ejecutar un healthcheck básico utilizando `search.py`.

    Este paso es opcional y depende de que `search.py` exponga una función
    compatible, por ejemplo `ensure_azure_search_index()` o similar.

    Si no existe, el script solo imprimirá una advertencia y terminará.
    """

    try:
        import search  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnóstico manual
        print("No se pudo importar 'search.py'. Healthcheck omitido.")
        print(f"Detalle: {exc}")
        return

    # Ejemplo de función opcional que podrías implementar en search.py:
    #   - ensure_azure_search_index()
    #   - healthcheck_azure_search()
    # Ajusta el nombre según lo que realmente tengas disponible.

    if hasattr(search, "healthcheck_azure_search"):
        try:
            print("\n>> Ejecutando healthcheck_azure_search() desde search.py...")
            result = search.healthcheck_azure_search()  # type: ignore[attr-defined]
            print("Healthcheck Azure AI Search OK:")
            print(result)
        except Exception as exc:  # pragma: no cover - diagnóstico manual
            print("\n❌ Error en healthcheck_azure_search():")
            print(exc)
    else:
        print(
            "\nAdvertencia: 'search.py' no expone la función "
            "healthcheck_azure_search().\n"
            "Puedes crear una para validar conexión, índice, etc., y luego "
            "reuse este script."
        )


# ---------------------------------------------------------------------------
# main(): validación + healthcheck
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n=== PREPARACIÓN ENTORNO AZURE AI SEARCH (PYTHON) ===\n")

    try:
        cfg = load_azure_search_config()
    except Exception as exc:
        print("❌ Configuración incompleta para Azure AI Search.")
        print(exc)
        print(
            "\nRevisa tu archivo 'config.py' o variables de entorno, "
            "especialmente: AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, "
            "AZURE_SEARCH_INDEX_NAME."
        )
        return

    print("Configuración detectada para Azure AI Search:")
    print(f"  - Endpoint    : {cfg.endpoint}")
    print(f"  - Index name  : {cfg.index_name}")
    print(f"  - API key set : {'sí' if cfg.api_key else 'no'}")
    print(
        f"  - Embeddings  : {cfg.embedding_model if cfg.embedding_model else 'no definido'}"
    )

    # Paso opcional: intentar healthcheck usando funciones de search.py
    try_healthcheck_with_search_module(cfg)

    print("\n=== FIN PREPARACIÓN ENTORNO AZURE AI SEARCH ===\n")


if __name__ == "__main__":
    main()
