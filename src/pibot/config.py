"""
config.py
---------
Centraliza la configuración del proyecto:
- Carga variables de entorno (.env)
- Expone un objeto Settings
"""

from dataclasses import dataclass
from typing import Optional
import os
from dotenv import load_dotenv

# Cargar variables desde .env (si existe)
load_dotenv()

# ── Azure OpenAI mode ────────────────────────────────────────────────
# Cuando USE_AZURE=true, se sobreescriben OPENAI_API_KEY y OPENAI_BASE_URL
# para que todos los consumidores (OpenAI SDK, LangChain) apunten a Azure
# de forma transparente sin cambiar su código.
USE_AZURE: bool = os.getenv("USE_AZURE", "false").lower() in {"1", "true", "yes", "on"}

if USE_AZURE:
    _azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    _azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    _azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    _azure_model = os.getenv("AZURE_OPENAI_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    # Sobreescribir env vars — el SDK OpenAI y LangChain las leen automáticamente
    os.environ["OPENAI_API_KEY"] = _azure_key
    os.environ["OPENAI_BASE_URL"] = (
        f"{_azure_endpoint}/openai/deployments/{_azure_model}"
    )
else:
    _azure_key = ""
    _azure_endpoint = ""
    _azure_api_version = ""
    _azure_model = ""

# Compatibilidad con módulos que esperan constantes a nivel de módulo
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
BCCH_USER: str = os.getenv("BCCH_USER", "")
BCCH_PASS: str = os.getenv("BCCH_PASS", "")
REDIS_URL: Optional[str] = os.getenv("REDIS_URL")

# Endpoints de clasificación remota
PREDICT_URL: str = os.getenv(
    "PREDICT_URL",
    "http://localhost:8000/predict",
)
PREDICT_TIMEOUT_SECONDS: float = float(os.getenv("PREDICT_TIMEOUT_SECONDS", "10"))

# Conexión al endpoint BDE
BDE_USER: str = os.getenv("BDE_USER", BCCH_USER)
BDE_PASS: str = os.getenv("BDE_PASS", BCCH_PASS)
BDE_BASE_URL: str = os.getenv(
    "BDE_BASE_URL",
    "https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx",
)
BDE_TIMEOUT_SEC: int = int(os.getenv("BDE_TIMEOUT_SEC", "15"))

# Flags de arranque y streaming
PREDICT_HEALTHCHECK_ON_START: bool = os.getenv(
    "PREDICT_HEALTHCHECK_ON_START", "1"
).lower() in {"1", "true", "yes", "on"}
PREDICT_HEALTH_TIMEOUT_SECONDS: float = float(
    os.getenv("PREDICT_HEALTH_TIMEOUT_SECONDS", "5")
)
INGEST_ON_START: bool = os.getenv(
    "INGEST_ON_START", "0"
).lower() in {"1", "true", "yes", "on"}

STREAM_CHUNK_LOGS: bool = os.getenv(
    "STREAM_CHUNK_LOGS", "0"
).lower() in {"1", "true", "yes", "on"}
LANGGRAPH_CHECKPOINT_NS: str = os.getenv("LANGGRAPH_CHECKPOINT_NS", "memory")

# SSL verification — set SSL_VERIFY=false to disable (corporate proxies with
# self-signed certs), or set it to a path to a custom CA bundle file.
_ssl_raw = os.getenv("SSL_VERIFY", "true").strip()
if _ssl_raw.lower() in {"0", "false", "no", "off"}:
    SSL_VERIFY: bool | str = False
elif _ssl_raw.lower() in {"1", "true", "yes", "on"}:
    SSL_VERIFY = True
else:
    SSL_VERIFY = _ssl_raw  # treat as path to CA bundle


def _azure_request_hook(request: "httpx.Request") -> None:
    """Reescribe requests del SDK OpenAI para compatibilidad con Azure OpenAI.

    Misma lógica de conexión que test_openai_azure.py (AzureOpenAI SDK):
    - Reemplaza header Authorization: Bearer → api-key
    - Agrega query param api-version
    - Redirige /embeddings al deployment de embeddings si difiere del de chat
    """
    import httpx as _hx

    # Auth: el SDK envía "Authorization: Bearer <key>", Azure espera "api-key: <key>"
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        request.headers["api-key"] = auth[7:]
        del request.headers["authorization"]

    # api-version obligatorio en Azure
    url_str = str(request.url)
    if "api-version" not in url_str:
        sep = "&" if "?" in url_str else "?"
        url_str = f"{url_str}{sep}api-version={_azure_api_version}"

    # Embeddings usan un deployment distinto al de chat
    _embed_model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large")
    if "/embeddings" in request.url.path and _azure_model != _embed_model:
        url_str = url_str.replace(
            f"/deployments/{_azure_model}/",
            f"/deployments/{_embed_model}/",
        )

    request.url = _hx.URL(url_str)


async def _azure_async_request_hook(request: "httpx.Request") -> None:
    """Versión async del hook para AsyncClient."""
    _azure_request_hook(request)


def get_httpx_client() -> "httpx.Client":
    """Return an ``httpx.Client`` respecting ``SSL_VERIFY`` and ``USE_AZURE``."""
    import httpx
    kwargs: dict = {"verify": SSL_VERIFY}
    if USE_AZURE:
        kwargs["event_hooks"] = {"request": [_azure_request_hook]}
    return httpx.Client(**kwargs)


def get_async_httpx_client() -> "httpx.AsyncClient":
    """Return an ``httpx.AsyncClient`` respecting ``SSL_VERIFY`` and ``USE_AZURE``."""
    import httpx
    kwargs: dict = {"verify": SSL_VERIFY}
    if USE_AZURE:
        kwargs["event_hooks"] = {"request": [_azure_async_request_hook]}
    return httpx.AsyncClient(**kwargs)


# Apply global SSL patch when verification is disabled (corporate proxy).
if SSL_VERIFY is False:
    import ssl as _ssl
    _ssl._create_default_https_context = _ssl._create_unverified_context  # type: ignore[attr-defined]
    try:
        import urllib3 as _urllib3
        _urllib3.disable_warnings(_urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass


@dataclass
class Settings:
    """Configuración principal del chatbot."""

    openai_api_key: str
    openai_model: str = "gpt-5.4-mini"
    openai_embeddings_model: str = "text-embedding-3-large" 
    bot_name: str = "PIBot"
    welcome_message: Optional[str] = None
    debug: bool = False
    pg_dsn: Optional[str] = None  # se rellena desde PG_DSN / DATABASE_URL si existen

    # Parámetros de comportamiento del frontend
    history_length: int = 5
    min_time_between_requests: float = 3.0  # segundos


def get_settings() -> Settings:
    """Lee las variables de entorno y devuelve un objeto Settings.

    Variables esperadas:
    - OPENAI_API_KEY (obligatoria)
    - OPENAI_MODEL (opcional, por defecto: gpt-4o-mini)
    - BOT_NAME (opcional, por defecto: PIBot)
    - DEBUG (opcional, true/false)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY no está definido. "
            "Crea un archivo .env con OPENAI_API_KEY=sk-xxxx o "
            "exporta la variable de entorno."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embeddings_model = (
        os.getenv("OPENAI_EMBEDDINGS_MODEL")
        or os.getenv("OPENAI_EMBED_MODEL")
        or "text-embedding-3-large"
    )
    bot_name = os.getenv("BOT_NAME", "PIBot")
    welcome_message = os.getenv("WELCOME_MESSAGE")
    debug = os.getenv("DEBUG", "false").lower() == "true"
    pg_dsn = os.getenv("PG_DSN") or os.getenv("DATABASE_URL")

    return Settings(
        openai_api_key=api_key,
        openai_model=model,
        openai_embeddings_model=embeddings_model,
        bot_name=bot_name,
        welcome_message=welcome_message,
        debug=debug,
        pg_dsn=pg_dsn,
    )
