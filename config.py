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

# Compatibilidad con módulos que esperan constantes a nivel de módulo
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
BCCH_USER: str = os.getenv("BCCH_USER", "")
BCCH_PASS: str = os.getenv("BCCH_PASS", "")
REDIS_URL: Optional[str] = os.getenv("REDIS_URL")

# Modelo JointBERT para clasificación de intenciones
JOINT_BERT_MODEL_DIR: str = os.getenv("JOINT_BERT_MODEL_DIR", "model/out/pibot_model_beto")

# Controla si se exponen enlaces de API con credenciales en texto plano en los logs.
# Por defecto ACTIVADO en este entorno protegido; puedes desactivarlo con LOG_EXPOSE_API_LINKS=0
LOG_EXPOSE_API_LINKS: bool = os.getenv("LOG_EXPOSE_API_LINKS", "1").lower() in ("1", "true", "yes")


@dataclass
class Settings:
    """Configuración principal del chatbot."""

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embeddings_model: str = "text-embedding-3-large"  # puede sobreescribirse con OPENAI_EMBEDDINGS_MODEL
    bot_name: str = "PIBot"
    welcome_message: Optional[str] = None
    debug: bool = False
    pg_dsn: Optional[str] = None  # se rellena desde PG_DSN / DATABASE_URL si existen

    # Parámetros de comportamiento del frontend
    show_suggestions: bool = False
    history_length: int = 5
    summarize_old_history: bool = False
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
