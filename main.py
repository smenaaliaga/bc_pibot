"""
main.py
-------
Punto de entrada del sistema.

- Carga configuración (config.get_settings)
- Inicializa el orquestador LangChain (orchestrator)
- Llama a la app de Streamlit (app.run_app)

Ejecutar con:
    streamlit run main.py
"""

from typing import List, Dict, Optional

from config import get_settings
from orchestrator import create_orchestrator_with_langchain
import app
import logging
import os
import datetime


def main() -> None:
    """Punto de entrada principal."""
    # Configurar logger local por ejecución
    logs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Permite forzar un único archivo de log entre reruns (útil en Streamlit)
    fixed_log = os.getenv("RUN_MAIN_LOG", "").strip()
    if fixed_log:
        log_path = os.path.join(logs_dir, fixed_log)
    else:
        log_path = os.path.join(logs_dir, f"run_main_{ts}.log")
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | session=%(session_id)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,  # asegura un único set de handlers por ejecución/rerun
    )

    class SessionFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "session_id"):
                record.session_id = "main"
            return True

    for h in logging.getLogger().handlers:
        h.addFilter(SessionFilter())

    base_logger = logging.getLogger(__name__)
    logger = logging.LoggerAdapter(base_logger, extra={"session_id": "main"})
    logger.info("Inicializando aplicación Streamlit")

    settings = get_settings()

    orch = create_orchestrator_with_langchain()
    logger.info("Orquestador LangChain inicializado")

    def stream_fn(question: str, history: Optional[List[Dict[str, str]]] = None, session_id: Optional[str] = None):
        logger.info(f"[STREAM] session={session_id} question={question[:80]}")
        return orch.stream(question, history=history, session_id=session_id)

    def invoke_fn(question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        logger.info(f"[INVOKE] question={question[:80]}")
        return "".join(stream_fn(question, history=history, session_id=None))

    # Delegar la construcción de la UI a app.py
    logger.info("Lanzando UI Streamlit")
    app.run_app(
        settings=settings,
        stream_fn=stream_fn,
        invoke_fn=invoke_fn,
    )


if __name__ == "__main__":
    # En ejecución normal (python main.py) simplemente llamamos main().
    # Para Streamlit, el comando recomendado sigue siendo:
    #   streamlit run main.py
    main()
