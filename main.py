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

from typing import List, Dict, Callable, Optional

from config import get_settings
import orchestrator
import app


def main() -> None:
    """Punto de entrada principal."""
    settings = get_settings()

    # Funciones que espera app.run_app (inyectadas desde el orquestador)
    def invoke_fn(question: str, history: Optional[List[Dict[str, str]]] = None, session_id: Optional[str] = None) -> str:
        return orchestrator.invoke(question, history=history, session_id=session_id)

    def stream_fn(question: str, history: Optional[List[Dict[str, str]]] = None, session_id: Optional[str] = None):
        return orchestrator.stream(question, history=history, session_id=session_id)

    # Delegar la construcción de la UI a app.py
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
