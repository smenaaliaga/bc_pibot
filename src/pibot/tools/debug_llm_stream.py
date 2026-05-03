"""Inspect ChatOpenAI streaming chunks."""
from orchestrator.rag.rag_factory import create_retriever
from orchestrator.llm.llm_adapter import build_llm


def main() -> None:
    question = "que es el pib?"
    history = [
        {
            "role": "assistant",
            "content": "Hola, soy PIBot, asistente económico del Banco Central de Chile. ¿En qué puedo ayudarte hoy?",
        }
    ]
    intent_info = {
        "intent": "METHODOLOGICAL",
        "score": 1.0,
        "entities": {
            "data_domain": "PIB",
            "is_generic": True,
            "default_key": "PIB_TOTAL",
        },
    }
    retriever = create_retriever()
    adapter = build_llm(streaming=True, retriever=retriever)
    print("Streaming chunks:")
    for idx, chunk in enumerate(adapter.stream(question, history=history, intent_info=intent_info), start=1):
        print(f"{idx:02d}: {repr(chunk)}")


if __name__ == "__main__":
    main()
