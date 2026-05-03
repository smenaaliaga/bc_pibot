"""Quick script to inspect LLM output with RAG context."""
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
    print("Retriever type:", type(retriever).__name__ if retriever else None)
    llm = build_llm(streaming=False, retriever=retriever)
    response = llm.generate(question, history=history, intent_info=intent_info)
    print("LLM response:\n", response)


if __name__ == "__main__":
    main()
