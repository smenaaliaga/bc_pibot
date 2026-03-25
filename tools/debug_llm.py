"""Quick script to inspect LLM output with RAG context.

Usage:
    python tools/debug_llm.py            # non-streaming (default)
    python tools/debug_llm.py --stream   # streaming chunks
"""
import argparse

from orchestrator.rag.rag_factory import create_retriever
from orchestrator.llm.llm_adapter import build_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug LLM output with RAG context")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    args = parser.parse_args()

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

    if args.stream:
        adapter = build_llm(streaming=True, retriever=retriever)
        print("Streaming chunks:")
        for idx, chunk in enumerate(adapter.stream(question, history=history, intent_info=intent_info), start=1):
            print(f"{idx:02d}: {repr(chunk)}")
    else:
        llm = build_llm(streaming=False, retriever=retriever)
        response = llm.generate(question, history=history, intent_info=intent_info)
        print("LLM response:\n", response)


if __name__ == "__main__":
    main()
