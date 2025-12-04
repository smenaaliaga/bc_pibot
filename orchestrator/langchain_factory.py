"""Factory for the LangChain-first orchestrator stack."""

from __future__ import annotations

import logging
from typing import Optional

from .langchain_memory import LangChainMemoryAdapter
from .langchain_llm import LangChainLLMAdapter
from .langchain_orchestrator import LangChainOrchestrator
from .intent_memory import IntentMemory
from .intent_store import IntentStoreBase, create_intent_store
from .rag_factory import create_retriever

logger = logging.getLogger(__name__)


def create_orchestrator_with_langchain(
    model: str = "gpt-3.5-turbo",
    persist_dir: Optional[str] = None,
    use_langgraph: bool = False,
    intent_classifier: Optional[object] = None,
    intent_memory: Optional[IntentMemory] = None,
    intent_store: Optional[IntentStoreBase] = None,
) -> LangChainOrchestrator:
    """Build a LangChain orchestrator without relying on orchestrator2."""

    if persist_dir or use_langgraph:
        logger.debug(
            "create_orchestrator_with_langchain: persist_dir/use_langgraph parameters are handled internally "
            "by LangChainMemoryAdapter; no additional action required."
        )

    mem_adapter = LangChainMemoryAdapter()
    retriever = create_retriever()
    llm_adapter = LangChainLLMAdapter(model=model, retriever=retriever)

    resolved_intent_memory = intent_memory
    resolved_intent_store = intent_store
    if intent_classifier:
        if resolved_intent_memory is None:
            resolved_intent_memory = IntentMemory()
        if resolved_intent_store is None:
            try:
                resolved_intent_store = create_intent_store()
            except Exception:
                resolved_intent_store = None
                logger.exception("create_intent_store failed; continuing without persistent intent store")

    orchestrator = LangChainOrchestrator(
        memory=mem_adapter,
        llm=llm_adapter,
        intent_classifier=intent_classifier,
        intent_memory=resolved_intent_memory,
        intent_store=resolved_intent_store,
    )

    return orchestrator
