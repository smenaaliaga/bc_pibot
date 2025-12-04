"""LangChain chat LLM adapter with streaming support."""

from __future__ import annotations

import os
import logging
from typing import Optional, List, Dict, Any, Iterable

logger = logging.getLogger(__name__)

FOLLOW_UP_SUFFIX = " Si quieres más detalle o que cite secciones específicas de la metodología, dímelo."

# Prefer langchain_openai for streaming
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore

try:
    from langchain.chat_models import init_chat_model  # type: ignore
    from langchain.messages import SystemMessage, HumanMessage, AIMessage  # type: ignore
    LANGCHAIN_AVAILABLE = True
except Exception:
    init_chat_model = None
    SystemMessage = None
    HumanMessage = None
    AIMessage = None
    LANGCHAIN_AVAILABLE = False

from .system_prompt import build_system_message


class LangChainLLMAdapter:
    """Chat generation adapter using LangChain (streaming when available)."""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.0, retriever: Optional[Any] = None):
        env_model = os.getenv("OPENAI_MODEL")
        self.model = (env_model or model)
        self.temperature = temperature
        self._chat = None
        self._retriever = retriever
        if ChatOpenAI is not None:
            try:
                self._chat = ChatOpenAI(model=self.model, temperature=self.temperature, streaming=True)
            except Exception:
                self._chat = None

    def _build_messages(self, question: str, history: List[Dict[str, str]], intent_info: Optional[Dict[str, Any]]):
        system_content = build_system_message(include_guards=True)
        if intent_info:
            try:
                intent = intent_info.get("intent")
                score = intent_info.get("score")
                entities = intent_info.get("entities") or {}
                spans = intent_info.get("spans") or []
                system_content += f" Intento detectado: {intent} (confianza {score:.2f}). Entidades: {entities}. Spans: {spans}."
            except Exception:
                pass
        facts = None
        try:
            if intent_info and "facts" in intent_info:
                facts = intent_info["facts"]
        except Exception:
            facts = None
        if facts:
            try:
                facts_text = "; ".join(f"{k}: {v}" for k, v in facts.items())
                system_content += f" Datos conocidos del usuario: {facts_text}."
            except Exception:
                pass
        # Knowledge base (RAG) context
        if self._retriever:
            try:
                meta_filter = {}
                qlow = question.lower()
                if "pib" in qlow:
                    meta_filter = {"topic": "pib"}
                elif "imacec" in qlow:
                    meta_filter = {"topic": "imacec"}
                elif "estacional" in qlow or "estacionalidad" in qlow or "see100" in qlow:
                    meta_filter = {"topic": "seasonality"}
                # Refine filter using detected entities (joint BIO model)
                try:
                    ent = intent_info.get("entities") if intent_info else {}
                except Exception:
                    ent = {}
                indicator = str(ent.get("indicator") or "").lower()
                if indicator:
                    if "pib" in indicator:
                        meta_filter = {"topic": "pib"}
                    elif "imacec" in indicator:
                        meta_filter = {"topic": "imacec"}
                sector = str(ent.get("sector") or "").lower()
                if sector and meta_filter.get("topic") == "imacec":
                    # Optionally include sector into filter if your chunks store it
                    meta_filter["sector"] = sector
                season = str(ent.get("seasonality") or "").lower()
                if season:
                    meta_filter["seasonality"] = season

                docs = []
                # New langchain_postgres retrievers are Runnable: prefer invoke({"query": ...})
                if callable(getattr(self._retriever, "invoke", None)):
                    try:
                        payload = {"query": question}
                        if meta_filter:
                            payload["filter"] = meta_filter
                        docs = self._retriever.invoke(payload)  # type: ignore
                    except Exception:
                        # some retrievers accept the raw string
                        docs = self._retriever.invoke(question)  # type: ignore
                elif hasattr(self._retriever, "get_relevant_documents"):
                    try:
                        docs = self._retriever.get_relevant_documents(question, filter=meta_filter)  # type: ignore
                    except Exception:
                        docs = self._retriever.get_relevant_documents(question)  # type: ignore
                try:
                    logger.debug("RAG retrieval | filter=%s | docs=%s | entities=%s", meta_filter, len(docs or []), ent)
                except Exception:
                    pass
                if docs:
                    context = "\n\n".join(getattr(d, "page_content", "") or "" for d in docs if getattr(d, "page_content", ""))
                    if context.strip():
                        system_content += f"\nContexto de base de conocimiento (RAG):\n{context}\nResponde citando este contexto y no agregues información externa."
                else:
                    system_content += "\nNo se recuperó contexto RAG relevante; si la respuesta depende de fuentes, indica que no dispones de datos y evita especular."
            except Exception as exc:
                logger.debug("Failed to fetch RAG context: %s", exc)

        if history:
            try:
                ctx = "\n".join(f"{h.get('role')}: {h.get('content')}" for h in history[-8:])
                system_content += (
                    f" Historial reciente (útil para recordar datos del usuario):\n{ctx}\n"
                    "Si el usuario pregunta por datos ya mencionados (nombre, estudios, etc.), respóndelos coherentemente."
                )
            except Exception:
                pass
        system = SystemMessage(content=system_content)
        msgs: List[Any] = [system]
        if history:
            try:
                context_text = "\n".join(f"{h.get('role')}: {h.get('content')}" for h in history[-8:])
                msgs.append(SystemMessage(content=f"Contexto reciente (usa esta información para responder preguntas personales del usuario):\n{context_text}"))
            except Exception:
                pass
        for h in history:
            role = h.get("role")
            content = h.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            else:
                msgs.append(AIMessage(content=content))
        msgs.append(HumanMessage(content=question))
        return msgs

    def stream(self, question: str, history: Optional[List[Dict[str, str]]] = None, intent_info: Optional[Dict[str, Any]] = None) -> Iterable[str]:
        """Yield assistant reply chunks (streaming when backend supports it)."""
        history = history or []
        if not LANGCHAIN_AVAILABLE or init_chat_model is None:
            yield f"Respuesta de prueba a: {question}. (LangChain no está disponible.)"
            return
        try:
            msgs = self._build_messages(question, history, intent_info)
            if self._chat is not None:
                for chunk in self._chat.stream(msgs):
                    text = getattr(chunk, "content", None) or getattr(chunk, "text", None) or ""
                    if text:
                        yield str(text)
                # cierre amable para fomentar follow-up
                yield FOLLOW_UP_SUFFIX
                return
            # fallback: non-streaming invoke
            llm = init_chat_model(model=self.model, temperature=self.temperature)
            out = llm.invoke(msgs)
            if hasattr(out, "content"):
                yield f"{out.content}{FOLLOW_UP_SUFFIX}"
            elif isinstance(out, list) and out:
                yield f"{getattr(out[0], 'content', str(out[0]))}{FOLLOW_UP_SUFFIX}"
            else:
                yield f"{str(out)}{FOLLOW_UP_SUFFIX}"
        except Exception:
            logger.exception("LangChainLLMAdapter.stream failed")
            yield f"(Error generando con LangChain) {question}"

    def generate(self, question: str, history: Optional[List[Dict[str, str]]] = None, intent_info: Optional[Dict[str, Any]] = None) -> str:
        """Compatibility: return full text by joining stream."""
        return "".join(self.stream(question, history=history or [], intent_info=intent_info))
