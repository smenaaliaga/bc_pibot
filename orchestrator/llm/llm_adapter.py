"""Chat LLM adapter with streaming support (LangChain backend)."""

from __future__ import annotations

import os
import logging
from typing import Optional, List, Dict, Any, Iterable

logger = logging.getLogger(__name__)

FOLLOW_UP_SUFFIX = ""

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

# Tipos y prompts inline
from typing import Literal
GuardrailMode = Literal["rag", "fallback"]


def _chunk_logs_enabled() -> bool:
    return os.getenv("STREAM_CHUNK_LOGS", "0").lower() in ("1", "true", "yes", "on")


def _build_system_prompt(mode: GuardrailMode = "rag") -> str:
    """Construye el prompt del sistema basado en el modo."""
    base = """Eres el asistente económico del Banco Central de Chile (PIBot).
Respondes SIEMPRE en español.

Ayudas con consultas sobre indicadores económicos chilenos (IMACEC, PIB).
- Responde de forma clara y concisa
- Usa los datos proporcionados cuando estén disponibles
- Si no tienes información, indícalo claramente
- No inventes datos numéricos"""
    
    if mode == "rag":
        return base + "\n\nMODO RAG: Usa el contexto de documentos recuperados para responder consultas metodológicas."
    return base + "\n\nMODO FALLBACK: Responde basándote en tu conocimiento general sobre economía chilena."


def _first_entity_value(entity: Any) -> str:
    if isinstance(entity, dict):
        for key in ("standard_name", "normalized", "label", "value", "text_normalized"):
            candidate = entity.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, str) and item.strip():
                        return item
        return ""
    if isinstance(entity, list):
        for item in entity:
            if isinstance(item, str) and item.strip():
                return item
        return ""
    if isinstance(entity, str):
        return entity
    return ""


class LLMAdapter:
    """Chat generation adapter using LangChain (streaming when available)."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        retriever: Optional[Any] = None,
        streaming: bool = True,
        mode: GuardrailMode = "rag",
    ):
        env_model = os.getenv("OPENAI_MODEL")
        self.model = (env_model or model)
        self.temperature = temperature
        self.streaming = streaming
        self._chat = None
        self._retriever = retriever
        self.mode: GuardrailMode = mode
        if ChatOpenAI is not None:
            try:
                self._chat = ChatOpenAI(model=self.model, temperature=self.temperature, streaming=self.streaming)
            except Exception:
                self._chat = None

    def _build_messages(self, question: str, history: List[Dict[str, str]], intent_info: Optional[Dict[str, Any]]):
        system_content = _build_system_prompt(mode=self.mode)
        if intent_info:
            try:
                intent = intent_info.get("intent")
                score = intent_info.get("score")
                entities = intent_info.get("normalized") or {}
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
                    ent = intent_info.get("normalized") if intent_info else {}
                except Exception:
                    ent = {}
                indicator = _first_entity_value(ent.get("indicator")).lower()
                if indicator:
                    if "pib" in indicator:
                        meta_filter = {"topic": "pib"}
                    elif "imacec" in indicator:
                        meta_filter = {"topic": "imacec"}
                sector = _first_entity_value(ent.get("sector")).lower()
                if sector and meta_filter.get("topic") == "imacec":
                    meta_filter["sector"] = sector
                season = _first_entity_value(ent.get("seasonality")).lower()
                if season:
                    meta_filter["seasonality"] = season

                docs = []
                if callable(getattr(self._retriever, "invoke", None)):
                    try:
                        payload = {"query": question}
                        if meta_filter:
                            payload["filter"] = meta_filter
                        docs = self._retriever.invoke(payload)  # type: ignore
                    except Exception:
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
                    context_chunks = []
                    max_chars = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "8000"))
                    current = 0
                    for d in docs:
                        page = getattr(d, "page_content", "") or ""
                        if not page:
                            continue
                        remaining = max_chars - current
                        if remaining <= 0:
                            break
                        snippet = page[:remaining]
                        context_chunks.append(snippet)
                        current += len(snippet)
                    context = "\n\n".join(context_chunks)
                    if context.strip():
                        system_content += (
                            "\nContexto de base de conocimiento (RAG):\n"
                            f"{context}\nResponde citando este contexto y evita inventar información fuera de estas referencias."
                        )
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
        if SystemMessage is None:
            return []
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
            yield f"Respuesta de prueba a: {question}. (Backend no disponible.)"
            return
        try:
            msgs = self._build_messages(question, history, intent_info)
            # If streaming is disabled, do a single invoke to avoid SSE parsing issues
            if self._chat is not None and not self.streaming:
                out = self._chat.invoke(msgs)
                text = getattr(out, "content", None) or str(out)
                if text:
                    yield str(text)
                return
            if self._chat is not None:
                try:
                    for chunk in self._chat.stream(msgs):
                        text = getattr(chunk, "content", None) or getattr(chunk, "text", None) or ""
                        if text:
                            try:
                                if _chunk_logs_enabled():
                                    logger.debug("[LLM_STREAM_CHUNK] %s", text[:200])
                            except Exception:
                                pass
                            yield str(text)
                finally:
                    # Ensure the stream is exhausted to avoid GeneratorExit
                    if hasattr(self._chat, "close"):
                        try:
                            self._chat.close()  # type: ignore
                        except Exception:
                            pass
                return
            llm = init_chat_model(model=self.model, temperature=self.temperature, streaming=self.streaming)
            if self.streaming and hasattr(llm, "stream"):
                try:
                    for chunk in llm.stream(msgs):
                        text = getattr(chunk, "content", None) or getattr(chunk, "text", None) or ""
                        if text:
                            # No per-chunk logs here; agent_graph handles optional logging
                            yield str(text)
                finally:
                    if hasattr(llm, "close"):
                        try:
                            llm.close()
                        except Exception:
                            pass
            else:
                out = llm.invoke(msgs)
                if hasattr(out, "content"):
                    yield f"{out.content}"
                elif isinstance(out, list) and out:
                    yield f"{getattr(out[0], 'content', str(out[0]))}"
                else:
                    yield f"{str(out)}"
        except Exception:
            logger.exception("LLMAdapter.stream failed")
            yield f"(Error generando) {question}"

    def generate(self, question: str, history: Optional[List[Dict[str, str]]] = None, intent_info: Optional[Dict[str, Any]] = None) -> str:
        """Return full text by joining stream."""
        return "".join(self.stream(question, history=history or [], intent_info=intent_info))

    # Alias for compatibility
    def invoke(self, question: str, history: Optional[List[Dict[str, str]]] = None, intent_info: Optional[Dict[str, Any]] = None):
        return self.generate(question, history=history, intent_info=intent_info)


def build_llm(
    streaming: bool = True,
    temperature: float = 0.0,
    retriever: Optional[Any] = None,
    mode: GuardrailMode = "rag",
) -> LLMAdapter:
    adapter = LLMAdapter(temperature=temperature, retriever=retriever, streaming=streaming, mode=mode)
    # streaming flag kept for signature compatibility; adapter handles streaming internally
    return adapter


# Backward compatibility
LangChainLLMAdapter = LLMAdapter
