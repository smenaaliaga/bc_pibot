"""Standalone orchestrator that wires LangChain adapters together."""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional

from .langchain_memory import LangChainMemoryAdapter
from .langchain_llm import LangChainLLMAdapter
from .classifier_agent import (
	classify_question,
	classify_question_with_history,
	build_intent_info,
)
from .data_router import can_handle_data, stream_data_flow

logger = logging.getLogger(__name__)

DEFAULT_WELCOME = (
	"Hola, soy PIBot, asistente económico del Banco Central de Chile. ¿En qué puedo ayudarte hoy?"
)


class LangChainOrchestrator:
	"""Minimal orchestrator that coordinates memory, classifier, and LLM adapters."""

	def __init__(
		self,
		*,
		memory: Optional[LangChainMemoryAdapter] = None,
		llm: Optional[LangChainLLMAdapter] = None,
		intent_classifier: Optional[Any] = None,
		intent_memory: Optional[Any] = None,
		intent_store: Optional[Any] = None,
	) -> None:
		self.memory = memory or LangChainMemoryAdapter()
		self.llm = llm or LangChainLLMAdapter()
		self.intent_classifier = intent_classifier
		self.intent_memory = intent_memory
		self.intent_store = intent_store
		self._locks: Dict[str, threading.RLock] = {}

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------
	def _session_lock(self, session_id: str) -> threading.RLock:
		if session_id not in self._locks:
			self._locks[session_id] = threading.RLock()
		return self._locks[session_id]

	def _build_history(self, session_id: str, history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
		if history:
			return history
		try:
			if hasattr(self.memory, "get_recent_history"):
				msgs = self.memory.get_recent_history(session_id)  # type: ignore[attr-defined]
			else:
				msgs = self.memory._load_messages(session_id)  # type: ignore[attr-defined]
			return [{"role": m.get("role", ""), "content": m.get("content", "")} for m in msgs]
		except Exception:
			logger.exception("build_history failed", extra={"session_id": session_id})
			return []

	def _write_classifier_fallback(self, summary: Dict[str, Any]) -> None:
		try:
			active = os.getenv("RUN_MAIN_LOG_ACTIVE")
			if not active:
				logs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "logs")
				os.makedirs(logs_dir, exist_ok=True)
				active = os.path.join(logs_dir, os.getenv("RUN_MAIN_LOG", "run_main.log"))
			with open(active, "a", encoding="utf-8") as f:
				f.write(
					"[CLASSIFIER_STREAM] query_type={query_type} data_domain={data_domain} "
					"is_generic={is_generic} default_key={default_key} error={error}\n".format(**summary)
				)
		except Exception:
			logger.debug("Could not append classifier fallback log", exc_info=True)

	def _chunk(self, text: str, chunk_size: int) -> Iterable[str]:
		chunk_size = max(chunk_size, 1)
		sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
		if not sentences:
			for i in range(0, len(text), chunk_size):
				yield text[i : i + chunk_size]
			return
		buf: List[str] = []
		buf_len = 0
		for sentence in sentences:
			add_len = len(sentence) + (1 if buf else 0)
			if buf and buf_len + add_len > chunk_size:
				yield " ".join(buf)
				buf = [sentence]
				buf_len = len(sentence)
			else:
				buf.append(sentence)
				buf_len += add_len
		if buf:
			yield " ".join(buf)

	def _enrich_intent_info(
		self,
		session_id: str,
		question: str,
		hist: List[Dict[str, str]],
		base_info: Optional[Dict[str, Any]],
	) -> Optional[Dict[str, Any]]:
		info: Dict[str, Any] = dict(base_info or {})
		if self.intent_classifier is None:
			return info or None
		try:
			intent, score, spans, entities = self.intent_classifier.predict(question, history=hist)
			info.setdefault("intent_classifier", intent)
			info.setdefault("intent_classifier_score", float(score or 0.0))
			info.setdefault("spans", [])
			info.setdefault("entities", {})
			info["spans"] = list(info["spans"]) + list(spans or [])
			info["entities"] = dict(info["entities"])  # shallow copy
			info["entities"].setdefault("classifier", entities or {})
			# If the classifier provides a confident intent, surface it to the LLM
			if not info.get("intent") or info.get("intent") == "unknown":
				info["intent"] = intent
				info["score"] = float(score or 0.0)
			# Facts from memory (if available)
			facts = None
			if hasattr(self.memory, "get_facts"):
				try:
					facts = self.memory.get_facts(session_id)  # type: ignore[attr-defined]
				except Exception:
					facts = None
			if facts:
				info["facts"] = facts
			turn_id = len(hist)
			if self.intent_memory is not None:
				try:
					self.intent_memory.record(
						session_id,
						intent=intent,
						score=score,
						spans=spans,
						entities=entities,
						turn_id=turn_id,
					)
				except Exception:
					logger.exception("intent_memory.record failed", extra={"session_id": session_id})
			if self.intent_store is not None:
				try:
					self.intent_store.record(
						session_id,
						intent=intent,
						score=score,
						spans=spans,
						entities=entities,
						turn_id=turn_id,
					)
				except Exception:
					logger.exception("intent_store.record failed", extra={"session_id": session_id})
		except Exception:
			logger.exception("intent_classifier.predict failed", extra={"session_id": session_id})
		return info or None

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def stream(
		self,
		question: str,
		history: Optional[List[Dict[str, str]]] = None,
		session_id: Optional[str] = None,
		stream_chunks: bool = False,
		chunk_size: int = 400,
	) -> Iterable[str]:
		session_id = session_id or str(uuid.uuid4())
		log_extra = {"session_id": session_id}

		try:
			self.memory.on_user_turn(session_id, question)
		except Exception:
			logger.exception("memory.on_user_turn failed", extra=log_extra)

		lock = self._session_lock(session_id)
		with lock:
			hist = self._build_history(session_id, history)
			first_turn = not any((m.get("role") or "").lower() == "assistant" for m in hist)

			try:
				logging.getLogger().info("[CLASSIFIER] start question=%s", question[:200], extra=log_extra)
				cls, history_text = classify_question_with_history(question, hist)
			except Exception:
				logger.exception("classify_question_with_history failed", extra=log_extra)
				cls = classify_question(question)
				history_text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in hist)

			intent_info = None
			try:
				intent_info = build_intent_info(cls)
			except Exception:
				intent_info = None

			if intent_info is not None:
				log_payload = (
					intent_info.get("intent"),
					(intent_info.get("entities") or {}).get("data_domain"),
					(intent_info.get("entities") or {}).get("is_generic"),
					(intent_info.get("entities") or {}).get("default_key"),
					(intent_info.get("entities") or {}).get("error"),
				)
			else:
				log_payload = (
					getattr(cls, "query_type", None),
					getattr(cls, "data_domain", None),
					getattr(cls, "is_generic", None),
					getattr(cls, "default_key", None),
					getattr(cls, "error", None),
				)
			try:
				logging.getLogger().info(
					"[CLASSIFIER] query_type=%s data_domain=%s is_generic=%s default_key=%s error=%s",
					*log_payload,
					extra=log_extra,
				)
				summary_dict = {
					"query_type": getattr(cls, "query_type", None),
					"data_domain": getattr(cls, "data_domain", None),
					"is_generic": getattr(cls, "is_generic", None),
					"default_key": getattr(cls, "default_key", None),
					"error": getattr(cls, "error", None),
				}
				self._write_classifier_fallback(summary_dict)
			except Exception:
				logger.debug("Failed to emit classifier log", exc_info=True, extra=log_extra)

			intent_info = self._enrich_intent_info(session_id, question, hist, intent_info)

			use_data_router = os.getenv("USE_DATA_ROUTER", "false").lower() in {"1", "true", "yes"}
			if use_data_router and cls and can_handle_data(cls):
				try:
					for chunk in stream_data_flow(cls, question, history_text):
						if chunk:
							yield str(chunk)
					return
				except Exception:
					logger.exception("data router failed", extra=log_extra)

			try:
				parts: List[str] = []
				for piece in self.llm.stream(question, history=hist, intent_info=intent_info):
					if piece:
						parts.append(str(piece))
				full = "".join(parts).strip()
			except Exception:
				logger.exception("LLM generation failed", extra=log_extra)
				full = ""

			if not full:
				return

			bypass_tokens = ("más", "mas", "detalle", "detalles", "otra", "exacta", "exacto", "lista", "son")
			skip_diversity = any(tok in (question or "").lower() for tok in bypass_tokens)
			if skip_diversity:
				is_redundant, sim = False, 0.0
			else:
				try:
					is_redundant, sim = self.memory.diversity_check(session_id, full)
				except Exception:
					logger.exception("diversity_check failed", extra=log_extra)
					is_redundant, sim = False, 0.0

			final = full
			if is_redundant:
				try:
					rewritten = self.memory.diversity_maybe_rewrite(session_id, full, question)
					if rewritten:
						final = rewritten
				except Exception:
					logger.exception("diversity_maybe_rewrite failed", extra=log_extra)

			if first_turn:
				welcome = os.getenv("WELCOME_MESSAGE", DEFAULT_WELCOME)
				is_greeting = bool(re.search(r"^(hola|buenas|hey|hi)\b", (question or "").strip(), flags=re.IGNORECASE))
				if is_greeting:
					final = welcome

			try:
				self.memory.on_assistant_turn(session_id, final)
			except Exception:
				logger.exception("memory.on_assistant_turn failed", extra=log_extra)

			try:
				self.memory.diversity_register(session_id, final)
			except Exception:
				logger.exception("memory.diversity_register failed", extra=log_extra)

			if stream_chunks:
				for chunk in self._chunk(final, chunk_size):
					yield chunk
			else:
				yield final
