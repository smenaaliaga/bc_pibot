"""Memory node helpers for PIBot graph."""

from __future__ import annotations

import logging
from typing import Any

from ..session import persist_agent_state
from ..state import AgentState
from ..suggestions import extract_chart_metadata_from_output, generate_suggested_questions

logger = logging.getLogger(__name__)


def make_memory_node(memory_adapter: Any, intent_store) -> Any:
    def memory_node(state: AgentState) -> AgentState:
        session_id = state.get("session_id")
        output = state.get("output", "")
        metadata = dict(state.get("memory_metadata") or {})
        chart_meta = extract_chart_metadata_from_output(output)
        if chart_meta:
            metadata.update(chart_meta)
        if memory_adapter and session_id:
            try:
                if output:
                    if metadata:
                        memory_adapter.on_assistant_turn(session_id, output, metadata=metadata)
                    else:
                        memory_adapter.on_assistant_turn(session_id, output)

                persist_agent_state(memory_adapter, session_id, state)
            except Exception:
                logger.debug("[GRAPH] Unable to persist assistant turn", exc_info=True)

        suggested_questions = generate_suggested_questions(state, intent_store)
        if not suggested_questions:
            suggested_questions = [
                "¿Quieres que busque los datos más recientes?",
                "¿Te muestro un gráfico con la última variación?",
                "¿Prefieres consultar IMACEC o PIB?",
            ]
        followup_block = "\n\n##FOLLOWUP_START\n"
        for i, question in enumerate(suggested_questions[:3], start=1):
            followup_block += f"suggestion_{i}={question}\n"
        followup_block += "##FOLLOWUP_END"
        output = output + followup_block

        return {"output": output}

    return memory_node


__all__ = ["make_memory_node"]
