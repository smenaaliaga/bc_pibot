"""Backward-compatible wrapper for the prompt registry guardrails."""

from __future__ import annotations

from orchestrator.prompts.registry import GuardrailMode, build_guardrail_prompt


def build_system_message(include_guards: bool = True, mode: GuardrailMode = "rag") -> str:
    """Return the guardrail prompt (kept for legacy imports)."""
    return build_guardrail_prompt(mode=mode, include_guards=include_guards)


__all__ = ["build_system_message"]
