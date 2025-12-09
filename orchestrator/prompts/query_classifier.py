"""
prompt.py
---------
Definición de dataclasses y prompt de clasificación (function calling)
para consultas económicas orientadas a IMACEC, PIB, PIB_REGIONAL u otras series.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Literal

from orchestrator.prompts.registry import build_classifier_prompt

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from config import get_settings
    _settings = get_settings()
except Exception:  # pragma: no cover
    class _StubSettings:
        openai_model = "gpt-4"
        openai_api_key = ""
    _settings = _StubSettings()

_client = OpenAI(api_key=_settings.openai_api_key) if (OpenAI and getattr(_settings, 'openai_api_key', None)) else None


@dataclass
class ImacecTree:
    nivel: Optional[str] = None
    sector: Optional[str] = None


@dataclass
class PibeTree:
    tipo: Optional[str] = None
    region: Optional[str] = None


@dataclass
class ClassificationResult:
    query_type: Optional[str] = None  # "DATA" o "METHODOLOGICAL"
    data_domain: Optional[str] = None  # "IMACEC" | "PIB" | "PIB_REGIONAL" | "OTHER"
    is_generic: bool = False
    default_key: Optional[str] = None
    imacec: Optional[ImacecTree] = None
    pibe: Optional[PibeTree] = None
    intent_frequency_change: Optional[str] = None
    error: Optional[str] = None


def classify_query(question: str) -> ClassificationResult:
    """Llama a OpenAI con function calling para clasificar la consulta económica."""
    # Reglas deterministas previas: casos frecuentes IMACEC/PIB
    try:
        q_lower = (question or "").lower()
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", q_lower))
        ultimo_patterns = ["ultimo valor", "último valor", "valor", "el valor"]
        ultimo_words = ["ultimo", "último"]
        action_patterns = ["dame", "muéstrame", "muestrame", "entrega", "entregame", "entregáme", "datos"]
        def _matches_any(patterns):
            return any(pat in q_lower for pat in patterns)
        if not has_year:
            # IMACEC genérico: último/valor/acciones
            if "imacec" in q_lower and (_matches_any(ultimo_patterns) or _matches_any(action_patterns) or _matches_any(ultimo_words)):
                return ClassificationResult(
                    query_type="DATA",
                    data_domain="IMACEC",
                    is_generic=True,
                    default_key="IMACEC",
                    imacec=ImacecTree(),
                )
            # PIB nacional genérico (sin región): último/valor/acciones
            if "pib" in q_lower and "regional" not in q_lower and "región" not in q_lower and (_matches_any(ultimo_patterns) or _matches_any(action_patterns) or _matches_any(ultimo_words)):
                return ClassificationResult(
                    query_type="DATA",
                    data_domain="PIB",
                    is_generic=True,
                    default_key="PIB_TOTAL",
                    pibe=PibeTree(),
                )
            # IMACEC índices + variación anual sin año
            if "imacec" in q_lower and (("indices" in q_lower or "índices" in q_lower) and ("variacion anual" in q_lower or "variación anual" in q_lower)):
                return ClassificationResult(
                    query_type="DATA",
                    data_domain="IMACEC",
                    is_generic=True,
                    default_key="IMACEC",
                    imacec=ImacecTree(),
                )
        else:
            # Año presente → genérico IMACEC/PIB nacional
            if "imacec" in q_lower:
                return ClassificationResult(
                    query_type="DATA",
                    data_domain="IMACEC",
                    is_generic=True,
                    default_key="IMACEC",
                    imacec=ImacecTree(),
                )
            if "pib" in q_lower and "regional" not in q_lower and "región" not in q_lower:
                return ClassificationResult(
                    query_type="DATA",
                    data_domain="PIB",
                    is_generic=True,
                    default_key="PIB_TOTAL",
                    pibe=PibeTree(),
                )
    except Exception:
        pass
    # Si cliente OpenAI no disponible, construir heurístico simple (solo post-determinista)
    if _client is None:
        # Heurística mínima adicional: si contiene 'pib' y 'region' -> PIB_REGIONAL DATA
        ql = q_lower if 'q_lower' in locals() else (question or '').lower()
        if 'pib' in ql and ('region' in ql or 'región' in ql):
            return ClassificationResult(query_type="DATA", data_domain="PIB_REGIONAL", is_generic=False, default_key="PIB_REGIONAL")
        # Default fallback
        return ClassificationResult(query_type=None, data_domain=None, is_generic=False, default_key=None, error="openai_client_unavailable")
    try:
        system_prompt = build_classifier_prompt()
        resp = _client.chat.completions.create(
            model=_settings.openai_model,
            messages=[
            {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "classify_economic_query",
                        "description": "Clasifica una consulta económica en términos de tipo, dominio de datos y otros parámetros estructurados.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query_type": {
                                    "type": "string",
                                    "enum": ["DATA", "METHODOLOGICAL"],
                                },
                                "data_domain": {
                                    "type": "string",
                                    "enum": [
                                        "IMACEC",
                                        "PIB",
                                        "PIB_REGIONAL",
                                        "OTHER",
                                    ],
                                },
                                "is_generic": {"type": "boolean"},
                                "default_key": {
                                    "type": ["string", "null"]
                                },
                                "tree_imacec": {
                                    "type": "object",
                                    "properties": {
                                        "nivel": {
                                            "type": ["string", "null"]
                                        },
                                        "sector": {
                                            "type": ["string", "null"]
                                        },
                                    },
                                    "required": ["nivel", "sector"],
                                },
                                "tree_pibe": {
                                    "type": "object",
                                    "properties": {
                                        "tipo": {
                                            "type": ["string", "null"]
                                        },
                                        "region": {
                                            "type": ["string", "null"]
                                        },
                                    },
                                    "required": ["tipo", "region"],
                                },
                                "intent_frequency_change": {
                                    "type": ["string", "null"]
                                },
                            },
                            "required": [
                                "query_type",
                                "data_domain",
                                "is_generic",
                            ],
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "classify_economic_query"}},
        )
        msg = resp.choices[0].message
        import json
        if getattr(msg, "tool_calls", None):
            tool_call = msg.tool_calls[0]
            args = tool_call.function.arguments  # type: ignore[attr-defined]
            parsed = json.loads(args)
        else:
            # Fallback: intentar parsear content como JSON
            try:
                parsed = json.loads(getattr(msg, "content", "") or "{}")
            except Exception:
                ql = (question or "").lower()
                if 'pib' in ql and 'regional' not in ql and 'región' not in ql:
                    return ClassificationResult(query_type="DATA", data_domain="PIB", is_generic=True, default_key="PIB_TOTAL")
                if 'imacec' in ql:
                    return ClassificationResult(query_type="DATA", data_domain="IMACEC", is_generic=True, default_key="IMACEC")
                return ClassificationResult(error="llm_no_tool_calls")
        im_tree = parsed.get("tree_imacec") or {}
        pi_tree = parsed.get("tree_pibe") or {}
        return ClassificationResult(
            query_type=parsed.get("query_type"),
            data_domain=parsed.get("data_domain"),
            is_generic=bool(parsed.get("is_generic")),
            default_key=parsed.get("default_key"),
            imacec=ImacecTree(
                nivel=im_tree.get("nivel"), sector=im_tree.get("sector")
            )
            if im_tree
            else None,
            pibe=PibeTree(
                tipo=pi_tree.get("tipo"), region=pi_tree.get("region")
            )
            if pi_tree
            else None,
            intent_frequency_change=parsed.get("intent_frequency_change"),
            error=None,
        )
    except Exception as e:
        return ClassificationResult(
            query_type=None,
            data_domain=None,
            is_generic=False,
            default_key=None,
            imacec=None,
            pibe=None,
            intent_frequency_change=None,
            error=str(e),
        )
