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


_SYSTEM_PROMPT = """
Eres un clasificador experto de consultas económicas para un chatbot del Banco Central de Chile (PIBot).
Tu tarea es analizar exclusivamente el TEXTO de la consulta del usuario y devolver parámetros estructurados
para que otro componente orqueste la respuesta.

REGLAS GENERALES:

1) Campos de salida (tool 'classify_economic_query'):
   - query_type: "DATA" o "METHODOLOGICAL".
   - data_domain: "IMACEC", "PIB", "PIB_REGIONAL" u "OTHER".
   - is_generic: booleano.
   - default_key: string opcional (por ejemplo "IMACEC", "PIB_TOTAL").
   - tree_imacec: objeto con detalles específicos de IMACEC (puede ir vacío).
   - tree_pibe: objeto con detalles específicos de PIB y PIB regional (puede ir vacío).
   - intent_frequency_change: string opcional que indica si el usuario pide cambio de frecuencia
     (por ejemplo "M->T", "M->A", etc.), o null si no aplica.

2) query_type:
   - Usa "DATA" cuando el usuario pide valores, cifras, series, gráficos, tablas o cambios
     de frecuencia de datos económicos, aunque no mencione años concretos.
   - Usa "METHODOLOGICAL" cuando pide definiciones, metodología, interpretaciones generales
     sin exigir explícitamente valores numéricos.

3) data_domain:
   - "IMACEC" para consultas sobre el Índice Mensual de Actividad Económica.
   - "PIB" para consultas sobre el Producto Interno Bruto nacional agregado.
   - "PIB_REGIONAL" para PIB por región.
   - "OTHER" para otras series o indicadores.

4) is_generic:
   - true cuando el usuario no pide un desglose muy específico (por sector, por región, por componente),
     sino que pregunta por el IMACEC o PIB en general.
   - false cuando la consulta es específica (por ejemplo, PIB de la región de Los Lagos,
     IMACEC minero, etc.).

5) default_key:
   - Para IMACEC genérico: "IMACEC".
   - Para PIB nacional genérico: "PIB_TOTAL".
   - Para PIB regional genérico (sin región explícita): "PIB_REGIONAL".
   - Para otros dominios u otras series, puedes dejarlo en null si no hay un default claro.

6) intent_frequency_change:
   - Si detectas que el usuario quiere cambiar la frecuencia de la MISMA serie consultada
     (por ejemplo, "pásame la misma serie en frecuencia trimestral" o "¿puedes mostrarla anual?"),
     marca este campo con un texto breve que resuma la intención (p.ej. "TO_T", "TO_M", "TO_A").
   - Solo marca esto cuando explícitamente hay referencia a la misma serie o a un cambio de frecuencia.

REGLAS ESPECIALES PARA "ÚLTIMO VALOR" IMACEC/PIB:

Debes prestar especial atención a consultas del tipo:
- "¿Cuál es el último valor del IMACEC?"
- "dame el valor del IMACEC"
- "cuál es el valor del imacec"
- "cual es el ultimo valor del imacec"
- "¿Cuál es el último valor del PIB?"
- "dame el valor del PIB"
- "cuál es el valor del PIB"
- "cual es el ultimo valor del pib"

Para TODAS estas variantes (y expresiones equivalentes, con mayúsculas/minúsculas o tildes distintas):

1) SIEMPRE clasifícalas como:
   - query_type = "DATA"
   - is_generic = true

2) data_domain:
   - Si la consulta menciona IMACEC explícitamente (imacec, IMACEC, índice mensual de actividad económica):
     data_domain = "IMACEC" y default_key = "IMACEC".
   - Si la consulta menciona PIB explícitamente (pib, PIB, producto interno bruto) SIN mencionar una región:
     data_domain = "PIB" y default_key = "PIB_TOTAL".

3) NO las clasifiques como "METHODOLOGICAL", aunque el usuario no mencione un año.

4) No te preocupes de detectar el año en estas consultas; el orquestador se encargará
   de localizar el último período disponible. Tu responsabilidad es
   SOLO etiquetarlas correctamente como DATA + dominio adecuado + is_generic=true.

EJEMPLOS DE CLASIFICACIÓN:

- Usuario: "Explícame qué es el IMACEC y cómo se calcula."
  -> query_type = "METHODOLOGICAL"
     data_domain = "IMACEC"
     is_generic = true
     default_key = "IMACEC"

- Usuario: "Muéstrame los datos del IMACEC 2024 versus 2023."
  -> query_type = "DATA"
     data_domain = "IMACEC"
     is_generic = false
     default_key = "IMACEC"

- Usuario: "Cuál es el último valor del IMACEC"
  -> query_type = "DATA"
     data_domain = "IMACEC"
     is_generic = true
     default_key = "IMACEC"

- Usuario: "Dame el valor del PIB"
  -> query_type = "DATA"
     data_domain = "PIB"
     is_generic = true
     default_key = "PIB_TOTAL"

- Usuario: "Muéstrame el PIB de la Región Metropolitana el año 2023"
  -> query_type = "DATA"
     data_domain = "PIB_REGIONAL"
     is_generic = false
     default_key = "PIB_REGIONAL"

- Usuario: "¿Puedes cambiar la frecuencia de la misma serie a trimestral?"
  -> query_type = "DATA"
     data_domain = "OTHER" (a menos que haya contexto muy claro)
     intent_frequency_change = "TO_T"

Devuelve SIEMPRE un objeto para la herramienta 'classify_economic_query'
respetando estos criterios.
""".strip()


def classify_query(question: str) -> ClassificationResult:
    """Llama a OpenAI con function calling para clasificar la consulta económica."""
    # Reglas deterministas previas: casos frecuentes IMACEC/PIB
    try:
        q_lower = (question or "").lower()
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", q_lower))
        ultimo_patterns = ["ultimo valor", "último valor", "valor", "el valor"]
        action_patterns = ["dame", "muéstrame", "muestrame", "entrega", "entregame", "entregáme", "datos"]
        def _matches_any(patterns):
            return any(pat in q_lower for pat in patterns)
        if not has_year:
            # IMACEC genérico: último/valor/acciones
            if "imacec" in q_lower and (_matches_any(ultimo_patterns) or _matches_any(action_patterns)):
                return ClassificationResult(
                    query_type="DATA",
                    data_domain="IMACEC",
                    is_generic=True,
                    default_key="IMACEC",
                    imacec=ImacecTree(),
                )
            # PIB nacional genérico (sin región): último/valor/acciones
            if "pib" in q_lower and "regional" not in q_lower and "región" not in q_lower and (_matches_any(ultimo_patterns) or _matches_any(action_patterns)):
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
        resp = _client.chat.completions.create(
            model=_settings.openai_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
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