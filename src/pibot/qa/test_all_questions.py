"""Test all 6 questions (Q0-Q5) against the LLM with current instructions."""
import os, sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from openai import OpenAI
from orchestrator.data.catalog_data_search import search_output_payloads
from orchestrator.data.response import (
    _build_initial_messages,
    _build_metric_priority_instruction,
    _build_seasonality_strict_instruction,
    _build_incomplete_period_instruction,
    _build_relative_period_fallback_instruction,
    _build_no_explicit_period_latest_instruction,
    _build_missing_activity_instruction,
    _build_annual_pib_completeness_instruction,
    TOOLS,
    handle_tool_call,
)

DATA_STORE_DIR = str(ROOT / "orchestrator" / "memory" / "data_store")
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TEMPERATURE = float(os.getenv("DATA_RESPONSE_TEMPERATURE", "0.35"))

QUESTIONS = [
    {
        "id": "Q0",
        "question": "Cual es el valor del imacec del ultimo mes",
        "entities": {
            "indicator_ent": "imacec", "seasonality_ent": "nsa", "frequency_ent": "m",
            "activity_ent": "imacec", "region_ent": None, "investment_ent": None,
            "price_ent": None, "period_ent": ["2026-02-01", "2026-02-28"],
            "calc_mode_cls": "original", "activity_cls": "none",
            "activity_cls_resolved": "none", "region_cls": "none",
            "investment_cls": "none", "req_form_cls": "point", "price": "enc", "hist": 0,
        },
        "search_kwargs": {
            "indicator": "imacec", "calc_mode": "original", "seasonality": "nsa",
            "frequency": "m", "price": "enc", "has_activity": 0, "has_region": 0,
            "has_investment": 0, "hist": 0,
        },
    },
    {
        "id": "Q1",
        "question": "Cual es el valor del ultimo imacec",
        "entities": {
            "indicator_ent": "imacec", "seasonality_ent": "nsa", "frequency_ent": "m",
            "activity_ent": "imacec", "region_ent": None, "investment_ent": None,
            "price_ent": None, "period_ent": ["2026-02-01", "2026-02-28"],
            "calc_mode_cls": "original", "activity_cls": "none",
            "activity_cls_resolved": "none", "region_cls": "none",
            "investment_cls": "none", "req_form_cls": "point", "price": "enc", "hist": 0,
        },
        "search_kwargs": {
            "indicator": "imacec", "calc_mode": "original", "seasonality": "nsa",
            "frequency": "m", "price": "enc", "has_activity": 0, "has_region": 0,
            "has_investment": 0, "hist": 0,
        },
    },
    {
        "id": "Q2",
        "question": "Cuanto crecio la economia el ultimo trimestre",
        "entities": {
            "indicator_ent": "pib", "seasonality_ent": "nsa", "frequency_ent": "q",
            "activity_ent": None, "region_ent": None, "investment_ent": None,
            "price_ent": None, "period_ent": ["2025-10-01", "2025-12-31"],
            "calc_mode_cls": "original", "activity_cls": "none",
            "activity_cls_resolved": "none", "region_cls": "none",
            "investment_cls": "none", "req_form_cls": "point", "price": "enc", "hist": 0,
        },
        "search_kwargs": {
            "indicator": "pib", "calc_mode": "original", "seasonality": "nsa",
            "frequency": "q", "price": "enc", "has_activity": 0, "has_region": 0,
            "has_investment": 0, "hist": 0,
        },
    },
    {
        "id": "Q3",
        "question": "Cual es el PIB del 2026",
        "entities": {
            "indicator_ent": "pib", "seasonality_ent": "nsa", "frequency_ent": "a",
            "activity_ent": None, "region_ent": None, "investment_ent": None,
            "price_ent": None, "period_ent": ["2026-01-01", "2026-12-31"],
            "calc_mode_cls": "original", "activity_cls": "none",
            "activity_cls_resolved": "none", "region_cls": "none",
            "investment_cls": "none", "req_form_cls": "point", "price": "enc", "hist": 0,
        },
        "search_kwargs": {
            "indicator": "pib", "calc_mode": "original", "seasonality": "nsa",
            "frequency": "a", "price": "enc", "has_activity": 0, "has_region": 0,
            "has_investment": 0, "hist": 0,
        },
    },
    {
        "id": "Q4",
        "question": "Cuanto crecio la economia el ultimo mes",
        "entities": {
            "indicator_ent": "imacec", "seasonality_ent": "nsa", "frequency_ent": "m",
            "activity_ent": "imacec", "region_ent": None, "investment_ent": None,
            "price_ent": None, "period_ent": ["2026-02-01", "2026-02-28"],
            "calc_mode_cls": "original", "activity_cls": "none",
            "activity_cls_resolved": "none", "region_cls": "none",
            "investment_cls": "none", "req_form_cls": "point", "price": "enc", "hist": 0,
        },
        "search_kwargs": {
            "indicator": "imacec", "calc_mode": "original", "seasonality": "nsa",
            "frequency": "m", "price": "enc", "has_activity": 0, "has_region": 0,
            "has_investment": 0, "hist": 0,
        },
    },
    {
        "id": "Q5",
        "question": "Que actividad impulso al PIB en Enero",
        "entities": {
            "indicator_ent": "pib", "seasonality_ent": "nsa", "frequency_ent": "q",
            "activity_ent": None, "region_ent": None, "investment_ent": None,
            "price_ent": None, "period_ent": ["2026-01-01", "2026-03-31"],
            "calc_mode_cls": "contribution", "activity_cls": "general",
            "activity_cls_resolved": "general", "region_cls": "none",
            "investment_cls": "none", "req_form_cls": "point", "price": "enc", "hist": 0,
        },
        "search_kwargs": {
            "indicator": "pib", "calc_mode": "contribution", "seasonality": "nsa",
            "frequency": "q", "price": "enc", "has_activity": 1, "has_region": 0,
            "has_investment": 0, "hist": 0,
        },
    },
]


def run_question(q):
    """Run a single question through the full function-calling loop."""
    question = q["question"]
    entities = q["entities"]
    calc_mode = entities["calc_mode_cls"]

    # Load observations
    matches = search_output_payloads(DATA_STORE_DIR, **q["search_kwargs"])
    if not matches:
        return f"[NO MATCHES for search_kwargs]"
    observations = matches[0]["payload"]

    # Build messages
    messages = _build_initial_messages()
    context_payload = {"calc_mode": calc_mode or None, "entities": entities}
    messages.append({
        "role": "system",
        "content": "CONTEXTO DE CLASIFICACION (usar para priorizar metricas): "
                   + json.dumps(context_payload, ensure_ascii=False),
    })

    for builder, kwargs in [
        (_build_metric_priority_instruction, {"calc_mode": calc_mode}),
        (_build_seasonality_strict_instruction, {"entities_ctx": entities, "observations": observations}),
        (_build_incomplete_period_instruction, {"question": question, "entities_ctx": entities, "observations": observations}),
        (_build_relative_period_fallback_instruction, {"question": question, "entities_ctx": entities, "observations": observations}),
        (_build_no_explicit_period_latest_instruction, {"question": question, "entities_ctx": entities, "observations": observations}),
        (_build_missing_activity_instruction, {"entities_ctx": entities, "observations": observations}),
        (_build_annual_pib_completeness_instruction, {"question": question, "entities_ctx": entities, "observations": observations}),
    ]:
        instr = builder(**kwargs)
        if instr:
            messages.append({"role": "system", "content": instr})

    messages.append({"role": "user", "content": question})

    # Function-calling loop
    work_messages = list(messages)
    tool_calls_log = []
    t0 = time.perf_counter()

    for iteration in range(16):
        response = client.chat.completions.create(
            model=MODEL, messages=work_messages, tools=TOOLS,
            temperature=TEMPERATURE, stream=False,
        )
        choice = response.choices[0]
        assistant_msg = choice.message

        if assistant_msg.tool_calls:
            work_messages.append(assistant_msg.model_dump())
            for tc in assistant_msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                result = handle_tool_call(fn_name, fn_args, observations)
                tool_calls_log.append(f"{fn_name}({json.dumps(fn_args, ensure_ascii=False)})")
                work_messages.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })
        else:
            elapsed = time.perf_counter() - t0
            final = assistant_msg.content or ""
            return {
                "response": final,
                "tools": tool_calls_log,
                "elapsed": elapsed,
                "tokens": response.usage.total_tokens,
            }

    return {"response": "[MAX ITERATIONS]", "tools": tool_calls_log, "elapsed": time.perf_counter() - t0, "tokens": 0}


if __name__ == "__main__":
    # Allow limiting to specific questions via CLI args
    ids = sys.argv[1:] if len(sys.argv) > 1 else [q["id"] for q in QUESTIONS]

    for q in QUESTIONS:
        if q["id"] not in ids:
            continue
        print(f"\n{'='*70}")
        print(f"{q['id']}: {q['question']}")
        print(f"{'='*70}")
        result = run_question(q)
        if isinstance(result, str):
            print(result)
            continue
        print(f"Tools: {', '.join(result['tools']) or 'none'}")
        print(f"Time: {result['elapsed']:.1f}s  |  Tokens: {result['tokens']}")
        print(f"\n--- RESPONSE ---")
        print(result["response"])
        print(f"--- END ---")
