"""FastAPI app exposing /intent for macro/intent/context classification."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .logic import classify_intent


class IntentRequest(BaseModel):
    text: str

class LabelConfidence(BaseModel):
    label: int | str
    confidence: float

class IntentResponse(BaseModel):
    macro: LabelConfidence
    intent: LabelConfidence
    context: LabelConfidence


app = FastAPI(title="PIBot Intent API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/intent", response_model=IntentResponse)
def intent_endpoint(payload: IntentRequest) -> IntentResponse:
    result = classify_intent(payload.text)
    return IntentResponse(**result)
