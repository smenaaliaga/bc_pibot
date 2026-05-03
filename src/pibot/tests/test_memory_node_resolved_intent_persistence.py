from orchestrator.graph.nodes.memory import make_memory_node


class StubMemoryAdapter:
    def __init__(self):
        self.facts = {}

    def on_assistant_turn(self, session_id, output, metadata=None):
        return None

    def set_facts(self, session_id, facts):
        current = self.facts.get(session_id, {})
        current.update(facts)
        self.facts[session_id] = current

    def save_checkpoint(self, session_id, payload, metadata=None):
        return True


class StubIntentStore:
    def __init__(self):
        self.calls = []

    def record(
        self,
        session_id,
        intent,
        score,
        *,
        spans=None,
        entities=None,
        intent_raw=None,
        predict_raw=None,
        turn_id=0,
        model_version=None,
    ):
        self.calls.append(
            {
                "session_id": session_id,
                "intent": intent,
                "score": score,
                "spans": spans or [],
                "entities": entities or {},
                "intent_raw": intent_raw or {},
                "predict_raw": predict_raw or {},
                "turn_id": turn_id,
            }
        )

    def history(self, session_id, k=25):
        return []


def test_memory_node_persists_resolved_intent_event_for_followup_inheritance():
    memory_adapter = StubMemoryAdapter()
    intent_store = StubIntentStore()
    node = make_memory_node(memory_adapter, intent_store)

    state = {
        "session_id": "sess-memory-1",
        "user_turn_id": 7,
        "output": "respuesta",
        "intent": {
            "macro_cls": 1,
            "intent_cls": "method",
            "context_cls": "followup",
        },
        "intent_info": {
            "score": 0.81,
            "spans": [{"text": "detalles", "label": "O", "start": 0, "end": 8}],
            "entities": {},
            "predict_raw": {
                "interpretation": {
                    "entities_normalized": {
                        "indicator": ["imacec"],
                        "seasonality": ["nsa"],
                    }
                }
            },
        },
        "entities": [
            {
                "indicator": "imacec",
                "seasonality": ["nsa"],
                "frequency": ["m"],
                "period": ["2026-02-01", "2026-02-28"],
            }
        ],
    }

    node(state)

    assert intent_store.calls
    resolved = intent_store.calls[-1]
    assert resolved["session_id"] == "sess-memory-1"
    assert resolved["turn_id"] == 7
    assert resolved["intent"] == "method"
    assert resolved["intent_raw"]["routing"]["intent"]["label"] == "method"
    assert resolved["intent_raw"]["routing"]["macro"]["label"] == 1
    assert resolved["intent_raw"]["routing"]["context"]["label"] == "followup"
    assert resolved["predict_raw"]["interpretation"]["entities_normalized"]["indicator"] == "imacec"
