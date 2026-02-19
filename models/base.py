from dataclasses import dataclass
from typing import Optional

@dataclass
class LabeledScore:
    label: str
    confidence: Optional[float] = None


@dataclass
class IntentRouterOutput:
    # Nueva taxonomía
    macro_cls: Optional[LabeledScore] = None
    intent_cls: Optional[LabeledScore] = None
    context_cls: Optional[LabeledScore] = None
    # Compatibilidad con versiones anteriores
    intent: Optional[LabeledScore] = None
    context_mode: Optional[LabeledScore] = None


@dataclass
class SeriesInterpreterOutput:
    # Nuevas cabezas de clasificación
    calc_mode_cls: Optional[LabeledScore] = None
    frequency_cls: Optional[LabeledScore] = None
    activity_cls: Optional[LabeledScore] = None
    region_cls: Optional[LabeledScore] = None
    req_form_cls: Optional[LabeledScore] = None
    # Compatibilidad legacy
    calc_mode: Optional[LabeledScore] = None
    frequency: Optional[LabeledScore] = None
    activity: Optional[LabeledScore] = None
    region: Optional[LabeledScore] = None
    req_form: Optional[LabeledScore] = None
    indicator: Optional[LabeledScore] = None
    metric_type: Optional[LabeledScore] = None
    seasonality: Optional[LabeledScore] = None

class BaseClassifier:
    def predict(self, query: str):
        raise NotImplementedError
