from dataclasses import dataclass
from typing import Optional

@dataclass
class LabeledScore:
    label: str
    confidence: Optional[float] = None


@dataclass
class IntentRouterOutput:
    intent: LabeledScore
    context_mode: LabeledScore


@dataclass
class SeriesInterpreterOutput:
    indicator: LabeledScore
    metric_type: LabeledScore
    seasonality: LabeledScore
    activity: LabeledScore
    frequency: LabeledScore
    calc_mode: LabeledScore
    req_form: LabeledScore

class BaseClassifier:
    def predict(self, query: str):
        raise NotImplementedError
