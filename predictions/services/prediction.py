from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from predictions.services.model_loader import (
    MODEL_UNAVAILABLE_DETAIL,
    ModelUnavailableError,
    get_model_bundle,
)

CHURN_THRESHOLD = 0.5
MEDIUM_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65
PROBABILITY_DECIMAL_PLACES = 4


@dataclass(frozen=True)
class PredictionResult:
    """Public churn prediction values returned by the API."""

    churn_probability: float
    will_churn: bool
    risk: str
    model_version: str

    def as_dict(self) -> dict[str, float | bool | str]:
        return {
            "churn_probability": self.churn_probability,
            "will_churn": self.will_churn,
            "risk": self.risk,
            "model_version": self.model_version,
        }


def get_risk_level(probability: float) -> str:
    """Map a churn probability to the documented risk bands."""
    if probability < MEDIUM_RISK_THRESHOLD:
        return "low"
    if probability < HIGH_RISK_THRESHOLD:
        return "medium"
    return "high"


def predict_customer_churn(customer: Mapping[str, object]) -> PredictionResult:
    """Run validated customer fields through the complete cached pipeline."""
    try:
        bundle = get_model_bundle()
        input_frame = pd.DataFrame([dict(customer)])
        probabilities = np.asarray(bundle.pipeline.predict_proba(input_frame))
        classes = np.asarray(bundle.pipeline.classes_)
        positive_class_index = int(np.flatnonzero(classes == 1)[0])

        if probabilities.shape != (1, classes.size):
            raise ValueError("Prediction output has an unexpected shape.")

        probability = float(probabilities[0, positive_class_index])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Prediction probability is outside the valid range.")
        rounded_probability = round(probability, PROBABILITY_DECIMAL_PLACES)

        return PredictionResult(
            churn_probability=rounded_probability,
            will_churn=bool(rounded_probability >= CHURN_THRESHOLD),
            risk=get_risk_level(rounded_probability),
            model_version=bundle.model_version,
        )
    except ModelUnavailableError:
        raise
    except Exception as error:
        raise ModelUnavailableError(MODEL_UNAVAILABLE_DETAIL) from error
