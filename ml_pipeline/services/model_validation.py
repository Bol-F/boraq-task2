from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from ml_pipeline.services.data import MODEL_FEATURE_COLUMNS

REQUIRED_PIPELINE_STEPS = (
    "feature_cleaning",
    "preprocessing",
    "classifier",
)


def validate_model_pipeline(pipeline: object) -> Pipeline:
    """Return a compatible churn pipeline or raise a descriptive error."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Saved model is not a scikit-learn Pipeline.")
    if tuple(pipeline.named_steps) != REQUIRED_PIPELINE_STEPS:
        raise ValueError("Saved pipeline has incompatible steps.")
    if not callable(getattr(pipeline, "predict", None)):
        raise TypeError("Saved pipeline does not support prediction.")
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise TypeError("Saved pipeline does not support probability prediction.")

    classes = np.asarray(getattr(pipeline, "classes_", []))
    has_binary_classes = (
        classes.ndim == 1
        and classes.size == 2
        and np.count_nonzero(classes == 0) == 1
        and np.count_nonzero(classes == 1) == 1
    )
    if not has_binary_classes:
        raise ValueError("Saved pipeline does not contain binary classes 0 and 1.")

    feature_names = np.asarray(getattr(pipeline, "feature_names_in_", []))
    if (
        feature_names.ndim != 1
        or tuple(feature_names.tolist()) != MODEL_FEATURE_COLUMNS
    ):
        raise ValueError("Saved pipeline expects incompatible feature columns.")
    return pipeline


def load_validated_pipeline(model_path: Path) -> Pipeline:
    """Load and validate a trusted, locally generated joblib pipeline."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        pipeline = joblib.load(model_path)
    return validate_model_pipeline(pipeline)
