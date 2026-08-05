from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock

import joblib
import numpy as np
from django.conf import settings
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

from ml_pipeline.services.data import MODEL_FEATURE_COLUMNS

MODEL_UNAVAILABLE_DETAIL = "Prediction model is not available."
REQUIRED_PIPELINE_STEPS = (
    "feature_cleaning",
    "preprocessing",
    "classifier",
)


class ModelUnavailableError(RuntimeError):
    """Raised when a complete, compatible prediction bundle cannot be loaded."""


@dataclass(frozen=True)
class ModelBundle:
    """A validated pipeline and the version reported by the API."""

    pipeline: Pipeline
    model_version: str


_cache_lock = RLock()


def _validate_pipeline(pipeline: object) -> Pipeline:
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Saved model is not a scikit-learn Pipeline.")
    if tuple(pipeline.named_steps) != REQUIRED_PIPELINE_STEPS:
        raise ValueError("Saved pipeline has incompatible steps.")
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


def _load_model_version(metadata_path: Path) -> str:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("Model metadata must be a JSON object.")

    model_version = metadata.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("Model metadata does not contain a valid model version.")
    return model_version.strip()


@lru_cache(maxsize=1)
def _load_model_bundle(model_path_value: str, metadata_path_value: str) -> ModelBundle:
    try:
        metadata_path = Path(metadata_path_value)
        model_version = _load_model_version(metadata_path)

        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            pipeline = _validate_pipeline(joblib.load(model_path_value))

        return ModelBundle(pipeline=pipeline, model_version=model_version)
    except Exception as error:
        raise ModelUnavailableError(MODEL_UNAVAILABLE_DETAIL) from error


def get_model_bundle() -> ModelBundle:
    """Return the cached bundle, loading configured artifacts only when needed."""
    model_path = str(Path(settings.CHURN_MODEL_PATH))
    metadata_path = str(Path(settings.CHURN_MODEL_METADATA_PATH))
    with _cache_lock:
        return _load_model_bundle(model_path, metadata_path)


def reset_model_cache() -> None:
    """Clear the in-memory bundle so tests or a process can force a reload."""
    with _cache_lock:
        _load_model_bundle.cache_clear()
