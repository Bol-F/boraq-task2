from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock

from django.conf import settings
from sklearn.pipeline import Pipeline

from ml_pipeline.services.model_validation import load_validated_pipeline

MODEL_UNAVAILABLE_DETAIL = "Prediction model is not available."


class ModelUnavailableError(RuntimeError):
    """Raised when a complete, compatible prediction bundle cannot be loaded."""


@dataclass(frozen=True)
class ModelBundle:
    """A validated pipeline and the version reported by the API."""

    pipeline: Pipeline
    model_version: str


_cache_lock = RLock()


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

        pipeline = load_validated_pipeline(Path(model_path_value))

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
