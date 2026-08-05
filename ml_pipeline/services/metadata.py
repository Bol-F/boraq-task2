from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
from django.conf import settings

from ml_pipeline.services.tracking import EXPERIMENT_NAME, TrackingSummary
from ml_pipeline.services.training import RANDOM_STATE, TrainingSummary

MODEL_VERSION = "1.0.0"


@dataclass(frozen=True)
class SavedArtifacts:
    """Paths and metadata for the locally saved winning model."""

    model_path: Path
    metadata_path: Path
    metadata: dict[str, object]


def get_model_path() -> Path:
    """Return the standard path for the complete winning pipeline."""
    return Path(settings.BASE_DIR) / "models" / "model.pkl"


def get_metadata_path() -> Path:
    """Return the standard path for model metadata."""
    return Path(settings.BASE_DIR) / "models" / "model_metadata.json"


def build_model_metadata(
    summary: TrainingSummary,
    tracking: TrackingSummary,
    *,
    dataset_rows: int,
    feature_count: int,
) -> dict[str, object]:
    """Build serializable metadata from actual winner metrics and run details."""
    winner = summary.winner
    return {
        "model_name": winner.model_name,
        "model_version": MODEL_VERSION,
        "roc_auc": winner.metrics.roc_auc,
        "pr_auc": winner.metrics.pr_auc,
        "f1": winner.metrics.f1,
        "training_date": datetime.now(UTC).isoformat(),
        "dataset_rows": int(dataset_rows),
        "feature_count": int(feature_count),
        "random_state": RANDOM_STATE,
        "selection_metric": "roc_auc",
        "mlflow_experiment": EXPERIMENT_NAME,
        "mlflow_experiment_id": tracking.experiment_id,
        "mlflow_run_id": tracking.run_ids[winner.model_name],
    }


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)


def save_pipeline(pipeline: object, destination: Path) -> None:
    """Write a complete fitted pipeline and atomically replace the old file."""
    temporary_path = _temporary_path(destination)
    try:
        joblib.dump(pipeline, temporary_path)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_metadata(metadata: dict[str, object], destination: Path) -> None:
    """Write readable JSON metadata and atomically replace the old file."""
    temporary_path = _temporary_path(destination)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, indent=2)
            file.write("\n")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_training_artifacts(
    summary: TrainingSummary,
    tracking: TrackingSummary,
    *,
    dataset_rows: int,
    feature_count: int,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> SavedArtifacts:
    """Save the winning pipeline and its calculated metadata together."""
    selected_model_path = model_path or get_model_path()
    selected_metadata_path = metadata_path or get_metadata_path()
    metadata = build_model_metadata(
        summary,
        tracking,
        dataset_rows=dataset_rows,
        feature_count=feature_count,
    )

    save_pipeline(summary.winner.pipeline, selected_model_path)
    save_metadata(metadata, selected_metadata_path)
    return SavedArtifacts(
        model_path=selected_model_path,
        metadata_path=selected_metadata_path,
        metadata=metadata,
    )
