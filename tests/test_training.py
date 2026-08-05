from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_pipeline.management.commands import train_churn_model as command_module
from ml_pipeline.services.data import get_dataset_path
from ml_pipeline.services.metadata import SavedArtifacts, save_training_artifacts
from ml_pipeline.services.preprocessing import load_features_and_target
from ml_pipeline.services.tracking import TrackingSummary
from ml_pipeline.services.training import (
    RANDOM_STATE,
    TrainingSummary,
    get_model_candidates,
    train_and_compare,
)


@pytest.fixture(scope="module")
def trained_summary(
    training_dataset: tuple[pd.DataFrame, pd.Series],
) -> TrainingSummary:
    features, target = training_dataset
    candidates = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=500,
            random_state=RANDOM_STATE,
        )
    }
    return train_and_compare(features, target, candidates=candidates)


@pytest.fixture
def saved_artifacts(
    tmp_path: Path,
    trained_summary: TrainingSummary,
) -> SavedArtifacts:
    tracking = TrackingSummary(
        experiment_id="test-experiment",
        run_ids={"logistic_regression": "test-run"},
        tracking_uri="sqlite:///test.db",
    )
    return save_training_artifacts(
        trained_summary,
        tracking,
        dataset_rows=80,
        feature_count=5,
        model_path=tmp_path / "model.pkl",
        metadata_path=tmp_path / "model_metadata.json",
    )


def test_pipeline_training_returns_a_complete_fitted_pipeline(
    trained_summary: TrainingSummary,
) -> None:
    pipeline = trained_summary.winner.pipeline

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == [
        "feature_cleaning",
        "preprocessing",
        "classifier",
    ]


def test_prediction_probabilities_have_expected_shape_and_range(
    trained_summary: TrainingSummary,
    training_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, _target = training_dataset

    probabilities = trained_summary.winner.pipeline.predict_proba(features.head(6))

    assert probabilities.shape == (6, 2)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_saved_model_can_be_loaded_and_used_directly(
    saved_artifacts: SavedArtifacts,
    training_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    loaded_pipeline = joblib.load(saved_artifacts.model_path)
    features, _target = training_dataset
    raw_features = features.head(2).copy()
    raw_features["TotalCharges"] = [" ", "invalid"]

    probabilities = loaded_pipeline.predict_proba(raw_features)

    assert isinstance(loaded_pipeline, Pipeline)
    assert probabilities.shape == (2, 2)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_metadata_file_contains_actual_training_information(
    saved_artifacts: SavedArtifacts,
) -> None:
    metadata = json.loads(saved_artifacts.metadata_path.read_text(encoding="utf-8"))

    assert metadata["model_name"] == "logistic_regression"
    assert metadata["model_version"] == "1.0.0"
    assert metadata["roc_auc"] == saved_artifacts.metadata["roc_auc"]
    assert metadata["pr_auc"] == saved_artifacts.metadata["pr_auc"]
    assert metadata["f1"] == saved_artifacts.metadata["f1"]
    assert metadata["dataset_rows"] == 80
    assert metadata["feature_count"] == 5
    assert metadata["random_state"] == 42
    assert datetime.fromisoformat(metadata["training_date"]).tzinfo is not None


def test_training_command_reports_a_missing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_path = tmp_path / "missing.csv"
    monkeypatch.setattr(command_module, "get_dataset_path", lambda: missing_path)

    with pytest.raises(CommandError, match="download_churn_data"):
        call_command("train_churn_model")


@pytest.mark.integration
def test_logistic_regression_roc_auc_exceeds_quality_threshold() -> None:
    dataset_path = get_dataset_path()
    if not dataset_path.is_file():
        pytest.skip("Run download_churn_data before the local quality test.")

    features, target = load_features_and_target(dataset_path)
    candidates = {"logistic_regression": get_model_candidates()["logistic_regression"]}

    summary = train_and_compare(features, target, candidates=candidates)

    assert summary.winner.metrics.roc_auc > 0.78
