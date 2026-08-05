from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ml_pipeline.services.preprocessing import (
    build_preprocessor,
    clean_model_features,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics used for churn model comparison."""

    roc_auc: float
    pr_auc: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class ModelResult:
    """A fitted candidate pipeline and its held-out metrics."""

    model_name: str
    pipeline: Pipeline
    metrics: ModelMetrics
    parameters: dict[str, object]


@dataclass(frozen=True)
class TrainingSummary:
    """All candidate results plus the winner chosen by ROC-AUC."""

    results: tuple[ModelResult, ...]
    winner: ModelResult
    input_example: pd.DataFrame
    training_rows: int
    test_rows: int


def get_model_candidates() -> dict[str, BaseEstimator]:
    """Return reproducible classifiers that account for class imbalance."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_model_pipeline(
    features: pd.DataFrame,
    classifier: BaseEstimator,
) -> Pipeline:
    """Build a complete cleaning, preprocessing, and classifier pipeline."""
    cleaned_features = clean_model_features(features)
    return Pipeline(
        steps=[
            (
                "feature_cleaning",
                FunctionTransformer(clean_model_features, validate=False),
            ),
            ("preprocessing", build_preprocessor(cleaned_features)),
            ("classifier", clone(classifier)),
        ]
    )


def split_training_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create one reproducible, stratified train/test split for all models."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def _positive_probabilities(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = pipeline.predict_proba(features)
    positive_class_index = list(pipeline.classes_).index(1)
    return probabilities[:, positive_class_index]


def evaluate_pipeline(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> ModelMetrics:
    """Calculate held-out metrics without using accuracy for selection."""
    positive_probabilities = _positive_probabilities(pipeline, features)
    predictions = pipeline.predict(features)
    return ModelMetrics(
        roc_auc=float(roc_auc_score(target, positive_probabilities)),
        pr_auc=float(average_precision_score(target, positive_probabilities)),
        f1=float(f1_score(target, predictions)),
    )


def train_and_compare(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    candidates: Mapping[str, BaseEstimator] | None = None,
) -> TrainingSummary:
    """Fit every candidate on one split and select the best ROC-AUC pipeline."""
    configured_candidates = candidates or get_model_candidates()
    if not configured_candidates:
        raise ValueError("At least one model candidate is required.")

    features_train, features_test, target_train, target_test = split_training_data(
        features,
        target,
    )
    results: list[ModelResult] = []

    for model_name, classifier in configured_candidates.items():
        pipeline = build_model_pipeline(features_train, classifier)
        pipeline.fit(features_train, target_train)
        metrics = evaluate_pipeline(pipeline, features_test, target_test)
        results.append(
            ModelResult(
                model_name=model_name,
                pipeline=pipeline,
                metrics=metrics,
                parameters=classifier.get_params(deep=False),
            )
        )

    winner = max(results, key=lambda result: result.metrics.roc_auc)
    return TrainingSummary(
        results=tuple(results),
        winner=winner,
        input_example=features_train.head(5).copy(),
        training_rows=len(features_train),
        test_rows=len(features_test),
    )
