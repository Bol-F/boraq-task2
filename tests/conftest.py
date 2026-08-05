from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from rest_framework.test import APIClient
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ml_pipeline.services.data import MODEL_FEATURE_COLUMNS, load_churn_data
from ml_pipeline.services.preprocessing import load_features_and_target
from predictions.services.model_loader import reset_model_cache

TEST_MODEL_VERSION = "test-2.4.6"
TEST_FIXTURES_DIR = Path(__file__).parent / "fixtures"


class DeterministicChurnClassifier(ClassifierMixin, BaseEstimator):
    """Small fitted-looking classifier for deterministic API integration tests."""

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        features: pd.DataFrame,
        _target: object = None,
    ) -> DeterministicChurnClassifier:
        """Record fitted feature information expected by scikit-learn."""
        self.n_features_in_ = features.shape[1]
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return stable probabilities based on the customer's contract."""
        churn_probabilities = features["Contract"].map(
            {
                "Two year": 0.2,
                "One year": 0.49996,
                "Month-to-month": 0.8,
            }
        )
        probabilities = churn_probabilities.to_numpy(dtype=float)
        return np.column_stack((1.0 - probabilities, probabilities))


@dataclass(frozen=True)
class TemporaryModelArtifacts:
    """Paths and version for the isolated prediction artifacts used by tests."""

    model_path: Path
    metadata_path: Path
    model_version: str


def build_deterministic_pipeline() -> Pipeline:
    """Build a lightweight fitted pipeline with all production feature names."""
    example_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 50.0,
    }
    features = pd.DataFrame([example_customer], columns=MODEL_FEATURE_COLUMNS)
    pipeline = Pipeline(
        steps=[
            ("feature_cleaning", FunctionTransformer(validate=False)),
            ("preprocessing", FunctionTransformer(validate=False)),
            ("classifier", DeterministicChurnClassifier()),
        ]
    )
    return pipeline.fit(features, np.array([0]))


@pytest.fixture(autouse=True)
def reset_prediction_model_cache() -> Iterator[None]:
    """Prevent a model bundle cached by one test from leaking into another."""
    reset_model_cache()
    yield
    reset_model_cache()


@pytest.fixture
def api_client() -> APIClient:
    """Return a client for making requests to the REST API."""
    return APIClient()


@pytest.fixture
def high_risk_customer() -> dict[str, object]:
    """Return one valid month-to-month fiber customer request."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.9,
        "TotalCharges": 450.5,
    }


@pytest.fixture
def low_risk_customer() -> dict[str, object]:
    """Return one valid long-tenure two-year customer request."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
    }


@pytest.fixture
def model_artifacts(tmp_path: Path, settings: object) -> TemporaryModelArtifacts:
    """Configure Django to use a temporary valid model and metadata bundle."""
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "model_metadata.json"
    joblib.dump(build_deterministic_pipeline(), model_path)
    metadata_path.write_text(
        json.dumps({"model_version": TEST_MODEL_VERSION}),
        encoding="utf-8",
    )

    settings.CHURN_MODEL_PATH = model_path
    settings.CHURN_MODEL_METADATA_PATH = metadata_path
    reset_model_cache()

    return TemporaryModelArtifacts(
        model_path=model_path,
        metadata_path=metadata_path,
        model_version=TEST_MODEL_VERSION,
    )


@pytest.fixture(scope="session")
def sample_churn_path() -> Path:
    """Return the tracked, deterministic churn CSV used by fast tests."""
    return TEST_FIXTURES_DIR / "sample_churn.csv"


@pytest.fixture
def churn_dataframe(sample_churn_path: Path) -> pd.DataFrame:
    """Return an isolated copy of the complete sample churn dataset."""
    return load_churn_data(sample_churn_path).copy(deep=True)


@pytest.fixture(scope="module")
def training_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return all production features from the deterministic sample CSV."""
    return load_features_and_target(TEST_FIXTURES_DIR / "sample_churn.csv")
