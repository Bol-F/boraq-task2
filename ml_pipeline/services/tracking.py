from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
from django.conf import settings
from mlflow.models import infer_signature

from ml_pipeline.services.training import (
    RANDOM_STATE,
    TEST_SIZE,
    ModelResult,
    TrainingSummary,
)

EXPERIMENT_NAME = "telecom-churn"


@dataclass(frozen=True)
class TrackingSummary:
    """Identifiers for the local MLflow experiment and model runs."""

    experiment_id: str
    run_ids: dict[str, str]
    tracking_uri: str


def get_tracking_uri() -> str:
    """Use an environment override or a project-local SQLite MLflow store."""
    configured_uri = os.getenv("MLFLOW_TRACKING_URI")
    if configured_uri:
        return configured_uri

    database_path = Path(settings.BASE_DIR) / "mlflow.db"
    return f"sqlite:///{database_path.as_posix()}"


def _logged_parameters(result: ModelResult) -> dict[str, object]:
    parameters: dict[str, object] = {
        "model_name": result.model_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "selection_metric": "roc_auc",
    }
    parameters.update(
        {f"classifier_{name}": value for name, value in result.parameters.items()}
    )
    return parameters


def track_training_results(
    summary: TrainingSummary,
    *,
    tracking_uri: str | None = None,
) -> TrackingSummary:
    """Log one local MLflow run and complete pipeline for each candidate."""
    selected_tracking_uri = tracking_uri or get_tracking_uri()
    mlflow.set_tracking_uri(selected_tracking_uri)
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    run_ids: dict[str, str] = {}

    for result in summary.results:
        with mlflow.start_run(run_name=result.model_name) as run:
            mlflow.log_params(_logged_parameters(result))
            mlflow.log_metrics(result.metrics.as_dict())
            mlflow.set_tags(
                {
                    "model_name": result.model_name,
                    "selection_metric": "roc_auc",
                }
            )

            predicted_probabilities = result.pipeline.predict_proba(
                summary.input_example
            )
            signature = infer_signature(
                summary.input_example,
                predicted_probabilities,
            )
            mlflow.sklearn.log_model(
                sk_model=result.pipeline,
                name="model",
                input_example=summary.input_example,
                signature=signature,
                pyfunc_predict_fn="predict_proba",
                serialization_format="cloudpickle",
            )
            run_ids[result.model_name] = run.info.run_id

    return TrackingSummary(
        experiment_id=experiment.experiment_id,
        run_ids=run_ids,
        tracking_uri=selected_tracking_uri,
    )
