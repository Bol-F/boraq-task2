from __future__ import annotations

import pytest

from ml_pipeline.services.data import get_dataset_path
from ml_pipeline.services.preprocessing import load_features_and_target
from ml_pipeline.services.training import get_model_candidates, train_and_compare

MINIMUM_ROC_AUC = 0.78

pytestmark = [pytest.mark.integration, pytest.mark.model_quality]


def test_logistic_regression_roc_auc_exceeds_quality_threshold() -> None:
    """Guard the production baseline against measurable quality regressions."""
    dataset_path = get_dataset_path()
    if not dataset_path.is_file():
        pytest.fail(
            "The real churn dataset is required for the model-quality gate. "
            "Run `uv run python manage.py download_churn_data` first."
        )

    features, target = load_features_and_target(dataset_path)
    logistic_regression = get_model_candidates()["logistic_regression"]
    summary = train_and_compare(
        features,
        target,
        candidates={"logistic_regression": logistic_regression},
    )
    measured_roc_auc = summary.winner.metrics.roc_auc

    assert measured_roc_auc > MINIMUM_ROC_AUC, (
        f"Model quality regressed: ROC-AUC={measured_roc_auc:.3f}, "
        f"required > {MINIMUM_ROC_AUC:.3f}"
    )
