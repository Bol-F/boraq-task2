from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from ml_pipeline.services.promotion import (
    MAX_ROC_AUC_REGRESSION,
    MINIMUM_ROC_AUC,
    InvalidArtifactError,
    PromotionRejectedError,
    assess_model_candidate,
    promote_model_candidate,
)

pytestmark = pytest.mark.unit


def _metadata(roc_auc: object = 0.84) -> dict[str, object]:
    return {
        "model_name": "logistic_regression",
        "model_version": "1.0.0",
        "roc_auc": roc_auc,
        "pr_auc": 0.63,
        "f1": 0.61,
        "training_date": "2026-08-06T03:00:00+00:00",
        "dataset_rows": 7043,
        "feature_count": 19,
        "random_state": 42,
        "selection_metric": "roc_auc",
        "mlflow_experiment": "telecom-churn",
        "mlflow_experiment_id": "1",
        "mlflow_run_id": "test-run-id",
    }


def _write_metadata(
    path: Path,
    roc_auc: object = 0.84,
) -> Path:
    path.write_text(json.dumps(_metadata(roc_auc)), encoding="utf-8")
    return path


@pytest.fixture
def promotion_artifacts(
    model_artifacts: object,
) -> tuple[Path, Path]:
    _write_metadata(model_artifacts.metadata_path)
    return model_artifacts.model_path, model_artifacts.metadata_path


def test_promotion_policy_constants_are_stable() -> None:
    assert MINIMUM_ROC_AUC == 0.78
    assert MAX_ROC_AUC_REGRESSION == 0.005


def test_candidate_above_baseline_and_current_model_is_promoted(
    tmp_path: Path,
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    _write_metadata(metadata_path, 0.85)
    current_metadata_path = _write_metadata(tmp_path / "current.json", 0.84)
    output_dir = tmp_path / "approved"

    assessment = promote_model_candidate(
        candidate_model_path=model_path,
        candidate_metadata_path=metadata_path,
        current_metadata_path=current_metadata_path,
        output_dir=output_dir,
    )

    assert assessment.candidate_roc_auc == 0.85
    assert assessment.current_roc_auc == 0.84
    assert assessment.regression == pytest.approx(-0.01)
    assert not assessment.bootstrap
    assert (output_dir / "model.pkl").read_bytes() == model_path.read_bytes()
    assert (output_dir / "model_metadata.json").read_bytes() == (
        metadata_path.read_bytes()
    )


def test_candidate_at_regression_tolerance_boundary_is_approved(
    tmp_path: Path,
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    _write_metadata(metadata_path, 0.835)
    current_metadata_path = _write_metadata(tmp_path / "current.json", 0.84)

    assessment = assess_model_candidate(
        candidate_model_path=model_path,
        candidate_metadata_path=metadata_path,
        current_metadata_path=current_metadata_path,
    )

    assert assessment.regression == pytest.approx(MAX_ROC_AUC_REGRESSION)


@pytest.mark.parametrize("roc_auc", [0.78, 0.77])
def test_candidate_must_strictly_exceed_absolute_quality_gate(
    tmp_path: Path,
    promotion_artifacts: tuple[Path, Path],
    roc_auc: float,
) -> None:
    model_path, metadata_path = promotion_artifacts
    _write_metadata(metadata_path, roc_auc)

    with pytest.raises(
        PromotionRejectedError,
        match=r"ROC-AUC=.*required > 0.780",
    ):
        assess_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=None,
            allow_missing_current=True,
        )

    assert not (tmp_path / "approved").exists()


def test_material_regression_is_rejected_without_creating_output(
    tmp_path: Path,
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    _write_metadata(metadata_path, 0.834)
    current_metadata_path = _write_metadata(tmp_path / "current.json", 0.84)
    output_dir = tmp_path / "approved"

    with pytest.raises(PromotionRejectedError, match=r"regression=0.006"):
        promote_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=current_metadata_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_missing_current_metadata_requires_explicit_bootstrap(
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts

    with pytest.raises(InvalidArtifactError, match="bootstrap"):
        assess_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=None,
        )

    assessment = assess_model_candidate(
        candidate_model_path=model_path,
        candidate_metadata_path=metadata_path,
        current_metadata_path=None,
        allow_missing_current=True,
    )

    assert assessment.bootstrap
    assert assessment.current_roc_auc is None


def test_invalid_existing_current_metadata_never_bootstraps(
    tmp_path: Path,
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    current_metadata_path = tmp_path / "current.json"
    current_metadata_path.write_text("not JSON", encoding="utf-8")

    with pytest.raises(InvalidArtifactError, match="not valid JSON"):
        assess_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=current_metadata_path,
            allow_missing_current=True,
        )


@pytest.mark.parametrize(
    "roc_auc",
    [True, "0.84", math.nan, math.inf, -0.1, 1.1],
    ids=["boolean", "string", "nan", "infinity", "negative", "above-one"],
)
def test_invalid_candidate_roc_auc_is_rejected(
    promotion_artifacts: tuple[Path, Path],
    roc_auc: object,
) -> None:
    model_path, metadata_path = promotion_artifacts
    _write_metadata(metadata_path, roc_auc)

    with pytest.raises(InvalidArtifactError, match="roc_auc"):
        assess_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=None,
            allow_missing_current=True,
        )


def test_missing_candidate_metadata_field_is_rejected(
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    incomplete_metadata = _metadata()
    incomplete_metadata.pop("model_name")
    metadata_path.write_text(json.dumps(incomplete_metadata), encoding="utf-8")

    with pytest.raises(InvalidArtifactError, match="model_name"):
        assess_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=None,
            allow_missing_current=True,
        )


def test_missing_or_corrupted_candidate_model_is_rejected(
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    model_path.unlink()

    with pytest.raises(InvalidArtifactError, match="non-empty"):
        assess_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=None,
            allow_missing_current=True,
        )

    model_path.write_bytes(b"not a joblib model")
    with pytest.raises(InvalidArtifactError, match="compatible"):
        assess_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=None,
            allow_missing_current=True,
        )


def test_incompatible_candidate_pipeline_is_rejected(
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    pipeline = joblib.load(model_path)
    pipeline.steps = [("classifier", pipeline.named_steps["classifier"])]
    joblib.dump(pipeline, model_path)

    with pytest.raises(InvalidArtifactError, match="compatible"):
        assess_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=None,
            allow_missing_current=True,
        )


def test_existing_promotion_directory_is_not_modified(
    tmp_path: Path,
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    output_dir = tmp_path / "approved"
    output_dir.mkdir()
    marker_path = output_dir / "keep.txt"
    marker_path.write_text("existing", encoding="utf-8")

    with pytest.raises(InvalidArtifactError, match="already exists"):
        promote_model_candidate(
            candidate_model_path=model_path,
            candidate_metadata_path=metadata_path,
            current_metadata_path=None,
            output_dir=output_dir,
            allow_missing_current=True,
        )

    assert marker_path.read_text(encoding="utf-8") == "existing"


def test_promotion_management_command_exports_approved_bundle(
    tmp_path: Path,
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    current_metadata_path = _write_metadata(tmp_path / "current.json", 0.84)
    output_dir = tmp_path / "approved"

    call_command(
        "promote_churn_model",
        "--candidate-model",
        str(model_path),
        "--candidate-metadata",
        str(metadata_path),
        "--current-metadata",
        str(current_metadata_path),
        "--output-dir",
        str(output_dir),
    )

    assert (output_dir / "model.pkl").is_file()
    assert (output_dir / "model_metadata.json").is_file()


def test_promotion_management_command_returns_nonzero_for_rejection(
    tmp_path: Path,
    promotion_artifacts: tuple[Path, Path],
) -> None:
    model_path, metadata_path = promotion_artifacts
    _write_metadata(metadata_path, MINIMUM_ROC_AUC)

    with pytest.raises(CommandError, match="required > 0.780"):
        call_command(
            "promote_churn_model",
            "--candidate-model",
            str(model_path),
            "--candidate-metadata",
            str(metadata_path),
            "--output-dir",
            str(tmp_path / "approved"),
            "--allow-missing-current",
        )
