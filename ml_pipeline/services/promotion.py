from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

from ml_pipeline.services.data import MODEL_FEATURE_COLUMNS
from ml_pipeline.services.model_validation import load_validated_pipeline

MINIMUM_ROC_AUC = 0.78
MAX_ROC_AUC_REGRESSION = 0.005

REQUIRED_METADATA_FIELDS = frozenset(
    {
        "model_name",
        "model_version",
        "roc_auc",
        "pr_auc",
        "f1",
        "training_date",
        "dataset_rows",
        "feature_count",
        "random_state",
        "selection_metric",
        "mlflow_experiment",
        "mlflow_experiment_id",
        "mlflow_run_id",
    }
)


class PromotionError(ValueError):
    """Base error raised when a candidate cannot be promoted safely."""


class InvalidArtifactError(PromotionError):
    """Raised when a model or metadata artifact is missing or invalid."""


class PromotionRejectedError(PromotionError):
    """Raised when valid candidate metrics do not satisfy promotion policy."""


@dataclass(frozen=True)
class PromotionPolicy:
    """Quality thresholds applied to every candidate model."""

    minimum_roc_auc: float = MINIMUM_ROC_AUC
    max_roc_auc_regression: float = MAX_ROC_AUC_REGRESSION


@dataclass(frozen=True)
class PromotionAssessment:
    """Metrics and comparison details for an approved candidate."""

    candidate_roc_auc: float
    current_roc_auc: float | None
    regression: float | None
    bootstrap: bool
    policy: PromotionPolicy


def _require_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise InvalidArtifactError(f"{label} must be a non-empty file.")


def _read_metadata(path: Path, label: str) -> dict[str, object]:
    _require_nonempty_file(path, label)
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as error:
        raise InvalidArtifactError(f"{label} is not valid JSON.") from error
    if not isinstance(metadata, dict):
        raise InvalidArtifactError(f"{label} must contain a JSON object.")
    return metadata


def _read_metric(
    metadata: Mapping[str, object],
    field_name: str,
    label: str,
) -> float:
    value = metadata.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidArtifactError(f"{label} {field_name} must be a number.")
    metric = float(value)
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise InvalidArtifactError(f"{label} {field_name} must be between 0 and 1.")
    return metric


def _require_nonempty_string(
    metadata: Mapping[str, object],
    field_name: str,
) -> None:
    value = metadata.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise InvalidArtifactError(
            f"Candidate metadata field {field_name} must be a non-empty string."
        )


def _validate_candidate_metadata(metadata: Mapping[str, object]) -> float:
    missing_fields = sorted(REQUIRED_METADATA_FIELDS - metadata.keys())
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise InvalidArtifactError(
            f"Candidate metadata is missing required fields: {missing}."
        )

    for field_name in (
        "model_name",
        "model_version",
        "training_date",
        "selection_metric",
        "mlflow_experiment",
        "mlflow_experiment_id",
        "mlflow_run_id",
    ):
        _require_nonempty_string(metadata, field_name)

    if metadata["selection_metric"] != "roc_auc":
        raise InvalidArtifactError(
            "Candidate metadata selection_metric must be roc_auc."
        )

    for field_name in ("dataset_rows", "feature_count", "random_state"):
        value = metadata[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvalidArtifactError(
                f"Candidate metadata field {field_name} must be an integer."
            )
    if metadata["dataset_rows"] <= 0:
        raise InvalidArtifactError("Candidate metadata dataset_rows must be positive.")
    if metadata["feature_count"] != len(MODEL_FEATURE_COLUMNS):
        raise InvalidArtifactError(
            "Candidate metadata feature_count does not match the model schema."
        )

    try:
        training_date = datetime.fromisoformat(str(metadata["training_date"]))
    except ValueError as error:
        raise InvalidArtifactError(
            "Candidate metadata training_date must be an ISO-8601 timestamp."
        ) from error
    if training_date.tzinfo is None:
        raise InvalidArtifactError(
            "Candidate metadata training_date must include a timezone."
        )

    candidate_roc_auc = _read_metric(metadata, "roc_auc", "Candidate metadata")
    _read_metric(metadata, "pr_auc", "Candidate metadata")
    _read_metric(metadata, "f1", "Candidate metadata")
    return candidate_roc_auc


def _validate_candidate_model(model_path: Path) -> None:
    _require_nonempty_file(model_path, "Candidate model")
    try:
        load_validated_pipeline(model_path)
    except Exception as error:
        raise InvalidArtifactError(
            "Candidate model is not a compatible churn prediction pipeline."
        ) from error


def assess_model_candidate(
    *,
    candidate_model_path: Path,
    candidate_metadata_path: Path,
    current_metadata_path: Path | None,
    allow_missing_current: bool = False,
    policy: PromotionPolicy | None = None,
) -> PromotionAssessment:
    """Validate artifacts and enforce the absolute and comparative quality gates."""
    configured_policy = policy or PromotionPolicy()
    _validate_candidate_model(candidate_model_path)
    candidate_metadata = _read_metadata(
        candidate_metadata_path,
        "Candidate metadata",
    )
    candidate_roc_auc = _validate_candidate_metadata(candidate_metadata)

    if candidate_roc_auc <= configured_policy.minimum_roc_auc:
        raise PromotionRejectedError(
            "Model promotion rejected: "
            f"ROC-AUC={candidate_roc_auc:.3f}, required > "
            f"{configured_policy.minimum_roc_auc:.3f}."
        )

    current_is_missing = (
        current_metadata_path is None or not current_metadata_path.exists()
    )
    if current_is_missing:
        if not allow_missing_current:
            raise InvalidArtifactError(
                "Current model metadata is required unless bootstrap promotion is "
                "explicitly enabled."
            )
        return PromotionAssessment(
            candidate_roc_auc=candidate_roc_auc,
            current_roc_auc=None,
            regression=None,
            bootstrap=True,
            policy=configured_policy,
        )

    assert current_metadata_path is not None
    current_metadata = _read_metadata(current_metadata_path, "Current metadata")
    current_roc_auc = _read_metric(
        current_metadata,
        "roc_auc",
        "Current metadata",
    )
    regression = current_roc_auc - candidate_roc_auc
    if candidate_roc_auc < (current_roc_auc - configured_policy.max_roc_auc_regression):
        raise PromotionRejectedError(
            "Model promotion rejected: candidate "
            f"ROC-AUC={candidate_roc_auc:.3f}, current ROC-AUC={current_roc_auc:.3f}, "
            f"regression={regression:.3f}, allowed <= "
            f"{configured_policy.max_roc_auc_regression:.3f}."
        )

    return PromotionAssessment(
        candidate_roc_auc=candidate_roc_auc,
        current_roc_auc=current_roc_auc,
        regression=regression,
        bootstrap=False,
        policy=configured_policy,
    )


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(64 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def promote_model_candidate(
    *,
    candidate_model_path: Path,
    candidate_metadata_path: Path,
    current_metadata_path: Path | None,
    output_dir: Path,
    allow_missing_current: bool = False,
    policy: PromotionPolicy | None = None,
) -> PromotionAssessment:
    """Export an approved model and metadata together to a new immutable directory."""
    assessment = assess_model_candidate(
        candidate_model_path=candidate_model_path,
        candidate_metadata_path=candidate_metadata_path,
        current_metadata_path=current_metadata_path,
        allow_missing_current=allow_missing_current,
        policy=policy,
    )
    if output_dir.exists():
        raise InvalidArtifactError("Promotion output directory already exists.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        promoted_model_path = temporary_dir / "model.pkl"
        promoted_metadata_path = temporary_dir / "model_metadata.json"
        shutil.copy2(candidate_model_path, promoted_model_path)
        shutil.copy2(candidate_metadata_path, promoted_metadata_path)

        if _sha256(candidate_model_path) != _sha256(promoted_model_path):
            raise InvalidArtifactError("Promoted model copy failed verification.")
        if _sha256(candidate_metadata_path) != _sha256(promoted_metadata_path):
            raise InvalidArtifactError("Promoted metadata copy failed verification.")

        temporary_dir.replace(output_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    return assessment
