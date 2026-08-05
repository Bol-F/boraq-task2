from __future__ import annotations

import json

import joblib
import pytest

from ml_pipeline.services import model_validation
from predictions.services.model_loader import (
    MODEL_UNAVAILABLE_DETAIL,
    ModelUnavailableError,
    get_model_bundle,
    reset_model_cache,
)


def test_model_bundle_cache_reuses_one_joblib_load(
    model_artifacts: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = model_validation.joblib.load
    loaded_paths: list[object] = []

    def counting_load(path: object) -> object:
        loaded_paths.append(path)
        return original_load(path)

    monkeypatch.setattr(model_validation.joblib, "load", counting_load)

    first_bundle = get_model_bundle()
    second_bundle = get_model_bundle()

    assert first_bundle is second_bundle
    assert loaded_paths == [model_artifacts.model_path]


def test_model_cache_can_be_reset_explicitly(model_artifacts: object) -> None:
    first_bundle = get_model_bundle()
    updated_version = "test-3.0.0"
    model_artifacts.metadata_path.write_text(
        json.dumps({"model_version": updated_version}),
        encoding="utf-8",
    )

    still_cached_bundle = get_model_bundle()
    reset_model_cache()
    reloaded_bundle = get_model_bundle()

    assert still_cached_bundle is first_bundle
    assert still_cached_bundle.model_version == model_artifacts.model_version
    assert reloaded_bundle is not first_bundle
    assert reloaded_bundle.model_version == updated_version


def test_incompatible_pipeline_is_reported_as_unavailable(
    model_artifacts: object,
) -> None:
    pipeline = joblib.load(model_artifacts.model_path)
    pipeline.steps = [("classifier", pipeline.named_steps["classifier"])]
    joblib.dump(pipeline, model_artifacts.model_path)
    reset_model_cache()

    with pytest.raises(ModelUnavailableError) as error_info:
        get_model_bundle()

    assert str(error_info.value) == MODEL_UNAVAILABLE_DETAIL


def test_failed_model_load_is_not_cached(model_artifacts: object) -> None:
    saved_model = model_artifacts.model_path.read_bytes()
    model_artifacts.model_path.unlink()

    with pytest.raises(ModelUnavailableError):
        get_model_bundle()

    model_artifacts.model_path.write_bytes(saved_model)

    recovered_bundle = get_model_bundle()

    assert recovered_bundle.model_version == model_artifacts.model_version
