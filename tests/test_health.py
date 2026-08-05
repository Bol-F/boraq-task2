from __future__ import annotations

import json

import pytest
from rest_framework.test import APIClient

from predictions.services.model_loader import reset_model_cache

DEGRADED_HEALTH_RESPONSE = {
    "status": "degraded",
    "service": "churn-prediction-api",
    "model_loaded": False,
    "model_version": None,
}


def assert_degraded_health_response(
    api_client: APIClient,
    *private_details: object,
) -> None:
    """Assert readiness fails safely without leaking artifact details."""
    response = api_client.get("/api/health/")

    assert response.status_code == 503
    assert response.json() == DEGRADED_HEALTH_RESPONSE

    response_text = response.content.decode()
    for detail in private_details:
        assert str(detail) not in response_text


def test_health_endpoint_reports_loaded_model(
    api_client: APIClient,
    model_artifacts: object,
) -> None:
    response = api_client.get("/api/health/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "churn-prediction-api",
        "model_loaded": True,
        "model_version": model_artifacts.model_version,
    }


def test_health_endpoint_remains_available_without_model(
    api_client: APIClient,
    model_artifacts: object,
) -> None:
    model_artifacts.model_path.unlink()
    reset_model_cache()

    assert_degraded_health_response(api_client, model_artifacts.model_path)


def test_health_endpoint_reports_corrupted_model_safely(
    api_client: APIClient,
    model_artifacts: object,
) -> None:
    corruption_marker = "private-corrupted-model-detail"
    model_artifacts.model_path.write_text(corruption_marker, encoding="utf-8")
    reset_model_cache()

    assert_degraded_health_response(
        api_client,
        model_artifacts.model_path,
        corruption_marker,
    )


def test_health_endpoint_reports_missing_metadata_safely(
    api_client: APIClient,
    model_artifacts: object,
) -> None:
    model_artifacts.metadata_path.unlink()
    reset_model_cache()

    assert_degraded_health_response(api_client, model_artifacts.metadata_path)


@pytest.mark.parametrize(
    ("metadata", "private_detail"),
    [
        ("{not-valid-json", "not-valid-json"),
        (json.dumps({"model_version": ""}), "valid model version"),
    ],
    ids=["malformed-json", "invalid-version"],
)
def test_health_endpoint_reports_invalid_metadata_safely(
    api_client: APIClient,
    model_artifacts: object,
    metadata: str,
    private_detail: str,
) -> None:
    model_artifacts.metadata_path.write_text(metadata, encoding="utf-8")
    reset_model_cache()

    assert_degraded_health_response(
        api_client,
        model_artifacts.metadata_path,
        private_detail,
    )


def test_health_endpoint_rejects_post(api_client: APIClient) -> None:
    response = api_client.post("/api/health/", data={}, format="json")

    assert response.status_code == 405
