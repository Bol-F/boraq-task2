from __future__ import annotations

from rest_framework.test import APIClient

from predictions.services.model_loader import reset_model_cache


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

    response = api_client.get("/api/health/")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "service": "churn-prediction-api",
        "model_loaded": False,
        "model_version": None,
    }


def test_health_endpoint_rejects_post(api_client: APIClient) -> None:
    response = api_client.post("/api/health/", data={}, format="json")

    assert response.status_code == 405
