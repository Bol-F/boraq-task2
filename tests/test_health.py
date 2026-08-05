import pytest
from rest_framework.test import APIClient


@pytest.fixture
def api_client() -> APIClient:
    """Return a client for making requests to the REST API."""
    return APIClient()


def test_health_endpoint_returns_http_200(api_client: APIClient) -> None:
    response = api_client.get("/api/health/")

    assert response.status_code == 200


def test_health_endpoint_returns_ok_status(api_client: APIClient) -> None:
    response = api_client.get("/api/health/")

    assert response.json()["status"] == "ok"


def test_health_endpoint_returns_service_name(api_client: APIClient) -> None:
    response = api_client.get("/api/health/")

    assert response.json()["service"] == "churn-prediction-api"


def test_health_endpoint_rejects_post(api_client: APIClient) -> None:
    response = api_client.post("/api/health/", data={}, format="json")

    assert response.status_code == 405
