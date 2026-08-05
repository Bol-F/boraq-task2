from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import requests

from dashboard import api_client
from dashboard.api_client import ApiError, HealthData, PredictionData


@dataclass
class FakeResponse:
    """Small requests.Response replacement for offline dashboard tests."""

    status_code: int
    payload: object = None
    json_error: ValueError | None = None

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/",
        "  http://127.0.0.1:8000///  ",
    ],
)
def test_prediction_url_is_constructed_safely(base_url: str) -> None:
    assert (
        api_client.get_prediction_url(base_url) == "http://127.0.0.1:8000/api/predict/"
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8000/",
        "  http://127.0.0.1:8000///  ",
    ],
)
def test_health_url_is_constructed_safely(base_url: str) -> None:
    assert api_client.get_health_url(base_url) == "http://127.0.0.1:8000/api/health/"


def test_api_base_url_uses_environment_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_URL", "  https://api.example.test///  ")

    assert api_client.get_api_base_url() == "https://api.example.test"


def test_api_base_url_uses_local_default_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("API_URL", raising=False)

    assert api_client.get_api_base_url() == api_client.DEFAULT_API_URL


def test_successful_prediction_sends_json_and_returns_structured_data(
    monkeypatch: pytest.MonkeyPatch,
    high_risk_customer: dict[str, object],
) -> None:
    captured_request: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured_request.update(url=url, **kwargs)
        return FakeResponse(
            200,
            {
                "churn_probability": 0.8912,
                "will_churn": True,
                "risk": "high",
                "model_version": "1.0.0",
            },
        )

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.predict_customer(
        high_risk_customer,
        base_url="https://api.example.test/",
    )

    assert captured_request == {
        "url": "https://api.example.test/api/predict/",
        "json": high_risk_customer,
        "timeout": api_client.REQUEST_TIMEOUT_SECONDS,
    }
    assert result.is_success
    assert result.error is None
    assert result.status_code == 200
    assert result.data == PredictionData(
        churn_probability=0.8912,
        will_churn=True,
        risk="high",
        model_version="1.0.0",
    )


def test_prediction_validation_error_preserves_field_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_body = {
        "tenure": ["Ensure this value is greater than or equal to 0."],
    }
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(400, response_body),
    )

    result = api_client.predict_customer({"tenure": -1})

    assert not result.is_success
    assert result.data is None
    assert result.status_code == 400
    assert result.error == ApiError(
        kind="validation",
        message="Please correct the highlighted customer information.",
        status_code=400,
        field_errors=response_body,
    )


def test_prediction_service_unavailable_returns_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            503,
            {"detail": "Prediction model is not available."},
        ),
    )

    result = api_client.predict_customer({})

    assert not result.is_success
    assert result.status_code == 503
    assert result.error == ApiError(
        kind="unavailable",
        message="The prediction model is not available right now.",
        status_code=503,
    )


def test_prediction_connection_error_is_handled_without_internal_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def connection_failure(*args: object, **kwargs: object) -> FakeResponse:
        raise requests.ConnectionError("secret internal host information")

    monkeypatch.setattr(api_client.requests, "post", connection_failure)

    result = api_client.predict_customer({})

    assert result.error == ApiError(
        kind="connection",
        message="The dashboard could not connect to the prediction API.",
    )
    assert result.status_code is None
    assert "secret" not in result.error.message


def test_prediction_timeout_is_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout_failure(*args: object, **kwargs: object) -> FakeResponse:
        raise requests.Timeout("private timeout details")

    monkeypatch.setattr(api_client.requests, "post", timeout_failure)

    result = api_client.predict_customer({})

    assert result.error == ApiError(
        kind="timeout",
        message="The prediction request timed out. Please try again.",
    )
    assert result.status_code is None


def test_prediction_invalid_json_is_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            200,
            json_error=requests.exceptions.JSONDecodeError(
                "Expecting value",
                "<html>not JSON</html>",
                0,
            ),
        ),
    )

    result = api_client.predict_customer({})

    assert result.error == ApiError(
        kind="invalid_response",
        message="The prediction API returned an unreadable response.",
        status_code=200,
    )


def test_prediction_unexpected_http_response_is_handled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(500, {"debug": "private details"}),
    )

    result = api_client.predict_customer({})

    assert result.error == ApiError(
        kind="http_error",
        message="The prediction API returned an unexpected response.",
        status_code=500,
    )


def test_ready_health_response_uses_get_and_returns_structured_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_request: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        captured_request.update(url=url, **kwargs)
        return FakeResponse(
            200,
            {
                "status": "ok",
                "service": "churn-prediction-api",
                "model_loaded": True,
                "model_version": "1.0.0",
            },
        )

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    result = api_client.get_health_status(base_url="https://api.example.test/")

    assert captured_request == {
        "url": "https://api.example.test/api/health/",
        "timeout": api_client.REQUEST_TIMEOUT_SECONDS,
    }
    assert result.is_success
    assert result.status_code == 200
    assert result.data == HealthData(
        status="ok",
        service="churn-prediction-api",
        model_loaded=True,
        model_version="1.0.0",
    )


def test_degraded_health_response_remains_a_reachable_health_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            503,
            {
                "status": "degraded",
                "service": "churn-prediction-api",
                "model_loaded": False,
                "model_version": None,
            },
        ),
    )

    result = api_client.get_health_status()

    assert result.is_success
    assert result.error is None
    assert result.status_code == 503
    assert result.data == HealthData(
        status="degraded",
        service="churn-prediction-api",
        model_loaded=False,
        model_version=None,
    )
