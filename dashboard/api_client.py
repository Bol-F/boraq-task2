"""Small HTTP client used by Streamlit to communicate with the Django API."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"
PREDICTION_PATH = "/api/predict/"
HEALTH_PATH = "/api/health/"
REQUEST_TIMEOUT_SECONDS = 10

ErrorKind = Literal[
    "validation",
    "unavailable",
    "connection",
    "timeout",
    "request",
    "invalid_response",
    "http_error",
]
JsonObject = dict[str, object]
ResultData = TypeVar("ResultData")


@dataclass(frozen=True)
class ApiError:
    """Safe error information that the dashboard can display to a user."""

    kind: ErrorKind
    message: str
    status_code: int | None = None
    field_errors: JsonObject | None = None


@dataclass(frozen=True)
class ApiResult(Generic[ResultData]):
    """Either parsed API data or a display-safe API error."""

    data: ResultData | None = None
    error: ApiError | None = None
    status_code: int | None = None

    @property
    def is_success(self) -> bool:
        return self.data is not None and self.error is None


@dataclass(frozen=True)
class PredictionData:
    """Validated values from a successful prediction response."""

    churn_probability: float
    will_churn: bool
    risk: Literal["low", "medium", "high"]
    model_version: str


@dataclass(frozen=True)
class HealthData:
    """Validated readiness information returned by the Django API."""

    status: Literal["ok", "degraded"]
    service: str
    model_loaded: bool
    model_version: str | None


def get_api_base_url() -> str:
    """Read and normalize the configured API base URL."""
    configured_url = os.getenv("API_URL", DEFAULT_API_URL).strip()
    return (configured_url or DEFAULT_API_URL).rstrip("/")


def build_api_url(base_url: str, path: str) -> str:
    """Join a base URL and API path with exactly one separating slash."""
    normalized_base = (base_url.strip() or DEFAULT_API_URL).rstrip("/")
    normalized_path = path.strip("/")
    if not normalized_path:
        return f"{normalized_base}/"
    return f"{normalized_base}/{normalized_path}/"


def get_prediction_url(base_url: str | None = None) -> str:
    """Return the normalized prediction endpoint URL."""
    return build_api_url(base_url or get_api_base_url(), PREDICTION_PATH)


def get_health_url(base_url: str | None = None) -> str:
    """Return the normalized health endpoint URL."""
    return build_api_url(base_url or get_api_base_url(), HEALTH_PATH)


def _failure(
    kind: ErrorKind,
    message: str,
    *,
    status_code: int | None = None,
    field_errors: JsonObject | None = None,
) -> ApiResult[ResultData]:
    return ApiResult(
        error=ApiError(
            kind=kind,
            message=message,
            status_code=status_code,
            field_errors=field_errors,
        ),
        status_code=status_code,
    )


def _read_json(response: requests.Response) -> object:
    try:
        return response.json()
    except ValueError as error:
        raise ValueError("The API response was not valid JSON.") from error


def _parse_prediction_data(response_data: object) -> PredictionData:
    if not isinstance(response_data, dict):
        raise ValueError("Prediction response must be a JSON object.")

    probability = response_data.get("churn_probability")
    will_churn = response_data.get("will_churn")
    risk = response_data.get("risk")
    model_version = response_data.get("model_version")

    if isinstance(probability, bool) or not isinstance(probability, int | float):
        raise ValueError("Prediction probability is invalid.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Prediction probability is outside the valid range.")
    if not isinstance(will_churn, bool):
        raise ValueError("Churn prediction is invalid.")
    if risk not in {"low", "medium", "high"}:
        raise ValueError("Prediction risk is invalid.")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("Model version is invalid.")

    return PredictionData(
        churn_probability=float(probability),
        will_churn=will_churn,
        risk=risk,
        model_version=model_version.strip(),
    )


def _parse_health_data(response_data: object) -> HealthData:
    if not isinstance(response_data, dict):
        raise ValueError("Health response must be a JSON object.")

    health_status = response_data.get("status")
    service = response_data.get("service")
    model_loaded = response_data.get("model_loaded")
    model_version = response_data.get("model_version")

    if health_status not in {"ok", "degraded"}:
        raise ValueError("Health status is invalid.")
    if not isinstance(service, str) or not service.strip():
        raise ValueError("Health service is invalid.")
    if not isinstance(model_loaded, bool):
        raise ValueError("Model readiness is invalid.")
    if model_version is not None and not isinstance(model_version, str):
        raise ValueError("Health model version is invalid.")

    return HealthData(
        status=health_status,
        service=service.strip(),
        model_loaded=model_loaded,
        model_version=model_version,
    )


def predict_customer(
    payload: Mapping[str, object],
    *,
    base_url: str | None = None,
    timeout: int | float = REQUEST_TIMEOUT_SECONDS,
) -> ApiResult[PredictionData]:
    """Request one churn prediction from Django and return structured data."""
    try:
        response = requests.post(
            get_prediction_url(base_url),
            json=dict(payload),
            timeout=timeout,
        )
    except requests.Timeout:
        return _failure(
            "timeout",
            "The prediction request timed out. Please try again.",
        )
    except requests.ConnectionError:
        return _failure(
            "connection",
            "The dashboard could not connect to the prediction API.",
        )
    except requests.RequestException:
        return _failure(
            "request",
            "The prediction request could not be completed.",
        )

    if response.status_code == 400:
        try:
            response_data = _read_json(response)
        except ValueError:
            return _failure(
                "invalid_response",
                "The prediction API returned an unreadable response.",
                status_code=response.status_code,
            )
        field_errors = response_data if isinstance(response_data, dict) else None
        return _failure(
            "validation",
            "Please correct the highlighted customer information.",
            status_code=response.status_code,
            field_errors=field_errors,
        )

    if response.status_code == 503:
        return _failure(
            "unavailable",
            "The prediction model is not available right now.",
            status_code=response.status_code,
        )

    if response.status_code != 200:
        return _failure(
            "http_error",
            "The prediction API returned an unexpected response.",
            status_code=response.status_code,
        )

    try:
        prediction = _parse_prediction_data(_read_json(response))
    except ValueError:
        return _failure(
            "invalid_response",
            "The prediction API returned an unreadable response.",
            status_code=response.status_code,
        )

    return ApiResult(data=prediction, status_code=response.status_code)


def get_health_status(
    *,
    base_url: str | None = None,
    timeout: int | float = REQUEST_TIMEOUT_SECONDS,
) -> ApiResult[HealthData]:
    """Request API readiness without preventing the dashboard from loading."""
    try:
        response = requests.get(get_health_url(base_url), timeout=timeout)
    except requests.Timeout:
        return _failure(
            "timeout",
            "The API health check timed out.",
        )
    except requests.ConnectionError:
        return _failure(
            "connection",
            "The dashboard could not connect to the API.",
        )
    except requests.RequestException:
        return _failure(
            "request",
            "The API health check could not be completed.",
        )

    if response.status_code not in {200, 503}:
        return _failure(
            "http_error",
            "The health endpoint returned an unexpected response.",
            status_code=response.status_code,
        )

    try:
        health = _parse_health_data(_read_json(response))
    except ValueError:
        return _failure(
            "invalid_response",
            "The health endpoint returned an unreadable response.",
            status_code=response.status_code,
        )

    return ApiResult(data=health, status_code=response.status_code)
