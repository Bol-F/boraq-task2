from __future__ import annotations

import pytest
from rest_framework.test import APIClient

from predictions.services.model_loader import (
    MODEL_UNAVAILABLE_DETAIL,
    reset_model_cache,
)


def test_successful_prediction_returns_complete_response(
    api_client: APIClient,
    high_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    response = api_client.post(
        "/api/predict/",
        data=high_risk_customer,
        format="json",
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "churn_probability",
        "will_churn",
        "risk",
        "model_version",
    }
    assert isinstance(body["churn_probability"], float)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["will_churn"], bool)
    assert body["risk"] in {"low", "medium", "high"}
    assert body["model_version"] == model_artifacts.model_version


def test_representative_low_risk_customer(
    api_client: APIClient,
    low_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    response = api_client.post(
        "/api/predict/",
        data=low_risk_customer,
        format="json",
    )

    assert response.status_code == 200
    assert response.json() == {
        "churn_probability": 0.2,
        "will_churn": False,
        "risk": "low",
        "model_version": model_artifacts.model_version,
    }


def test_representative_high_risk_customer(
    api_client: APIClient,
    high_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    response = api_client.post(
        "/api/predict/",
        data=high_risk_customer,
        format="json",
    )

    assert response.status_code == 200
    assert response.json() == {
        "churn_probability": 0.8,
        "will_churn": True,
        "risk": "high",
        "model_version": model_artifacts.model_version,
    }


def test_prediction_thresholds_use_the_returned_rounded_probability(
    api_client: APIClient,
    high_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    customer = {**high_risk_customer, "Contract": "One year"}

    response = api_client.post("/api/predict/", data=customer, format="json")

    assert response.status_code == 200
    assert response.json() == {
        "churn_probability": 0.5,
        "will_churn": True,
        "risk": "medium",
        "model_version": model_artifacts.model_version,
    }


def test_dataset_special_categorical_values_are_accepted(
    api_client: APIClient,
    low_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    customer = {
        **low_risk_customer,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
    }

    response = api_client.post("/api/predict/", data=customer, format="json")

    assert response.status_code == 200
    assert response.json()["model_version"] == model_artifacts.model_version


def test_prediction_rejects_missing_required_field(
    api_client: APIClient,
    high_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    customer = high_risk_customer.copy()
    customer.pop("gender")

    response = api_client.post("/api/predict/", data=customer, format="json")

    assert response.status_code == 400
    assert "gender" in response.json()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        pytest.param("InternetService", "Satellite", id="categorical-choice"),
        pytest.param("SeniorCitizen", 2, id="senior-citizen"),
        pytest.param("tenure", -1, id="negative-tenure"),
        pytest.param("tenure", 73, id="excessive-tenure"),
        pytest.param("MonthlyCharges", -0.01, id="negative-monthly-charges"),
        pytest.param("TotalCharges", -0.01, id="negative-total-charges"),
    ],
)
def test_prediction_rejects_invalid_field_values(
    field_name: str,
    invalid_value: object,
    api_client: APIClient,
    high_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    customer = {**high_risk_customer, field_name: invalid_value}

    response = api_client.post("/api/predict/", data=customer, format="json")

    assert response.status_code == 400
    assert field_name in response.json()


def test_prediction_rejects_unknown_fields(
    api_client: APIClient,
    high_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    customer = {**high_risk_customer, "customerID": "0001-TEST"}

    response = api_client.post("/api/predict/", data=customer, format="json")

    assert response.status_code == 400
    assert response.json() == {"customerID": ["Unknown field."]}


def test_prediction_rejects_malformed_json(api_client: APIClient) -> None:
    response = api_client.generic(
        "POST",
        "/api/predict/",
        data='{"gender":',
        content_type="application/json",
    )

    assert response.status_code == 400
    assert "detail" in response.json()


def test_prediction_rejects_get(api_client: APIClient) -> None:
    response = api_client.get("/api/predict/")

    assert response.status_code == 405


def test_prediction_returns_503_when_model_file_is_missing(
    api_client: APIClient,
    high_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    model_artifacts.model_path.unlink()
    reset_model_cache()

    response = api_client.post(
        "/api/predict/",
        data=high_risk_customer,
        format="json",
    )

    assert response.status_code == 503
    assert response.json() == {"detail": MODEL_UNAVAILABLE_DETAIL}


def test_prediction_returns_503_when_model_file_is_corrupted(
    api_client: APIClient,
    high_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    model_artifacts.model_path.write_bytes(b"not a valid joblib artifact")
    reset_model_cache()

    response = api_client.post(
        "/api/predict/",
        data=high_risk_customer,
        format="json",
    )

    assert response.status_code == 503
    assert response.json() == {"detail": MODEL_UNAVAILABLE_DETAIL}


def test_prediction_returns_503_when_metadata_file_is_missing(
    api_client: APIClient,
    high_risk_customer: dict[str, object],
    model_artifacts: object,
) -> None:
    model_artifacts.metadata_path.unlink()
    reset_model_cache()

    response = api_client.post(
        "/api/predict/",
        data=high_risk_customer,
        format="json",
    )

    assert response.status_code == 503
    assert response.json() == {"detail": MODEL_UNAVAILABLE_DETAIL}
