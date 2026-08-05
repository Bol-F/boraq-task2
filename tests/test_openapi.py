from __future__ import annotations

import json

from rest_framework.test import APIClient


def test_openapi_schema_documents_prediction_contract(api_client: APIClient) -> None:
    response = api_client.get("/api/schema/", {"format": "json"})

    assert response.status_code == 200
    schema = response.json()
    prediction_operation = schema["paths"]["/api/predict/"]["post"]
    assert "requestBody" in prediction_operation
    assert set(prediction_operation["responses"]) >= {"200", "400", "503"}

    documented_schema = json.dumps(schema)
    for field_name in (
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ):
        assert field_name in documented_schema

    assert "No phone service" in documented_schema
    assert "No internet service" in documented_schema
    assert "High-risk customer" in documented_schema
    assert "Validation error" in documented_schema
    assert "Prediction model unavailable" in documented_schema


def test_openapi_schema_documents_health_readiness(api_client: APIClient) -> None:
    response = api_client.get("/api/schema/", {"format": "json"})

    assert response.status_code == 200
    health_operation = response.json()["paths"]["/api/health/"]["get"]
    assert set(health_operation["responses"]) >= {"200", "503"}


def test_api_documentation_page_is_available(api_client: APIClient) -> None:
    response = api_client.get("/api/docs/")

    assert response.status_code == 200
    assert b"swagger-ui" in response.content.lower()
