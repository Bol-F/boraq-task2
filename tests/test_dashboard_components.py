from __future__ import annotations

import math

import pytest

from dashboard.components import (
    DashboardValidationError,
    RiskPresentation,
    build_customer_payload,
    format_probability,
    get_risk_presentation,
)
from dashboard.constants import MODEL_FEATURE_NAMES, RISK_INTERPRETATIONS


def test_customer_payload_contains_exact_api_fields(
    high_risk_customer: dict[str, object],
) -> None:
    form_values = {**high_risk_customer, "dashboard_only": "not sent"}

    payload = build_customer_payload(form_values)

    assert tuple(payload) == MODEL_FEATURE_NAMES
    assert payload == high_risk_customer
    assert isinstance(payload["SeniorCitizen"], int)
    assert isinstance(payload["tenure"], int)
    assert isinstance(payload["MonthlyCharges"], float)
    assert isinstance(payload["TotalCharges"], float)


def test_customer_payload_rejects_a_missing_required_field(
    high_risk_customer: dict[str, object],
) -> None:
    form_values = high_risk_customer.copy()
    form_values.pop("gender")

    with pytest.raises(DashboardValidationError, match="gender"):
        build_customer_payload(form_values)


def test_customer_payload_rejects_an_empty_required_field(
    high_risk_customer: dict[str, object],
) -> None:
    with pytest.raises(DashboardValidationError, match="PaymentMethod"):
        build_customer_payload({**high_risk_customer, "PaymentMethod": "  "})


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        pytest.param("tenure", -1, "cannot be negative", id="negative-tenure"),
        pytest.param("tenure", 73, "cannot exceed 72", id="excessive-tenure"),
        pytest.param(
            "MonthlyCharges",
            -0.01,
            "cannot be negative",
            id="negative-monthly-charges",
        ),
        pytest.param(
            "TotalCharges",
            -0.01,
            "cannot be negative",
            id="negative-total-charges",
        ),
        pytest.param(
            "MonthlyCharges",
            math.nan,
            "finite number",
            id="nan-monthly-charges",
        ),
        pytest.param(
            "TotalCharges",
            math.inf,
            "finite number",
            id="infinite-total-charges",
        ),
        pytest.param("tenure", True, "must be a number", id="boolean-tenure"),
        pytest.param(
            "MonthlyCharges",
            "89.9",
            "must be a number",
            id="string-monthly-charges",
        ),
    ],
)
def test_customer_payload_rejects_invalid_numeric_values(
    field_name: str,
    invalid_value: object,
    message: str,
    high_risk_customer: dict[str, object],
) -> None:
    with pytest.raises(DashboardValidationError, match=message):
        build_customer_payload(
            {**high_risk_customer, field_name: invalid_value},
        )


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        pytest.param(0.0, "0.0%", id="zero"),
        pytest.param(0.8912, "89.1%", id="example"),
        pytest.param(1.0, "100.0%", id="one"),
    ],
)
def test_probability_formatting(probability: float, expected: str) -> None:
    assert format_probability(probability) == expected


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01, math.nan, math.inf],
    ids=["negative", "above-one", "nan", "infinite"],
)
def test_probability_formatting_rejects_invalid_values(probability: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        format_probability(probability)


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        pytest.param(
            "low",
            RiskPresentation(
                label="LOW",
                interpretation=RISK_INTERPRETATIONS["low"],
                status="success",
            ),
            id="low",
        ),
        pytest.param(
            "medium",
            RiskPresentation(
                label="MEDIUM",
                interpretation=RISK_INTERPRETATIONS["medium"],
                status="warning",
            ),
            id="medium",
        ),
        pytest.param(
            "high",
            RiskPresentation(
                label="HIGH",
                interpretation=RISK_INTERPRETATIONS["high"],
                status="error",
            ),
            id="high",
        ),
    ],
)
def test_risk_interpretation_logic(
    risk: str,
    expected: RiskPresentation,
) -> None:
    assert get_risk_presentation(risk) == expected


def test_risk_interpretation_is_case_insensitive() -> None:
    assert get_risk_presentation("HIGH").label == "HIGH"


def test_risk_interpretation_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported risk level"):
        get_risk_presentation("critical")
