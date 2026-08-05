"""Reusable form and presentation helpers for the Streamlit dashboard."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import streamlit as st

from dashboard.api_client import ApiError, PredictionData
from dashboard.constants import (
    CONTRACT_CHOICES,
    GENDER_CHOICES,
    INTERNET_ADDON_CHOICES,
    INTERNET_SERVICE_CHOICES,
    MODEL_FEATURE_NAMES,
    MULTIPLE_LINES_CHOICES,
    PAYMENT_METHOD_CHOICES,
    RISK_INTERPRETATIONS,
    YES_NO_CHOICES,
)

CustomerPayload = dict[str, str | int | float]


class DashboardValidationError(ValueError):
    """Raised when the dashboard cannot build a complete customer payload."""


@dataclass(frozen=True)
class RiskPresentation:
    """Text and Streamlit status style for one risk level."""

    label: str
    interpretation: str
    status: Literal["success", "warning", "error"]


def build_customer_payload(values: Mapping[str, object]) -> CustomerPayload:
    """Return the exact JSON-compatible feature payload expected by Django."""
    missing_fields = [field for field in MODEL_FEATURE_NAMES if field not in values]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise DashboardValidationError(f"Complete all required fields: {missing}.")

    for field_name in ("tenure", "MonthlyCharges", "TotalCharges"):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DashboardValidationError(f"{field_name} must be a number.")
        if not math.isfinite(value):
            raise DashboardValidationError(f"{field_name} must be a finite number.")
        if value < 0:
            raise DashboardValidationError(f"{field_name} cannot be negative.")
        if field_name == "tenure" and value > 72:
            raise DashboardValidationError("tenure cannot exceed 72 months.")

    empty_fields = [
        field_name
        for field_name in MODEL_FEATURE_NAMES
        if isinstance(values[field_name], str) and not values[field_name].strip()
    ]
    if empty_fields:
        empty = ", ".join(empty_fields)
        raise DashboardValidationError(f"Complete all required fields: {empty}.")

    payload: CustomerPayload = {}
    for field_name in MODEL_FEATURE_NAMES:
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, str | int | float):
            raise DashboardValidationError(f"{field_name} has an invalid value.")
        payload[field_name] = value
    return payload


def format_probability(probability: float) -> str:
    """Format a zero-to-one probability as a one-decimal percentage."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0 and 1.")
    return f"{probability:.1%}"


def get_risk_presentation(risk: str) -> RiskPresentation:
    """Return visible text and a status style for a supported risk value."""
    presentations = {
        "low": RiskPresentation(
            label="LOW",
            interpretation=RISK_INTERPRETATIONS["low"],
            status="success",
        ),
        "medium": RiskPresentation(
            label="MEDIUM",
            interpretation=RISK_INTERPRETATIONS["medium"],
            status="warning",
        ),
        "high": RiskPresentation(
            label="HIGH",
            interpretation=RISK_INTERPRETATIONS["high"],
            status="error",
        ),
    }
    try:
        return presentations[risk.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported risk level: {risk}") from error


def render_api_error(error: ApiError) -> None:
    """Display a safe API error and optional Django field validation details."""
    st.error(error.message)
    if not error.field_errors:
        return

    for field_name, messages in error.field_errors.items():
        if isinstance(messages, list):
            message_text = "; ".join(str(message) for message in messages)
        else:
            message_text = str(messages)
        st.markdown(f"- **{field_name}:** {message_text}")


def render_prediction_result(prediction: PredictionData) -> None:
    """Present a successful churn prediction without implying certainty."""
    st.divider()
    st.subheader("Prediction result")

    probability_column, churn_column, version_column = st.columns(3)
    probability_column.metric(
        "Churn probability",
        format_probability(prediction.churn_probability),
    )
    churn_column.metric(
        "Will churn (0.5 threshold)",
        "Yes" if prediction.will_churn else "No",
    )
    version_column.metric("Model version", prediction.model_version)

    risk = get_risk_presentation(prediction.risk)
    risk_message = f"{risk.label} RISK — {risk.interpretation}"
    getattr(st, risk.status)(risk_message)
    st.caption("This result is an estimate and should not be treated as certain.")


def render_customer_form() -> CustomerPayload | None:
    """Render all API customer fields and return a payload after submission."""
    with st.form("customer_churn_form", border=True):
        st.subheader("1. Customer information")
        customer_left, customer_right = st.columns(2)
        with customer_left:
            gender = st.selectbox("Gender", GENDER_CHOICES)
            senior_citizen = st.selectbox(
                "Senior citizen",
                (0, 1),
                format_func=lambda value: "Yes" if value else "No",
            )
        with customer_right:
            partner = st.selectbox("Partner", YES_NO_CHOICES, index=1)
            dependents = st.selectbox("Dependents", YES_NO_CHOICES)

        st.subheader("2. Account information")
        account_left, account_middle, account_right = st.columns(3)
        with account_left:
            tenure = st.slider("Tenure (months)", 0, 72, 5)
        with account_middle:
            contract = st.selectbox("Contract", CONTRACT_CHOICES)
        with account_right:
            paperless_billing = st.selectbox(
                "Paperless billing",
                YES_NO_CHOICES,
                index=1,
            )

        st.subheader("3. Phone and internet services")
        phone_left, phone_middle, phone_right = st.columns(3)
        with phone_left:
            phone_service = st.selectbox("Phone service", YES_NO_CHOICES, index=1)
        with phone_middle:
            multiple_lines = st.selectbox("Multiple lines", MULTIPLE_LINES_CHOICES)
        with phone_right:
            internet_service = st.selectbox(
                "Internet service",
                INTERNET_SERVICE_CHOICES,
                index=1,
            )

        internet_left, internet_middle, internet_right = st.columns(3)
        with internet_left:
            online_security = st.selectbox(
                "Online security",
                INTERNET_ADDON_CHOICES,
            )
            online_backup = st.selectbox("Online backup", INTERNET_ADDON_CHOICES)
        with internet_middle:
            device_protection = st.selectbox(
                "Device protection",
                INTERNET_ADDON_CHOICES,
            )
            tech_support = st.selectbox("Tech support", INTERNET_ADDON_CHOICES)
        with internet_right:
            streaming_tv = st.selectbox(
                "Streaming TV",
                INTERNET_ADDON_CHOICES,
                index=2,
            )
            streaming_movies = st.selectbox(
                "Streaming movies",
                INTERNET_ADDON_CHOICES,
                index=2,
            )

        st.subheader("4. Billing information")
        billing_left, billing_middle, billing_right = st.columns(3)
        with billing_left:
            payment_method = st.selectbox(
                "Payment method",
                PAYMENT_METHOD_CHOICES,
                index=2,
            )
        with billing_middle:
            monthly_charges = st.number_input(
                "Monthly charges",
                min_value=0.0,
                value=89.9,
                step=0.1,
                format="%.2f",
            )
        with billing_right:
            total_charges = st.number_input(
                "Total charges",
                min_value=0.0,
                value=450.5,
                step=0.1,
                format="%.2f",
            )

        submitted = st.form_submit_button(
            "Predict churn risk",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    return build_customer_payload(
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    )
