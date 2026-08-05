"""Reusable form and presentation helpers for the Streamlit dashboard."""

from __future__ import annotations

import math
from collections.abc import Mapping

import streamlit as st

from dashboard.constants import (
    CONTRACT_CHOICES,
    GENDER_CHOICES,
    INTERNET_ADDON_CHOICES,
    INTERNET_SERVICE_CHOICES,
    MODEL_FEATURE_NAMES,
    MULTIPLE_LINES_CHOICES,
    PAYMENT_METHOD_CHOICES,
    YES_NO_CHOICES,
)

CustomerPayload = dict[str, str | int | float]


class DashboardValidationError(ValueError):
    """Raised when the dashboard cannot build a complete customer payload."""


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
        if not math.isfinite(value) or value < 0:
            raise DashboardValidationError(f"{field_name} cannot be negative.")

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
