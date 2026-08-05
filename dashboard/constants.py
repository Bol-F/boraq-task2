"""Shared customer form choices and labels for the Streamlit dashboard."""

from __future__ import annotations

MODEL_FEATURE_NAMES = (
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
)

GENDER_CHOICES = ("Female", "Male")
YES_NO_CHOICES = ("No", "Yes")
MULTIPLE_LINES_CHOICES = ("No", "No phone service", "Yes")
INTERNET_SERVICE_CHOICES = ("DSL", "Fiber optic", "No")
INTERNET_ADDON_CHOICES = ("No", "No internet service", "Yes")
CONTRACT_CHOICES = ("Month-to-month", "One year", "Two year")
PAYMENT_METHOD_CHOICES = (
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
)

RISK_INTERPRETATIONS = {
    "low": "The model estimates a relatively low chance of churn.",
    "medium": "The customer may need additional review.",
    "high": "The model estimates an elevated churn risk.",
}
