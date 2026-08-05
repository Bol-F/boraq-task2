from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def churn_dataframe() -> pd.DataFrame:
    """Return a small dataframe shaped like the IBM churn dataset."""
    return pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B", "0003-C", "0004-D"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "tenure": [1, 24, 6, 48],
            "PhoneService": ["No", "Yes", "Yes", "Yes"],
            "MultipleLines": [
                "No phone service",
                "No",
                "Yes",
                "No",
            ],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "No", "Yes", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No", "No internet service"],
            "TechSupport": ["No", "No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No", "No internet service"],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "Yes", "No", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Credit card (automatic)",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [29.85, 89.10, 45.00, 20.00],
            "TotalCharges": ["29.85", " ", "invalid", "960.0"],
            "Churn": ["No", "Yes", "No", "Yes"],
        }
    )
