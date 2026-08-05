from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline.services.data import (
    DatasetValidationError,
    load_churn_data,
    validate_required_columns,
)

CUSTOMER_ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
TOTAL_CHARGES_COLUMN = "TotalCharges"
TARGET_MAPPING = {"No": 0, "Yes": 1}


def clean_model_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return model features with identifiers removed and charges made numeric."""
    if TOTAL_CHARGES_COLUMN not in features.columns:
        raise DatasetValidationError(
            f"Dataset is missing required column: {TOTAL_CHARGES_COLUMN}"
        )

    cleaned = features.drop(
        columns=[CUSTOMER_ID_COLUMN, TARGET_COLUMN],
        errors="ignore",
    ).copy()
    cleaned[TOTAL_CHARGES_COLUMN] = pd.to_numeric(
        cleaned[TOTAL_CHARGES_COLUMN],
        errors="coerce",
    ).fillna(0.0)
    return cleaned


def encode_target(target: pd.Series) -> pd.Series:
    """Convert the Yes/No churn target to integer one/zero values."""
    normalized = target.astype("string").str.strip()
    invalid_values = sorted(
        value for value in normalized.dropna().unique() if value not in TARGET_MAPPING
    )
    if normalized.isna().any() or invalid_values:
        details = ", ".join(invalid_values) or "missing values"
        raise DatasetValidationError(
            f"Churn target must contain only Yes or No; found {details}."
        )

    return normalized.map(TARGET_MAPPING).astype("int8")


def prepare_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and split a raw churn dataframe into clean features and target."""
    validate_required_columns(dataframe)
    features = clean_model_features(dataframe)
    target = encode_target(dataframe[TARGET_COLUMN])
    return features, target


def load_features_and_target(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV and return features and target ready for model training."""
    dataframe = load_churn_data(path)
    return prepare_features_and_target(dataframe)


def detect_feature_columns(
    features: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Detect numerical and categorical feature names from pandas dtypes."""
    numerical_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numerical_columns
    ]
    return numerical_columns, categorical_columns


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build the shared scaler and one-hot preprocessing transformer."""
    numerical_columns, categorical_columns = detect_feature_columns(features)
    return ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
