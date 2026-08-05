from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline.services.data import DatasetValidationError, load_churn_data
from ml_pipeline.services.preprocessing import (
    build_preprocessor,
    detect_feature_columns,
    prepare_features_and_target,
)

pytestmark = pytest.mark.unit


def test_dataset_loading_reads_csv(
    sample_churn_path: Path,
) -> None:
    loaded = load_churn_data(sample_churn_path)

    assert len(loaded) == 12
    assert "Churn" in loaded.columns


def test_dataset_loading_requires_expected_columns(
    tmp_path: Path,
    churn_dataframe: pd.DataFrame,
) -> None:
    dataset_path = tmp_path / "churn.csv"
    churn_dataframe.drop(columns="Churn").to_csv(dataset_path, index=False)

    with pytest.raises(DatasetValidationError, match="Churn"):
        load_churn_data(dataset_path)


def test_total_charges_is_converted_to_a_numeric_column(
    churn_dataframe: pd.DataFrame,
) -> None:
    features, _target = prepare_features_and_target(churn_dataframe)

    assert pd.api.types.is_numeric_dtype(features["TotalCharges"])
    assert features.loc[1, "TotalCharges"] == pytest.approx(178.2)


def test_blank_and_invalid_total_charges_are_replaced_with_zero(
    churn_dataframe: pd.DataFrame,
) -> None:
    features, _target = prepare_features_and_target(churn_dataframe)

    assert features.loc[0, "TotalCharges"] == 0.0
    assert features.loc[10, "TotalCharges"] == 0.0


def test_customer_id_is_removed_from_model_features(
    churn_dataframe: pd.DataFrame,
) -> None:
    features, _target = prepare_features_and_target(churn_dataframe)

    assert "customerID" not in features.columns


def test_target_is_converted_to_zero_and_one(
    churn_dataframe: pd.DataFrame,
) -> None:
    _features, target = prepare_features_and_target(churn_dataframe)

    assert set(target.unique()) == {0, 1}
    assert target.value_counts().to_dict() == {0: 6, 1: 6}


def test_features_and_target_have_matching_row_counts(
    churn_dataframe: pd.DataFrame,
) -> None:
    features, target = prepare_features_and_target(churn_dataframe)

    assert len(features) == len(target) == len(churn_dataframe)


def test_feature_columns_are_detected_from_dtypes(
    churn_dataframe: pd.DataFrame,
) -> None:
    features, _target = prepare_features_and_target(churn_dataframe)

    numerical_columns, categorical_columns = detect_feature_columns(features)

    assert numerical_columns == [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    assert "gender" in categorical_columns
    assert set(numerical_columns).isdisjoint(categorical_columns)
    assert set(numerical_columns + categorical_columns) == set(features.columns)


def test_preprocessor_uses_scaling_and_unknown_safe_encoding(
    churn_dataframe: pd.DataFrame,
) -> None:
    features, _target = prepare_features_and_target(churn_dataframe)

    preprocessor = build_preprocessor(features)
    transformers = {
        name: transformer for name, transformer, _columns in preprocessor.transformers
    }

    assert isinstance(preprocessor, ColumnTransformer)
    assert isinstance(transformers["numerical"], StandardScaler)
    assert isinstance(transformers["categorical"], OneHotEncoder)
    assert transformers["categorical"].handle_unknown == "ignore"


def test_preprocessor_can_fit_and_transform(
    churn_dataframe: pd.DataFrame,
) -> None:
    features, _target = prepare_features_and_target(churn_dataframe)
    preprocessor = build_preprocessor(features)

    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert transformed.shape[1] > len(features.columns)


def test_preprocessor_accepts_unknown_categorical_values(
    churn_dataframe: pd.DataFrame,
) -> None:
    features, _target = prepare_features_and_target(churn_dataframe)
    preprocessor = build_preprocessor(features)
    fitted = preprocessor.fit_transform(features)
    unknown_customer = features.head(1).copy()
    unknown_customer.loc[:, "PaymentMethod"] = "Cryptocurrency"

    transformed = preprocessor.transform(unknown_customer)

    assert transformed.shape == (1, fitted.shape[1])
