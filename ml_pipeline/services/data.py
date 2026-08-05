from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests
from django.conf import settings

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/"
    "d5371f5d83a446ad5673cbcca3b814b926491f8a/"
    "data/Telco-Customer-Churn.csv"
)
DATASET_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DOWNLOAD_TIMEOUT_SECONDS = 30

REQUIRED_COLUMNS = (
    "customerID",
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
    "Churn",
)


class DatasetValidationError(ValueError):
    """Raised when a churn dataset is incomplete or invalid."""


@dataclass(frozen=True)
class DatasetSummary:
    """Small validated summary returned after loading or downloading data."""

    path: Path
    rows: int
    columns: int


def get_dataset_path() -> Path:
    """Return the standard local path for the churn CSV."""
    return Path(settings.BASE_DIR) / "data" / "telco_churn.csv"


def validate_required_columns(dataframe: pd.DataFrame) -> None:
    """Ensure that the dataframe contains the IBM churn dataset columns."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise DatasetValidationError(f"Dataset is missing required columns: {missing}")


def load_churn_data(path: Path) -> pd.DataFrame:
    """Read a churn CSV and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Churn dataset was not found at {path}")

    try:
        dataframe = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DatasetValidationError(
            f"Dataset is not a readable CSV: {error}"
        ) from error

    validate_required_columns(dataframe)
    return dataframe


def _response_chunks(response: requests.Response) -> Iterator[bytes]:
    """Yield non-empty response chunks for a streaming download."""
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if chunk:
            yield chunk


def download_churn_dataset(
    destination: Path,
    *,
    overwrite: bool = False,
    url: str = DATASET_URL,
    expected_sha256: str = DATASET_SHA256,
) -> DatasetSummary:
    """Download and validate the IBM dataset without silent overwrites."""
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Dataset already exists at {destination}. Use --overwrite to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            checksum = hashlib.sha256()

            with requests.get(
                url,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()
                for chunk in _response_chunks(response):
                    temporary_file.write(chunk)
                    checksum.update(chunk)

        actual_sha256 = checksum.hexdigest()
        if actual_sha256 != expected_sha256:
            raise DatasetValidationError(
                "Downloaded dataset checksum does not match the expected IBM file."
            )

        dataframe = load_churn_data(temporary_path)
        temporary_path.replace(destination)
        temporary_path = None

        return DatasetSummary(
            path=destination,
            rows=len(dataframe),
            columns=len(dataframe.columns),
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
