from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from ml_pipeline.management.commands import download_churn_data as command_module
from ml_pipeline.services import data as data_service
from ml_pipeline.services.data import (
    REQUIRED_COLUMNS,
    DatasetValidationError,
    download_churn_dataset,
    validate_required_columns,
)


class FakeResponse:
    """Small requests.Response replacement used to keep tests offline."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.content]


def make_csv_bytes() -> bytes:
    row = {column: "sample" for column in REQUIRED_COLUMNS}
    return pd.DataFrame([row]).to_csv(index=False).encode()


def test_download_command_refuses_to_overwrite_existing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "telco_churn.csv"
    dataset_path.write_text("existing data", encoding="utf-8")
    monkeypatch.setattr(command_module, "get_dataset_path", lambda: dataset_path)
    monkeypatch.setattr(
        data_service.requests,
        "get",
        lambda *args, **kwargs: pytest.fail("Existing data must prevent a request"),
    )

    with pytest.raises(CommandError, match="--overwrite"):
        call_command("download_churn_data")

    assert dataset_path.read_text(encoding="utf-8") == "existing data"


def test_download_dataset_saves_a_valid_csv_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_content = make_csv_bytes()
    expected_sha256 = hashlib.sha256(csv_content).hexdigest()
    monkeypatch.setattr(
        data_service.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(csv_content),
    )
    destination = tmp_path / "telco_churn.csv"

    summary = download_churn_dataset(
        destination,
        expected_sha256=expected_sha256,
    )

    assert summary.rows == 1
    assert summary.columns == len(REQUIRED_COLUMNS)
    assert destination.read_bytes() == csv_content
    assert list(tmp_path.glob("*.tmp")) == []


def test_required_column_validation_lists_missing_columns() -> None:
    dataframe = pd.DataFrame(columns=["customerID", "Churn"])

    with pytest.raises(DatasetValidationError, match="TotalCharges"):
        validate_required_columns(dataframe)
