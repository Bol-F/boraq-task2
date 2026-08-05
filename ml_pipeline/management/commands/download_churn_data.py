from __future__ import annotations

import pandas as pd
import requests
from django.core.management.base import BaseCommand, CommandError, CommandParser

from ml_pipeline.services.data import (
    DatasetValidationError,
    download_churn_dataset,
    get_dataset_path,
)


class Command(BaseCommand):
    help = "Download and validate the public IBM Telco Customer Churn dataset."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Explicitly replace an existing local dataset.",
        )

    def handle(self, *args: object, **options: object) -> None:
        destination = get_dataset_path()
        overwrite = bool(options["overwrite"])

        try:
            summary = download_churn_dataset(destination, overwrite=overwrite)
        except FileExistsError as error:
            raise CommandError(str(error)) from error
        except (
            DatasetValidationError,
            OSError,
            pd.errors.ParserError,
            requests.RequestException,
        ) as error:
            raise CommandError(f"Dataset download failed: {error}") from error

        self.stdout.write(
            self.style.SUCCESS(
                f"Downloaded and validated {summary.rows} rows and "
                f"{summary.columns} columns at {summary.path}."
            )
        )
