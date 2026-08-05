from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError, CommandParser

from ml_pipeline.services.data import get_dataset_path
from ml_pipeline.services.metadata import save_training_artifacts
from ml_pipeline.services.preprocessing import load_features_and_target
from ml_pipeline.services.tracking import EXPERIMENT_NAME, track_training_results
from ml_pipeline.services.training import TrainingSummary, train_and_compare


def format_comparison_table(summary: TrainingSummary) -> str:
    """Return a readable fixed-width table of held-out model metrics."""
    headers = ("Model", "ROC-AUC", "PR-AUC", "F1")
    rows = [
        (
            result.model_name,
            f"{result.metrics.roc_auc:.4f}",
            f"{result.metrics.pr_auc:.4f}",
            f"{result.metrics.f1:.4f}",
        )
        for result in summary.results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join(
        [
            format_row(headers),
            separator,
            *(format_row(row) for row in rows),
        ]
    )


class Command(BaseCommand):
    help = "Train, compare, and track Telecom Customer Churn models."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--model-path",
            type=Path,
            help="Optional output path for the winning model pipeline.",
        )
        parser.add_argument(
            "--metadata-path",
            type=Path,
            help="Optional output path for the model metadata JSON.",
        )

    def handle(self, *args: object, **options: object) -> None:
        dataset_path = get_dataset_path()
        if not dataset_path.is_file():
            raise CommandError(
                f"Dataset not found at {dataset_path}. Run "
                "'uv run python manage.py download_churn_data' first."
            )

        try:
            features, target = load_features_and_target(dataset_path)
            summary = train_and_compare(features, target)
            tracking = track_training_results(summary)
            artifacts = save_training_artifacts(
                summary,
                tracking,
                dataset_rows=len(features),
                feature_count=features.shape[1],
                model_path=options["model_path"],
                metadata_path=options["metadata_path"],
            )
        except Exception as error:
            raise CommandError(f"Model training failed: {error}") from error

        self.stdout.write("\nModel comparison (held-out test set):")
        self.stdout.write(format_comparison_table(summary))
        self.stdout.write(
            self.style.SUCCESS(
                f"\nWinning model: {summary.winner.model_name} "
                f"(ROC-AUC: {summary.winner.metrics.roc_auc:.4f})"
            )
        )
        self.stdout.write(
            f"MLflow experiment: {EXPERIMENT_NAME} (ID: {tracking.experiment_id})"
        )
        for model_name, run_id in tracking.run_ids.items():
            self.stdout.write(f"  {model_name}: {run_id}")
        self.stdout.write(f"Saved model: {artifacts.model_path}")
        self.stdout.write(f"Saved metadata: {artifacts.metadata_path}")
