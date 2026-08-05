from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError, CommandParser

from ml_pipeline.services.promotion import (
    PromotionError,
    promote_model_candidate,
)


class Command(BaseCommand):
    help = "Validate and export a churn model candidate that passes promotion policy."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--candidate-model",
            type=Path,
            required=True,
            help="Path to the trusted candidate model.pkl file.",
        )
        parser.add_argument(
            "--candidate-metadata",
            type=Path,
            required=True,
            help="Path to the candidate model_metadata.json file.",
        )
        parser.add_argument(
            "--current-metadata",
            type=Path,
            help="Path to metadata for the currently promoted model.",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="New directory that will receive the approved artifact pair.",
        )
        parser.add_argument(
            "--allow-missing-current",
            action="store_true",
            help="Explicitly permit a first-model bootstrap without current metadata.",
        )

    def handle(self, *args: object, **options: object) -> None:
        try:
            assessment = promote_model_candidate(
                candidate_model_path=options["candidate_model"],
                candidate_metadata_path=options["candidate_metadata"],
                current_metadata_path=options["current_metadata"],
                output_dir=options["output_dir"],
                allow_missing_current=bool(options["allow_missing_current"]),
            )
        except (OSError, PromotionError) as error:
            raise CommandError(str(error)) from error

        if assessment.bootstrap:
            comparison = "bootstrap promotion (no current model)"
        else:
            comparison = (
                f"current ROC-AUC={assessment.current_roc_auc:.4f}, "
                f"regression={assessment.regression:.4f}"
            )

        self.stdout.write(
            self.style.SUCCESS(
                "Model candidate approved: "
                f"ROC-AUC={assessment.candidate_roc_auc:.4f}; {comparison}."
            )
        )
        self.stdout.write(
            "Policy: "
            f"ROC-AUC > {assessment.policy.minimum_roc_auc:.3f}; "
            "maximum regression "
            f"{assessment.policy.max_roc_auc_regression:.3f}."
        )
        self.stdout.write(f"Approved artifacts: {options['output_dir']}")
