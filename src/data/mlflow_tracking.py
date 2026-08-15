"""MLflow helpers for dataset versioning and experiment reproducibility."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def log_dataset_version(
    processed_dir: Path | str = "data/processed",
    run_name: str | None = None,
    experiment_name: str = "data-engineering",
    extra_params: dict[str, Any] | None = None,
) -> str | None:
    """
    Log dataset snapshot metadata to MLflow.

    Returns run_id if logging succeeds, else None.
    """
    processed_dir = Path(processed_dir)
    metadata = _load_metadata(processed_dir)
    if not metadata:
        return None

    snapshot = metadata.get("snapshot", {})
    snapshot_id = snapshot.get("snapshot_id", "unknown")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name or f"dataset-{snapshot_id}") as run:
        mlflow.log_param("snapshot_id", snapshot_id)
        mlflow.log_param("dataset_name", snapshot.get("dataset_name", ""))
        mlflow.log_param("dataset_config", snapshot.get("dataset_config", ""))
        mlflow.log_param("dataset_license", snapshot.get("dataset_license", ""))
        mlflow.log_param("seed", snapshot.get("seed", ""))

        for split, count in snapshot.get("splits", {}).items():
            mlflow.log_metric(f"rows_raw_{split}", int(count))

        for split, count in metadata.get("distribution_summary", {}).items():
            mlflow.log_metric(f"rows_processed_{split}", int(count))

        phase2 = metadata.get("phase2", {})
        if phase2:
            mlflow.log_param("phase2_balancing_strategy", phase2.get("balancing_strategy", ""))
            mlflow.log_metric(
                "rows_train_augmented",
                int(phase2.get("augmentation", {}).get("output_rows", 0)),
            )

        if extra_params:
            mlflow.log_params({k: str(v) for k, v in extra_params.items()})

        metadata_file = processed_dir / "metadata.json"
        if metadata_file.exists():
            mlflow.log_artifact(str(metadata_file), artifact_path="dataset")

        reports_dir = processed_dir / "reports"
        if reports_dir.exists():
            for report in reports_dir.glob("*.json"):
                mlflow.log_artifact(str(report), artifact_path="dataset/reports")

        return run.info.run_id


def get_dataset_version_params(processed_dir: Path | str = "data/processed") -> dict[str, Any]:
    """Load dataset version fields for training experiment logging."""
    metadata = _load_metadata(Path(processed_dir))
    snapshot = metadata.get("snapshot", {})
    return {
        "dataset_snapshot_id": snapshot.get("snapshot_id"),
        "dataset_name": snapshot.get("dataset_name"),
        "dataset_config": snapshot.get("dataset_config"),
        "dataset_created_at_utc": snapshot.get("created_at_utc"),
        "dataset_train_rows": metadata.get("distribution_summary", {}).get("train"),
        "dataset_train_augmented_rows": metadata.get("phase2", {})
        .get("augmentation", {})
        .get("output_rows"),
    }
