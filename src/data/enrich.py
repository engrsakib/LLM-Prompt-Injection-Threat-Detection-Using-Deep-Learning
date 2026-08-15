"""Phase-2 dataset enrichment: balancing, augmentation, analysis, MLflow."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from src.data.analysis import add_derived_columns, write_phase2_reports
from src.data.augment import AugmentConfig, augment_training_frame
from src.data.balancing import balance_minority_classes
from src.data.mlflow_tracking import log_dataset_version
from src.data.obfuscation import obfuscation_features
from src.data.reporting import write_distribution_reports

logger = logging.getLogger(__name__)


@dataclass
class Phase2Config:
    seed: int = 42
    data_dir: Path = Path("data")
    enabled: bool = True
    balancing_strategy: str = "oversample_median"
    augment_enabled: bool = True
    augment_minority_only: bool = True
    synonym_replace_prob: float = 0.25
    token_noise_prob: float = 0.10
    max_augmentations_per_row: int = 1
    mlflow_enabled: bool = True
    mlflow_experiment: str = "data-engineering"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase2Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        phase2 = payload.get("phase2", {})
        aug = phase2.get("augmentation", {})
        mlflow_cfg = phase2.get("mlflow", {})
        data_cfg = payload.get("data", {})

        return cls(
            seed=int(payload.get("seed", 42)),
            data_dir=Path(data_cfg.get("data_dir", "data")),
            enabled=bool(phase2.get("enabled", True)),
            balancing_strategy=phase2.get("balancing_strategy", "oversample_median"),
            augment_enabled=bool(aug.get("enabled", True)),
            augment_minority_only=bool(aug.get("minority_only", True)),
            synonym_replace_prob=float(aug.get("synonym_replace_prob", 0.25)),
            token_noise_prob=float(aug.get("token_noise_prob", 0.10)),
            max_augmentations_per_row=int(aug.get("max_augmentations_per_row", 1)),
            mlflow_enabled=bool(mlflow_cfg.get("enabled", True)),
            mlflow_experiment=mlflow_cfg.get("experiment_name", "data-engineering"),
        )


def _load_processed_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")
    return pd.read_parquet(path)


def _apply_obfuscation_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        features = obfuscation_features(
            row.get("text_clean", row.get("text", "")),
            intent_label=int(row.get("intent_label", row.get("label", 0))),
        )
        enriched = row.to_dict()
        enriched.update(features)
        rows.append(enriched)
    return pd.DataFrame(rows)


def run_phase2_enrichment(config: Phase2Config) -> dict:
    """Run Phase-2 enrichment on processed dataset artifacts."""
    processed_dir = config.data_dir / "processed"
    reports_dir = processed_dir / "reports"
    subsets_dir = processed_dir / "subsets"

    split_frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        split_frames[split] = add_derived_columns(_load_processed_split(processed_dir, split))

    logger.info("Applying obfuscation-aware normalization columns")
    for split in split_frames:
        split_frames[split] = _apply_obfuscation_columns(split_frames[split])
        split_frames[split]["text"] = split_frames[split]["text_obfuscation_aware"]
        out = processed_dir / f"{split}.parquet"
        split_frames[split].to_parquet(out, index=False)

    phase2_reports = write_phase2_reports(split_frames, reports_dir, subsets_dir)

    balancing_report: dict = {}
    augmentation_report: dict = {}

    train_df = split_frames["train"]
    if config.enabled:
        logger.info("Balancing minority classes in train split")
        balanced_train, balancing_report = balance_minority_classes(
            train_df,
            strategy=config.balancing_strategy,
            seed=config.seed,
        )
        balanced_path = processed_dir / "train_balanced.parquet"
        balanced_train.to_parquet(balanced_path, index=False)

        augment_source = balanced_train
        if config.augment_enabled:
            logger.info("Running train-only augmentation")
            aug_config = AugmentConfig(
                enabled=True,
                synonym_replace_prob=config.synonym_replace_prob,
                token_noise_prob=config.token_noise_prob,
                max_augmentations_per_row=config.max_augmentations_per_row,
                augment_minority_only=config.augment_minority_only,
            )
            augmented_train, augmentation_report = augment_training_frame(
                augment_source,
                config=aug_config,
                seed=config.seed,
            )
            augmented_path = processed_dir / "train_augmented.parquet"
            augmented_train.to_parquet(augmented_path, index=False)
            split_frames["train_augmented"] = augmented_train
        else:
            split_frames["train_balanced"] = balanced_train

    write_distribution_reports(split_frames, reports_dir)

    metadata_path = processed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata["phase2"] = {
        "enabled": config.enabled,
        "balancing_strategy": config.balancing_strategy,
        "balancing": balancing_report,
        "augmentation": augmentation_report,
        "reports": {
            "severity_buckets": "reports/severity_buckets.json",
            "ambiguity_subsets": "reports/ambiguity_subsets.json",
            "subsets_dir": "subsets/",
        },
        "artifacts": {
            "train_balanced": "train_balanced.parquet",
            "train_augmented": "train_augmented.parquet",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    mlflow_run_id = None
    if config.mlflow_enabled:
        mlflow_run_id = log_dataset_version(
            processed_dir=processed_dir,
            experiment_name=config.mlflow_experiment,
            extra_params={"phase": "phase2"},
        )

    return {
        "phase2_reports": phase2_reports,
        "balancing": balancing_report,
        "augmentation": augmentation_report,
        "mlflow_run_id": mlflow_run_id,
    }


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-2 dataset enrichment.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    config = Phase2Config.from_yaml(args.config)
    result = run_phase2_enrichment(config)
    print("Phase-2 enrichment completed.")
    if result.get("mlflow_run_id"):
        print(f"MLflow run ID: {result['mlflow_run_id']}")
    if result.get("augmentation"):
        print(f"Augmented train rows: {result['augmentation'].get('output_rows', 'n/a')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
