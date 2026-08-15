"""Phase-3 finalization: technique subsets, adversarial aug, leakage, reproducibility, Kaggle."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from src.data.adversarial_paraphrase import (
    AdversarialParaphraseConfig,
    augment_adversarial_paraphrase,
)
from src.data.analysis import add_derived_columns
from src.data.kaggle_package import export_technique_subsets, package_kaggle_dataset
from src.data.leakage import run_leakage_audit, write_leakage_report
from src.data.mlflow_tracking import log_dataset_version
from src.data.reproducibility import write_reproducibility_appendix

logger = logging.getLogger(__name__)


@dataclass
class Phase3Config:
    seed: int = 42
    data_dir: Path = Path("data")
    docs_dir: Path = Path("docs")
    enabled: bool = True
    adversarial_enabled: bool = True
    adversarial_max_variants: int = 1
    leakage_near_threshold: float = 0.85
    mlflow_enabled: bool = True
    mlflow_experiment: str = "data-engineering"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase3Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        phase3 = payload.get("phase3", {})
        adv = phase3.get("adversarial_paraphrase", {})
        mlflow_cfg = phase3.get("mlflow", payload.get("phase2", {}).get("mlflow", {}))
        data_cfg = payload.get("data", {})

        return cls(
            seed=int(payload.get("seed", 42)),
            data_dir=Path(data_cfg.get("data_dir", "data")),
            docs_dir=Path(phase3.get("docs_dir", "docs")),
            enabled=bool(phase3.get("enabled", True)),
            adversarial_enabled=bool(adv.get("enabled", True)),
            adversarial_max_variants=int(adv.get("max_variants_per_row", 1)),
            leakage_near_threshold=float(phase3.get("leakage_near_threshold", 0.85)),
            mlflow_enabled=bool(mlflow_cfg.get("enabled", True)),
            mlflow_experiment=mlflow_cfg.get("experiment_name", "data-engineering"),
        )


def _load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "validation", "test"):
        path = processed_dir / f"{name}.parquet"
        if path.exists():
            splits[name] = add_derived_columns(pd.read_parquet(path))
    return splits


def run_phase3_finalization(config: Phase3Config) -> dict:
    """Execute Phase-3 IEEE-grade finalization pipeline."""
    processed_dir = config.data_dir / "processed"
    subsets_dir = processed_dir / "subsets"
    reports_dir = processed_dir / "reports"

    split_frames = _load_splits(processed_dir)
    if not split_frames:
        raise FileNotFoundError("Processed splits not found. Run Phase 1 first.")

    technique_report = export_technique_subsets(split_frames, subsets_dir)
    logger.info("Exported technique-specific subsets")

    adversarial_report: dict = {}
    train_path = processed_dir / "train_augmented.parquet"
    if not train_path.exists():
        train_path = processed_dir / "train_balanced.parquet"
    if not train_path.exists():
        train_path = processed_dir / "train.parquet"

    train_df = pd.read_parquet(train_path)
    if config.enabled and config.adversarial_enabled:
        logger.info("Applying adversarial paraphrase augmentation (intent-preserving)")
        adv_config = AdversarialParaphraseConfig(
            enabled=True,
            max_variants_per_row=config.adversarial_max_variants,
        )
        train_adv, adversarial_report = augment_adversarial_paraphrase(
            train_df,
            config=adv_config,
            seed=config.seed,
        )
        adv_path = processed_dir / "train_adversarial.parquet"
        train_adv.to_parquet(adv_path, index=False)
        split_frames["train_adversarial"] = train_adv

    leakage_report = run_leakage_audit(
        split_frames,
        near_threshold=config.leakage_near_threshold,
    )
    write_leakage_report(leakage_report, reports_dir / "leakage_report.json")

    repro_manifest = write_reproducibility_appendix(processed_dir, config.docs_dir)
    kaggle_manifest = package_kaggle_dataset(processed_dir)

    metadata_path = processed_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata["phase3"] = {
        "enabled": config.enabled,
        "technique_subsets": technique_report,
        "adversarial_paraphrase": adversarial_report,
        "leakage_audit": leakage_report,
        "reproducibility_manifest": "reports/reproducibility_manifest.json",
        "kaggle_package": kaggle_manifest,
        "artifacts": {
            "train_adversarial": "train_adversarial.parquet",
            "kaggle_package_zip": "kaggle_package.zip",
            "reproducibility_doc": "docs/REPRODUCIBILITY.md",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    mlflow_run_id = None
    if config.mlflow_enabled:
        mlflow_run_id = log_dataset_version(
            processed_dir=processed_dir,
            experiment_name=config.mlflow_experiment,
            extra_params={"phase": "phase3", "leakage_passed": leakage_report["passed"]},
        )

    return {
        "technique_subsets": technique_report,
        "adversarial_paraphrase": adversarial_report,
        "leakage_audit": leakage_report,
        "reproducibility": repro_manifest,
        "kaggle_package": kaggle_manifest,
        "mlflow_run_id": mlflow_run_id,
    }


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-3 dataset finalization.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    config = Phase3Config.from_yaml(args.config)
    result = run_phase3_finalization(config)

    print("Phase-3 finalization completed.")
    leakage = result["leakage_audit"]
    print(f"Leakage audit passed: {leakage['passed']}")
    print(f"Exact overlap pairs: {leakage['summary']['exact_leak_pairs']}")
    print(f"Near overlap pairs: {leakage['summary']['near_leak_pairs']}")
    if result.get("adversarial_paraphrase"):
        print(
            "Adversarial train rows:",
            result["adversarial_paraphrase"].get("output_rows", "n/a"),
        )
    print("Kaggle package:", result["kaggle_package"].get("zip_path"))
    if result.get("mlflow_run_id"):
        print(f"MLflow run ID: {result['mlflow_run_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
