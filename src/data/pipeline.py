"""End-to-end data preparation pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from src.data.cleaning import CleaningConfig, clean_text
from src.data.dedup import deduplicate_records
from src.data.ingest import SnapshotMetadata, ingest_dataset
from src.data.reporting import write_distribution_reports
from src.data.validation import filter_valid_records

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    seed: int = 42
    dataset_name: str = "neuralchemy/prompt-injection-Threat-Matrix"
    dataset_config: str = "multiclass"
    data_dir: Path = Path("data")
    lowercase: bool = False
    near_duplicate_threshold: float = 0.85
    drop_invalid_rows: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)

        data_cfg = payload.get("data", {})
        dedup_cfg = payload.get("dedup", {})
        return cls(
            seed=int(payload.get("seed", 42)),
            dataset_name=data_cfg.get("dataset", cls.dataset_name),
            dataset_config=data_cfg.get("config", cls.dataset_config),
            data_dir=Path(data_cfg.get("data_dir", "data")),
            lowercase=bool(data_cfg.get("lowercase", False)),
            near_duplicate_threshold=float(
                dedup_cfg.get("near_duplicate_threshold", 0.85)
            ),
            drop_invalid_rows=bool(data_cfg.get("drop_invalid_rows", True)),
        )


def _normalize_fields(record: dict) -> dict:
    """Align upstream schema aliases to canonical field names."""
    normalized = dict(record)
    if "intent_label" not in normalized and "label" in normalized:
        normalized["intent_label"] = normalized["label"]
    if "label" not in normalized and "intent_label" in normalized:
        normalized["label"] = normalized["intent_label"]
    return normalized


def _clean_record(record: dict, cleaning: CleaningConfig) -> dict:
    normalized = _normalize_fields(record)
    cleaned = dict(normalized)
    cleaned["text_original"] = normalized.get("text", "")
    cleaned["text_clean"] = clean_text(normalized.get("text", ""), cleaning)
    cleaned["text"] = cleaned["text_clean"]
    return cleaned


def process_split(
    records: list[dict],
    config: PipelineConfig,
) -> tuple[list[dict], dict]:
    """Clean, deduplicate, and validate one dataset split."""
    cleaning = CleaningConfig(
        lowercase=config.lowercase,
        normalize_unicode=True,
        remove_control_chars=True,
        remove_zero_width=True,
        collapse_whitespace=True,
        strip_text=True,
    )

    cleaned_records = [_clean_record(row, cleaning) for row in records]
    deduped_records, dedup_report = deduplicate_records(
        cleaned_records,
        near_duplicate_threshold=config.near_duplicate_threshold,
    )

    if config.drop_invalid_rows:
        valid_records, validation_report = filter_valid_records(deduped_records)
    else:
        from src.data.validation import validate_records

        valid_records = deduped_records
        validation_report = validate_records(deduped_records)

    split_report = {
        "input_rows": len(records),
        "after_cleaning": len(cleaned_records),
        "after_deduplication": dedup_report.output_count,
        "exact_duplicates_removed": dedup_report.exact_duplicates_removed,
        "near_duplicates_removed": dedup_report.near_duplicates_removed,
        "validation": validation_report.to_dict(),
        "output_rows": len(valid_records),
    }
    return valid_records, split_report


def run_prepare_pipeline(config: PipelineConfig) -> dict:
    """Execute full Phase-1 data engineering pipeline."""
    processed_root = config.data_dir / "processed"
    reports_root = processed_root / "reports"
    processed_root.mkdir(parents=True, exist_ok=True)

    logger.info("Ingesting dataset: %s (%s)", config.dataset_name, config.dataset_config)
    split_records, snapshot = ingest_dataset(
        output_dir=config.data_dir,
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        seed=config.seed,
    )

    split_frames: dict[str, pd.DataFrame] = {}
    split_reports: dict[str, dict] = {}

    for split_name, records in split_records.items():
        logger.info("Processing split: %s (%d rows)", split_name, len(records))
        processed_records, split_report = process_split(records, config)
        split_reports[split_name] = split_report
        frame = pd.DataFrame(processed_records)
        split_frames[split_name] = frame

        parquet_path = processed_root / f"{split_name}.parquet"
        frame.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s", parquet_path)

    distribution = write_distribution_reports(split_frames, reports_root)

    pipeline_report = {
        "snapshot": snapshot.to_dict(),
        "config": {
            "seed": config.seed,
            "dataset_name": config.dataset_name,
            "dataset_config": config.dataset_config,
            "lowercase": config.lowercase,
            "near_duplicate_threshold": config.near_duplicate_threshold,
            "drop_invalid_rows": config.drop_invalid_rows,
        },
        "splits": split_reports,
        "distribution_summary": {
            split: data["row_count"] for split, data in distribution.items()
        },
    }

    metadata_path = processed_root / "metadata.json"
    metadata_path.write_text(json.dumps(pipeline_report, indent=2), encoding="utf-8")

    validation_path = reports_root / "validation_report.json"
    validation_path.write_text(
        json.dumps({k: v["validation"] for k, v in split_reports.items()}, indent=2),
        encoding="utf-8",
    )

    dedup_path = reports_root / "dedup_report.json"
    dedup_path.write_text(
        json.dumps(
            {
                k: {
                    "exact_duplicates_removed": v["exact_duplicates_removed"],
                    "near_duplicates_removed": v["near_duplicates_removed"],
                    "output_rows": v["output_rows"],
                }
                for k, v in split_reports.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Pipeline complete. Snapshot ID: %s", snapshot.snapshot_id)
    return pipeline_report
