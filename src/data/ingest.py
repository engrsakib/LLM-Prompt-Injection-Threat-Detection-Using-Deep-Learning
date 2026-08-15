"""Dataset ingestion with snapshot metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from datasets import load_dataset

from src.data.constants import DATASET_LICENSE, DATASET_NAME, DATASET_URL, SPLITS


@dataclass(frozen=True)
class SnapshotMetadata:
    snapshot_id: str
    dataset_name: str
    dataset_config: str
    dataset_url: str
    dataset_license: str
    seed: int
    created_at_utc: str
    splits: dict[str, int]
    schema_fields: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _short_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_snapshot_id(
    dataset_name: str,
    dataset_config: str,
    seed: int,
    split_counts: dict[str, int],
) -> str:
    """Deterministic snapshot identifier from dataset identity and counts."""
    payload = json.dumps(
        {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "seed": seed,
            "split_counts": split_counts,
        },
        sort_keys=True,
    )
    return f"tm-{_short_hash(payload)}"


def ingest_dataset(
    output_dir: Path,
    dataset_name: str = DATASET_NAME,
    dataset_config: str = "multiclass",
    seed: int = 42,
) -> tuple[dict[str, list[dict]], SnapshotMetadata]:
    """Download dataset from Hugging Face and persist raw JSONL snapshots."""
    ds = load_dataset(dataset_name, dataset_config)

    raw_root = output_dir / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    split_records: dict[str, list[dict]] = {}
    split_counts: dict[str, int] = {}

    for split in SPLITS:
        if split not in ds:
            continue
        records = [dict(row) for row in ds[split]]
        split_records[split] = records
        split_counts[split] = len(records)

        split_path = raw_root / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    schema_fields = sorted(split_records[SPLITS[0]][0].keys()) if split_records else []
    snapshot_id = build_snapshot_id(dataset_name, dataset_config, seed, split_counts)

    metadata = SnapshotMetadata(
        snapshot_id=snapshot_id,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_url=DATASET_URL,
        dataset_license=DATASET_LICENSE,
        seed=seed,
        created_at_utc=datetime.now(UTC).isoformat(),
        splits=split_counts,
        schema_fields=schema_fields,
    )

    metadata_path = raw_root / "snapshot_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding="utf-8",
    )

    return split_records, metadata
