"""Kaggle-ready processed dataset packaging."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data.constants import DATASET_LICENSE, DATASET_NAME, DATASET_URL
from src.data.reproducibility import sha256_file


KAGGLE_FILES = (
    "train.parquet",
    "validation.parquet",
    "test.parquet",
    "train_augmented.parquet",
    "metadata.json",
)


def export_technique_subsets(
    split_frames: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    """Export technique-specific subsets for encoding, role_play, tool_abuse."""
    from src.data.technique_subsets import TECHNIQUE_SUBSETS

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}

    for subset_name, spec in TECHNIQUE_SUBSETS.items():
        summary[subset_name] = {}
        keywords = tuple(k.lower() for k in spec["technique_keywords"])
        intent_labels = set(spec["intent_labels"])

        for split, frame in split_frames.items():
            if split not in ("train", "validation", "test"):
                continue

            technique_col = frame["technique"].astype(str).str.lower()
            keyword_match = technique_col.apply(
                lambda value: any(k in value for k in keywords)
            )
            intent_match = frame["intent_label"].astype(int).isin(intent_labels)
            subset = frame[keyword_match | intent_match]
            summary[subset_name][split] = int(len(subset))

            if len(subset) > 0:
                subset.to_parquet(
                    output_dir / f"technique_{subset_name}_{split}.parquet",
                    index=False,
                )

    report_path = output_dir.parent / "reports" / "technique_subsets.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


PUBLISH_CAVEAT = (
    "Packaging this folder does NOT grant permission to publish it. The upstream dataset is "
    "declared CC BY-NC 4.0 by local metadata only; the upstream page could not be re-verified. "
    "Use as a private Kaggle dataset for your own experiments unless you have confirmed the licence."
)


def package_kaggle_dataset(
    processed_dir: Path,
    output_dir: Path | None = None,
    files: tuple[str, ...] = KAGGLE_FILES,
    zip_path: Path | None = None,
    title: str = "Prompt Injection Threat Matrix (Processed)",
    slug: str = "prompt-injection-threat-matrix-processed",
    extra_readme: str = "",
) -> dict:
    """
    Build Kaggle-ready dataset package with parquet files and metadata.

    Default output (legacy pipeline):
      data/processed/kaggle_package/
      data/processed/kaggle_package.zip
    Phase 3.5 passes ``files``/``zip_path`` to build ``kaggle_package_v1_leakfree.zip``.
    No upload is performed by this code.
    """
    processed_dir = Path(processed_dir)
    package_dir = output_dir or (processed_dir / "kaggle_package")
    package_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for filename in files:
        src = processed_dir / filename
        if src.exists():
            dst = package_dir / Path(filename).name
            shutil.copy2(src, dst)
            copied.append(Path(filename).name)

    metadata_path = processed_dir / "metadata.json"
    manifest_path_v1 = processed_dir / "manifest.json"
    snapshot_id = "unknown"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        snapshot_id = metadata.get("snapshot", {}).get("snapshot_id", "unknown")
    elif manifest_path_v1.exists():
        manifest_v1 = json.loads(manifest_path_v1.read_text(encoding="utf-8"))
        snapshot_id = manifest_v1.get("source_snapshot_id", "unknown")

    dataset_card = {
        "title": title,
        "id": slug,
        "licenses": [{"name": DATASET_LICENSE}],
        "keywords": ["prompt-injection", "llm-security", "nlp", "classification"],
        "source": DATASET_URL,
        "upstream_dataset": DATASET_NAME,
        "snapshot_id": snapshot_id,
        "files": copied,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "publish_caveat": PUBLISH_CAVEAT,
    }

    (package_dir / "dataset-metadata.json").write_text(
        json.dumps(dataset_card, indent=2),
        encoding="utf-8",
    )

    readme = f"""# {title} — Kaggle Package

Upstream (per local metadata): {DATASET_NAME}
Snapshot ID: `{snapshot_id}`
Declared license: {DATASET_LICENSE}

> {PUBLISH_CAVEAT}

## Files

{chr(10).join(f"- `{name}`" for name in copied)}

## Usage

```python
import pandas as pd
train = pd.read_parquet("train.parquet")
```
{extra_readme}
Generated for capstone thesis research (DIU).
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")

    zip_path = Path(zip_path) if zip_path is not None else processed_dir / "kaggle_package.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(package_dir).as_posix())

    file_hashes = {
        str(p.relative_to(processed_dir).as_posix()): sha256_file(p)
        for p in package_dir.rglob("*")
        if p.is_file()
    }
    file_hashes[zip_path.name] = sha256_file(zip_path)

    manifest = {
        "package_dir": str(package_dir.as_posix()),
        "zip_path": str(zip_path.as_posix()),
        "files": copied + ["dataset-metadata.json", "README.md"],
        "file_hashes_sha256": file_hashes,
        "publish_caveat": PUBLISH_CAVEAT,
    }

    manifest_path = processed_dir / "reports" / "kaggle_package_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
