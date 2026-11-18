"""Regression tests for Phase 3.5 Kaggle dataset archives."""

from __future__ import annotations

import json
import zipfile
from hashlib import sha256

import pytest

from src.data.constants import INTENT_LABELS, SPLITS
from src.data.kaggle_package import main, package_frozen_splits
from src.data.leakage import LeakageError
from src.data.phase35 import (
    TRAIN_VARIANT_FILES,
    Phase35Config,
    load_frozen_manifest,
    run_phase35,
)


@pytest.fixture
def frozen_split(tmp_path):
    directory = tmp_path / "v1_leakfree"
    directory.mkdir()
    artifacts = {
        name: name.encode()
        for name in (*(f"{split}.parquet" for split in SPLITS), *TRAIN_VARIANT_FILES)
    }
    manifest = {
        "version": directory.name,
        "frozen": True,
        "leakage_audit": {"passed": True},
        "artifact_hashes_sha256": {
            name: sha256(data).hexdigest() for name, data in artifacts.items()
        },
    }
    contents = artifacts | {"manifest.json": json.dumps(manifest, indent=2).encode()}
    for name, data in contents.items():
        (directory / name).write_bytes(data)
    return directory, contents


def test_repackage_keeps_manifest_and_all_artifacts_byte_identical(
    frozen_split, tmp_path
):
    directory, contents = frozen_split
    output = tmp_path / "data.zip"
    assert main(["--splits-dir", str(directory), "--output", str(output)]) == 0

    extracted = tmp_path / "attached"
    with zipfile.ZipFile(output) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == contents
        archive.extractall(extracted)
    assert load_frozen_manifest(extracted)["version"] == "v1_leakfree"
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == contents


def test_default_archive_path_preserves_original_package(frozen_split):
    directory, contents = frozen_split
    original = directory / "kaggle_package_v1_leakfree.zip"
    original.write_bytes(b"historical archive")

    assert main(["--splits-dir", str(directory)]) == 0

    assert original.read_bytes() == b"historical archive"
    with zipfile.ZipFile(directory / "kaggle_upload_v1_leakfree.zip") as archive:
        assert archive.read("manifest.json") == contents["manifest.json"]


@pytest.mark.parametrize(
    ("operation", "exception", "message"),
    [
        ("remove", FileNotFoundError, "train_balanced.parquet"),
        ("corrupt", LeakageError, "changed since the manifest was frozen"),
        ("unfreeze", LeakageError, "is not frozen"),
    ],
)
def test_repackage_rejects_invalid_bundles(
    frozen_split, tmp_path, operation, exception, message
):
    directory, contents = frozen_split
    output = tmp_path / "existing.zip"
    output.write_bytes(b"prior output")
    if operation == "remove":
        (directory / "train_balanced.parquet").unlink()
    elif operation == "corrupt":
        (directory / "test.parquet").write_bytes(b"changed")
    else:
        manifest = json.loads(contents["manifest.json"])
        manifest["frozen"] = False
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(exception, match=message):
        package_frozen_splits(directory, output)
    assert output.read_bytes() == b"prior output"


def test_new_phase35_archive_contains_a_valid_frozen_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for split in SPLITS:
        rows = [
            {
                "text": f"{split} word{index}{label}",
                "label": label,
                "intent_label": label,
                "binary_label": int(label > 0),
                "intent": intent,
                "technique": "none",
                "severity": 1,
                "source": "fixture",
            }
            for label, intent in INTENT_LABELS.items()
            for index in range(5)
        ]
        (raw_dir / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )

    config = Phase35Config(data_dir=tmp_path, regenerate_train_variants=False)
    canonical = run_phase35(config)
    extracted = tmp_path / "attached"
    with zipfile.ZipFile(canonical["kaggle_package"]["zip_path"]) as archive:
        archive.extractall(extracted)
    packaged = load_frozen_manifest(extracted)
    assert packaged["artifact_hashes_sha256"] == canonical["artifact_hashes_sha256"]
    assert packaged["kaggle_package"] == {}
