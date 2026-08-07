#!/usr/bin/env python3
"""Download neurological MRI dataset from Kaggle or Google Drive."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuro_mri_xai.config import load_config
from neuro_mri_xai.utils.paths import detect_runtime_env, ensure_dir, get_project_root


def download_kaggle(dataset_slug: str, dest: Path) -> Path:
    ensure_dir(dest)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest), "--unzip"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    data_dir = dest / "data"
    if data_dir.exists() and any(data_dir.iterdir()):
        return data_dir
    for item in dest.iterdir():
        if item.is_dir() and item.name != "__MACOSX":
            inner = item / "data"
            return inner if inner.is_dir() else item
    archive = dest / "dataset.zip"
    if archive.exists():
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)
    return data_dir if (dest / "data").exists() else dest


def setup_gdrive(gdrive_path: str, dest: Path) -> Path:
    source = Path(gdrive_path)
    if not source.exists():
        raise FileNotFoundError(f"Google Drive path not found: {gdrive_path}")
    ensure_dir(dest)
    if dest.exists() and any(dest.iterdir()):
        return dest
    try:
        dest.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, dest, dirs_exist_ok=True)
    inner = dest / "data"
    return inner if inner.is_dir() else dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MRI dataset")
    parser.add_argument("--source", choices=["kaggle", "gdrive"], default=None)
    parser.add_argument("--config", default=str(get_project_root() / "configs" / "default.yaml"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    source = args.source or config.dataset.source
    runtime = detect_runtime_env()
    dest = Path(args.output) if args.output else (Path("/content/data") if runtime.value == "colab" else get_project_root() / "data")
    ensure_dir(dest)
    data_path = download_kaggle(config.dataset.kaggle_dataset, dest) if source == "kaggle" else setup_gdrive(config.dataset.gdrive_path, dest)
    print(f"Dataset ready at: {data_path}")
    print(f"Classes: {sorted(d.name for d in data_path.iterdir() if d.is_dir())}")


if __name__ == "__main__":
    main()
