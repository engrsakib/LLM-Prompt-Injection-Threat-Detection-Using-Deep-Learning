# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Dataset download helpers: kagglehub primary with legacy fallbacks."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

from neuro_mri_xai.config import Config
from neuro_mri_xai.utils.paths import (
    RuntimeEnv,
    detect_runtime_env,
    ensure_dir,
    find_kagglehub_cache,
    get_project_root,
    resolve_imagefolder_root,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def download_kagglehub(handle: str) -> Path:
    """Download dataset via kagglehub and return ImageFolder-ready root."""
    import kagglehub

    raw_path = Path(kagglehub.dataset_download(handle))
    resolved = resolve_imagefolder_root(raw_path)
    if resolved is None:
        raise FileNotFoundError(
            f"kagglehub downloaded to {raw_path}, but no ImageFolder layout was found.",
        )
    return resolved.resolve()


def download_kaggle(dataset_slug: str, dest: Path) -> Path:
    """Download dataset using the Kaggle CLI API."""
    ensure_dir(dest)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(dest),
        "--unzip",
    ]
    logger.info("Running Kaggle CLI: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    candidate = dest / "data"
    if candidate.is_dir() and any(candidate.iterdir()):
        resolved = resolve_imagefolder_root(candidate)
        return resolved if resolved else candidate.resolve()

    for item in dest.iterdir():
        if item.is_dir() and item.name != "__MACOSX":
            inner = item / "data"
            target = inner if inner.is_dir() else item
            resolved = resolve_imagefolder_root(target)
            return resolved if resolved else target.resolve()

    archive = dest / "dataset.zip"
    if archive.exists():
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest)

    fallback = dest / "data" if (dest / "data").is_dir() else dest
    resolved = resolve_imagefolder_root(fallback)
    return resolved if resolved else fallback.resolve()


def setup_gdrive(gdrive_path: str, dest: Path) -> Path:
    """Link or copy Google Drive dataset into dest."""
    source = Path(gdrive_path)
    if not source.exists():
        raise FileNotFoundError(f"Google Drive path not found: {gdrive_path}")

    ensure_dir(dest)
    if dest.exists() and any(dest.iterdir()):
        resolved = resolve_imagefolder_root(dest)
        return resolved if resolved else dest.resolve()

    try:
        dest.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, dest, dirs_exist_ok=True)

    target = dest / "data" if (dest / "data").is_dir() else dest
    resolved = resolve_imagefolder_root(target)
    return resolved if resolved else target.resolve()


def default_download_dest() -> Path:
    project_root = get_project_root()
    runtime = detect_runtime_env()
    if runtime == RuntimeEnv.COLAB:
        return project_root / "data"
    return project_root / "data"


def _fallback_source(configured_source: str) -> str:
    return "kaggle" if configured_source == "kagglehub" else configured_source


def download_dataset(
    config: Config,
    *,
    source: str | None = None,
    output: Path | None = None,
    use_kagglehub: bool = False,
) -> Path:
    """Download or resolve dataset path, preferring kagglehub when requested."""
    configured_source = source or config.dataset.source
    dest = Path(output) if output else default_download_dest()
    ensure_dir(dest)

    should_try_kagglehub = use_kagglehub or configured_source == "kagglehub"
    if should_try_kagglehub and config.dataset.kagglehub_handle:
        try:
            cached = find_kagglehub_cache(config.dataset.kagglehub_handle)
            if cached is not None:
                logger.info("Using cached kagglehub dataset at %s", cached)
                return cached

            path = download_kagglehub(config.dataset.kagglehub_handle)
            logger.info("Downloaded kagglehub dataset to %s", path)
            return path
        except ImportError:
            warnings.warn(
                "kagglehub is not installed; falling back to configured dataset source.",
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(
                f"kagglehub download failed ({exc}); falling back to configured dataset source.",
                stacklevel=2,
            )

    fallback = _fallback_source(configured_source)
    if fallback == "kaggle":
        if not config.dataset.kaggle_dataset:
            raise ValueError("dataset.kaggle_dataset must be set for Kaggle CLI fallback.")
        path = download_kaggle(config.dataset.kaggle_dataset, dest)
        logger.info("Downloaded Kaggle dataset to %s", path)
        return path
    if fallback == "gdrive":
        path = setup_gdrive(config.dataset.gdrive_path, dest)
        logger.info("Linked Google Drive dataset at %s", path)
        return path

    resolved = resolve_imagefolder_root(dest)
    if resolved is not None:
        return resolved

    raise ValueError(
        f"Unsupported dataset source '{configured_source}'. Use kagglehub, kaggle, or gdrive.",
    )
