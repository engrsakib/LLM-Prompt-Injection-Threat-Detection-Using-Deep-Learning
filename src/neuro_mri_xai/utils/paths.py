# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Environment-aware path resolution."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from neuro_mri_xai.data.constants import (
    DEFAULT_KAGGLE_DATA_SUBDIR,
    DEFAULT_KAGGLE_MOUNT_ROOT,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class RuntimeEnv(str, Enum):
    COLAB = "colab"
    KAGGLE = "kaggle"
    LOCAL = "local"


def detect_runtime_env() -> RuntimeEnv:
    if os.path.exists("/content"):
        try:
            import google.colab  # noqa: F401

            return RuntimeEnv.COLAB
        except ImportError:
            if Path("/content/neuro-mri-xai").exists():
                return RuntimeEnv.COLAB
    if os.path.exists("/kaggle/working"):
        return RuntimeEnv.KAGGLE
    return RuntimeEnv.LOCAL


def get_project_root() -> Path:
    env_root = os.environ.get("NEURO_MRI_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    runtime = detect_runtime_env()
    if runtime == RuntimeEnv.COLAB:
        candidates = [Path("/content/neuro-mri-xai"), Path.cwd()]
    elif runtime == RuntimeEnv.KAGGLE:
        candidates = [Path("/kaggle/working/neuro-mri-xai"), Path("/kaggle/working"), Path.cwd()]
    else:
        candidates = [Path.cwd()]

    for candidate in candidates:
        if (candidate / "configs" / "default.yaml").exists():
            return candidate.resolve()
        if (candidate / "src" / "neuro_mri_xai").exists():
            return candidate.resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "default.yaml").exists():
            return parent
        if parent.name == "neuro_mri_xai" and (parent.parent / "configs").exists():
            return parent.parent

    return Path.cwd().resolve()


def _has_imagefolder_layout(path: Path) -> bool:
    if not path.is_dir():
        return False
    class_dirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not class_dirs:
        return False
    return any(
        any(class_dir.glob(f"*{ext}")) for class_dir in class_dirs for ext in IMAGE_EXTENSIONS
    )


def resolve_imagefolder_root(candidate: Path) -> Path | None:
    """Locate an ImageFolder-style class directory tree under candidate."""
    candidate = Path(candidate).resolve()
    if _has_imagefolder_layout(candidate):
        return candidate

    for sub_name in ("data", "images", "train", "dataset"):
        sub_path = candidate / sub_name
        if _has_imagefolder_layout(sub_path):
            return sub_path.resolve()

    if candidate.is_dir():
        for child in candidate.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            if _has_imagefolder_layout(child):
                return child.resolve()
            nested = child / "data"
            if _has_imagefolder_layout(nested):
                return nested.resolve()

    return None


def resolve_dataset_root(candidate: Path) -> Path | None:
    """Resolve ImageFolder root: prefer ``candidate/data/``, else ``candidate`` itself."""
    candidate = Path(candidate).expanduser().resolve()
    if not candidate.is_dir():
        return None

    data_sub = candidate / DEFAULT_KAGGLE_DATA_SUBDIR
    if data_sub.is_dir():
        resolved = resolve_imagefolder_root(data_sub)
        if resolved is not None:
            return resolved

    return resolve_imagefolder_root(candidate)


def resolve_kaggle_dataset_root(mount_root: Path | str | None = None) -> Path | None:
    """Resolve the canonical Kaggle dataset mount to an ImageFolder root."""
    root = Path(mount_root) if mount_root is not None else Path(DEFAULT_KAGGLE_MOUNT_ROOT)
    return resolve_dataset_root(root)


def find_kagglehub_cache(handle: str) -> Path | None:
    """Return cached kagglehub ImageFolder root without triggering a download."""
    if not handle or "/" not in handle:
        return None

    owner, dataset_name = handle.split("/", 1)
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / owner / dataset_name
    if not cache_root.exists():
        return None

    version_dirs = sorted(cache_root.glob("versions/*"), reverse=True)
    for version_dir in version_dirs:
        resolved = resolve_imagefolder_root(version_dir)
        if resolved is not None:
            return resolved.resolve()
    return None


def _discover_kaggle_input_root(config: dict | None = None) -> Path | None:
    """Scan mounted Kaggle input datasets for root/data/<classes> or flat class layout."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        return None

    # Prefer verified mount: /kaggle/input/datasets/engrsakib02/.../data or flat classes
    canonical = resolve_kaggle_dataset_root()
    if canonical is not None:
        return canonical

    data_subdir = DEFAULT_KAGGLE_DATA_SUBDIR
    if config:
        data_subdir = config.get("dataset", {}).get("kaggle_data_subdir", data_subdir)

    legacy_roots = [
        kaggle_input / "neurological-disorders-mri-dataset-for-xai",
        kaggle_input / "datasets" / "engrsakib02" / "neurological-disorders-mri-dataset-for-xai",
    ]
    for preferred_root in legacy_roots:
        resolved = resolve_kaggle_dataset_root(preferred_root)
        if resolved is not None:
            return resolved

    datasets_dir = kaggle_input / "datasets"
    search_roots: list[Path] = []
    if datasets_dir.is_dir():
        search_roots.extend(
            child
            for child in sorted(datasets_dir.iterdir())
            if child.is_dir() and not child.name.startswith(".")
        )
    search_roots.extend(
        child
        for child in sorted(kaggle_input.iterdir())
        if child.is_dir() and child.name not in {"datasets"} and not child.name.startswith(".")
    )

    for dataset_dir in search_roots:
        nested_data = dataset_dir / data_subdir
        if _has_imagefolder_layout(nested_data):
            return nested_data.resolve()
        resolved = resolve_imagefolder_root(dataset_dir)
        if resolved is not None:
            return resolved.resolve()

    return None


def _discover_colab_data_root(project_root: Path) -> Path:
    """Resolve Colab dataset path from project data/ or /content subdirectories."""
    candidates: list[Path] = [project_root / "data"]

    content_root = Path("/content")
    if content_root.is_dir():
        for child in content_root.iterdir():
            if child.is_dir() and child.name not in {"sample_data", ".config"}:
                candidates.append(child)

    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = resolve_imagefolder_root(candidate)
        if resolved is not None:
            return resolved.resolve()
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate.resolve()

    return (project_root / "data").resolve()


def get_data_root(config: dict | None = None) -> Path:
    env_data = os.environ.get("NEURO_MRI_DATA_DIR")
    if env_data:
        path = Path(env_data).expanduser().resolve()
        resolved = resolve_dataset_root(path)
        return resolved if resolved is not None else path

    if config:
        dataset_cfg = config.get("dataset", {})
        source = dataset_cfg.get("source", "kaggle")
        kagglehub_handle = dataset_cfg.get("kagglehub_handle")
        if source == "kagglehub" and kagglehub_handle:
            cached = find_kagglehub_cache(kagglehub_handle)
            if cached is not None:
                return cached

    runtime = detect_runtime_env()
    project_root = get_project_root()

    if config:
        source = config.get("dataset", {}).get("source", "kaggle")
        if source == "gdrive":
            gdrive_path = config["dataset"].get("gdrive_path")
            if gdrive_path and Path(gdrive_path).exists():
                gdrive = Path(gdrive_path).resolve()
                resolved = resolve_imagefolder_root(gdrive)
                return resolved if resolved is not None else gdrive

    if runtime == RuntimeEnv.KAGGLE:
        kaggle_root = resolve_kaggle_dataset_root()
        if kaggle_root is not None:
            return kaggle_root
        kaggle_root = _discover_kaggle_input_root(config)
        if kaggle_root is not None:
            return kaggle_root

    default_data = project_root / "data"
    if runtime == RuntimeEnv.COLAB:
        return _discover_colab_data_root(project_root)

    if default_data.exists():
        resolved = resolve_imagefolder_root(default_data)
        return resolved if resolved is not None else default_data.resolve()

    return default_data.resolve()


def resolve_path(relative: str | Path, base: Path | None = None) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path.resolve()
    root = base or get_project_root()
    return (root / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
