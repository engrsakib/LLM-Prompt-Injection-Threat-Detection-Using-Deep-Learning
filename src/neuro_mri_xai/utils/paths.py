"""Environment-aware path resolution."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


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


def get_data_root(config: dict | None = None) -> Path:
    env_data = os.environ.get("NEURO_MRI_DATA_DIR")
    if env_data:
        return Path(env_data).resolve()

    runtime = detect_runtime_env()
    project_root = get_project_root()

    if config:
        source = config.get("dataset", {}).get("source", "kaggle")
        if source == "gdrive":
            gdrive_path = config["dataset"].get("gdrive_path")
            if gdrive_path and Path(gdrive_path).exists():
                return Path(gdrive_path).resolve()

    if runtime == RuntimeEnv.KAGGLE:
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for path in kaggle_input.rglob("data"):
                if path.is_dir() and any(path.iterdir()):
                    return path.resolve()
            for path in kaggle_input.iterdir():
                if path.is_dir():
                    data_sub = path / "data"
                    if data_sub.is_dir():
                        return data_sub.resolve()
                    return path.resolve()

    default_data = project_root / "data"
    if runtime == RuntimeEnv.COLAB:
        colab_data = Path("/content/data")
        if colab_data.exists() and any(colab_data.iterdir()):
            return colab_data.resolve()
        return colab_data

    if default_data.exists():
        return default_data.resolve()

    return default_data


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
