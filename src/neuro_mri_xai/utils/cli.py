# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Shared CLI helpers for entry-point scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuro_mri_xai.config import Config
from neuro_mri_xai.utils.paths import resolve_dataset_root


def add_data_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Register a standard --data-dir flag on an argparse parser."""
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override dataset directory (ImageFolder root). "
        "Takes precedence over config and NEURO_MRI_DATA_DIR.",
    )


def apply_data_dir_override(config: Config, data_dir: str | Path | None) -> Config:
    """Apply CLI --data-dir override, resolving nested ImageFolder layouts."""
    if not data_dir:
        return config

    path = Path(data_dir).expanduser().resolve()
    resolved = resolve_dataset_root(path)
    if resolved is not None:
        config.dataset.data_dir = resolved
    elif path.is_dir():
        config.dataset.data_dir = path
    else:
        # Missing --data-dir: defer to kagglehub fallback in ensure_dataset_available()
        config.dataset.data_dir = path
    return config
