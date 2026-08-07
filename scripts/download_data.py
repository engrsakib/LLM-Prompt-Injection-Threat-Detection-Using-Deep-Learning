#!/usr/bin/env python3
# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Download neurological MRI dataset via kagglehub, Kaggle CLI, or Google Drive."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuro_mri_xai.config import load_config
from neuro_mri_xai.data.download import download_dataset
from neuro_mri_xai.utils.paths import get_project_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MRI dataset")
    parser.add_argument(
        "--source",
        choices=["kagglehub", "kaggle", "gdrive"],
        default=None,
        help="Dataset source (default: configs/default.yaml dataset.source)",
    )
    parser.add_argument(
        "--use-kagglehub",
        action="store_true",
        help="Try kagglehub first, then fall back to --source or config fallback",
    )
    parser.add_argument("--config", default=str(get_project_root() / "configs" / "default.yaml"))
    parser.add_argument("--output", default=None, help="Download destination directory")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = download_dataset(
        config,
        source=args.source,
        output=Path(args.output) if args.output else None,
        use_kagglehub=args.use_kagglehub,
    )

    print(f"Dataset ready at: {data_path}")
    classes = sorted(
        d.name for d in data_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    print(f"Classes ({len(classes)}): {classes}")


if __name__ == "__main__":
    main()
