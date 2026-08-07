#!/usr/bin/env python3
"""Download SAM and other model weight checkpoints."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from neuro_mri_xai.config import load_config
from neuro_mri_xai.utils.paths import ensure_dir, get_project_root

SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    ensure_dir(dest.parent)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--config", default=str(get_project_root() / "configs" / "default.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    weights_dir = ensure_dir(get_project_root() / config.sam.weights_dir)
    url = SAM_URLS.get(config.sam.model_type, SAM_URLS["vit_b"])
    dest = weights_dir / config.sam.checkpoint
    download_file(url, dest)
    print(f"SAM weights saved to {dest}")


if __name__ == "__main__":
    main()
