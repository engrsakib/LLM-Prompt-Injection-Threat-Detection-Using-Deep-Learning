"""Detect the execution environment (Kaggle / local), accelerator, VRAM, disk and recommend a tier.

Pure inspection: no network, no credentials, no Kaggle API. Credentials (``HF_TOKEN``) are only
ever read from environment variables; on Kaggle attach them as a Secret which Kaggle exposes to
``os.environ`` when enabled for the notebook.

CLI::

    python -m src.kaggle.env
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Which compute tiers (see registry.compute_tier) a given free VRAM budget can serve.
TIER_MIN_VRAM_GB = {"cpu_ok": 0.0, "gpu_small": 6.0, "gpu_medium": 12.0, "gpu_large_qlora": 15.0}


@dataclass
class EnvInfo:
    is_kaggle: bool
    input_dir: str
    working_dir: str
    cuda_available: bool
    gpu_count: int
    gpu_names: list[str]
    gpu_total_mem_gb: list[float]
    bf16_supported: bool
    disk_free_gb: float
    cpu_count: int
    hf_token_present: bool
    hf_offline: bool
    bitsandbytes_available: bool
    recommended_tiers: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect(working_dir: str | Path | None = None) -> EnvInfo:
    is_kaggle = KAGGLE_INPUT.exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    work = Path(working_dir) if working_dir else (KAGGLE_WORKING if is_kaggle else Path("outputs"))
    notes: list[str] = []
    try:
        import torch

        cuda = torch.cuda.is_available()
        count = torch.cuda.device_count() if cuda else 0
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        mem = [round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2) for i in range(count)]
        bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    except Exception as exc:  # noqa: BLE001
        cuda, count, names, mem, bf16 = False, 0, [], [], False
        notes.append(f"torch unavailable: {exc}")
    try:
        import bitsandbytes  # noqa: F401

        bnb = True
    except Exception:  # noqa: BLE001
        bnb = False
    try:
        disk = shutil.disk_usage(str(work if work.exists() else Path.cwd())).free / 1e9
    except OSError:
        disk = float("nan")
    max_mem = max(mem) if mem else 0.0
    tiers = [t for t, need in TIER_MIN_VRAM_GB.items() if (max_mem >= need if need > 0 else True)]
    if "gpu_large_qlora" in tiers and not bnb:
        tiers.remove("gpu_large_qlora")
        notes.append("bitsandbytes missing: 4-bit QLoRA tier disabled (install bitsandbytes on Linux+CUDA)")
    if not cuda:
        notes.append("no CUDA device: only cpu_ok tier is realistic; decoder LLMs must not be loaded")
    if is_kaggle and disk < 15:
        notes.append("less than 15 GB free in /kaggle/working: keep checkpoints small / delete 'last' dirs")
    return EnvInfo(
        is_kaggle=is_kaggle,
        input_dir=str(KAGGLE_INPUT if is_kaggle else Path("data").resolve()),
        working_dir=str(work),
        cuda_available=cuda,
        gpu_count=count,
        gpu_names=names,
        gpu_total_mem_gb=mem,
        bf16_supported=bf16,
        disk_free_gb=round(disk, 2),
        cpu_count=os.cpu_count() or 1,
        hf_token_present=bool(os.environ.get("HF_TOKEN")),
        hf_offline=os.environ.get("HF_HUB_OFFLINE") == "1",
        bitsandbytes_available=bnb,
        recommended_tiers=tiers,
        notes=notes,
    )


def find_kaggle_splits_dir(preferred: str | Path | None = None) -> Path | None:
    """Locate a mounted dataset directory containing ``manifest.json`` + ``train.parquet``."""
    candidates = [Path(preferred)] if preferred else []
    if KAGGLE_INPUT.exists():
        candidates += sorted(KAGGLE_INPUT.glob("*"))
    for cand in candidates:
        for path in [cand, *cand.glob("*")]:
            if (path / "manifest.json").exists() and (path / "train.parquet").exists():
                return path
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Print environment / accelerator info as JSON.")
    p.add_argument("--working-dir", default=None)
    args = p.parse_args(argv)
    print(json.dumps(detect(args.working_dir).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
