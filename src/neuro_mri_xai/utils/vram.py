# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""GPU memory lifecycle helpers for sequential model loading."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Callable, Iterator

import torch

from neuro_mri_xai.config import Config


def gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024**3)


def empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_mem(stage: str) -> None:
    if torch.cuda.is_available():
        print(f"[VRAM] {stage}: {gpu_mem_gb():.2f} GB allocated")


@contextmanager
def model_slot(
    load_fn: Callable[[], object],
    unload_fn: Callable[[], None],
    stage: str,
    config: Config | None = None,
) -> Iterator[object]:
    """Load a model slot, yield it, then unload to free VRAM."""
    obj = load_fn()
    if config is None or getattr(config.vram, "empty_cache_between", True):
        log_gpu_mem(f"{stage} loaded")
    try:
        yield obj
    finally:
        unload_fn()
        if config is None or getattr(config.vram, "empty_cache_between", True):
            empty_cuda_cache()
            log_gpu_mem(f"{stage} unloaded")
