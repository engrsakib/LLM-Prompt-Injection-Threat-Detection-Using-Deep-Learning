# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for VRAM lifecycle helpers."""

from __future__ import annotations

from neuro_mri_xai.models.sam_roi import unload_sam
from neuro_mri_xai.models.florence_reporter import unload_florence
from neuro_mri_xai.utils.vram import empty_cuda_cache, gpu_mem_gb


def test_gpu_mem_gb_no_cuda():
    # Should not raise even without CUDA
    mem = gpu_mem_gb()
    assert mem >= 0.0


def test_unload_singletons():
    unload_sam()
    unload_florence()
    empty_cuda_cache()
