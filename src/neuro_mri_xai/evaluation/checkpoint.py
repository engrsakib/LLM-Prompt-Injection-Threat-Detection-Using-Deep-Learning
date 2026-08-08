# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Checkpoint loading utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from neuro_mri_xai.config import Config
from neuro_mri_xai.models import build_model
from neuro_mri_xai.models.lora import load_lora_adapter


def _adapter_dir_from_checkpoint(ckpt: dict) -> Path | None:
    adapter_dir = ckpt.get("lora_adapter_dir")
    if not adapter_dir:
        return None
    path = Path(adapter_dir)
    if not path.exists():
        return None
    if (path / "adapter_config.json").exists() or (path / "adapter_weights.pt").exists():
        return path
    return None


def load_checkpoint_model(
    checkpoint_path: str | Path,
    config: Config,
) -> tuple[nn.Module, list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names: list[str] = ckpt.get("class_names", config.get_class_names())

    if ckpt.get("backbone"):
        config.model.backbone = ckpt["backbone"]

    use_lora = bool(ckpt.get("use_lora", config.model.use_lora))
    config.model.use_lora = use_lora
    config.model.num_classes = len(class_names)

    adapter_path = _adapter_dir_from_checkpoint(ckpt)

    # Apply LoRA exactly once: prefer saved adapter dir on a raw timm backbone.
    if use_lora and adapter_path is not None:
        config.model.use_lora = False
        model = build_model(config, pretrained=False)
        config.model.use_lora = use_lora
        model = load_lora_adapter(model, adapter_path)
    else:
        model = build_model(config, pretrained=False)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    return model.to(device).eval(), class_names
