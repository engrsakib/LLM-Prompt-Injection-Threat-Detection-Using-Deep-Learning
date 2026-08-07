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


def load_checkpoint_model(
    checkpoint_path: str | Path,
    config: Config,
) -> tuple[nn.Module, list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names: list[str] = ckpt.get("class_names", config.classes)
    config.model.num_classes = len(class_names)
    model = build_model(config, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(device).eval(), class_names
