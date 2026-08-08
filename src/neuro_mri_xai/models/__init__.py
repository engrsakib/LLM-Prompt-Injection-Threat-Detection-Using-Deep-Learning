# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Model registry and factory."""

from __future__ import annotations

import torch.nn as nn

from neuro_mri_xai.config import Config
from neuro_mri_xai.models.classifier import build_timm_classifier
from neuro_mri_xai.models.lora import apply_lora, get_trainable_param_count


def build_model(
    config: Config,
    pretrained: bool = True,
    backbone: str | None = None,
) -> nn.Module:
    model = build_timm_classifier(config, backbone=backbone, pretrained=pretrained)

    if config.model.use_lora:
        model = apply_lora(model, config)
        trainable, total = get_trainable_param_count(model)
        print(f"LoRA enabled: {trainable:,} / {total:,} trainable parameters")
    return model
