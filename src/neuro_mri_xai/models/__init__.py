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
from neuro_mri_xai.models.lora import apply_lora, get_trainable_param_count
from neuro_mri_xai.models.swin_classifier import build_swin_classifier


def build_model(config: Config, pretrained: bool = True) -> nn.Module:
    if config.model.backbone.startswith("swin"):
        model = build_swin_classifier(config, pretrained=pretrained)
    else:
        model = build_swin_classifier(config, pretrained=pretrained)

    if config.model.use_lora:
        model = apply_lora(model, config)
        trainable, total = get_trainable_param_count(model)
        print(f"LoRA enabled: {trainable:,} / {total:,} trainable parameters")
    return model
