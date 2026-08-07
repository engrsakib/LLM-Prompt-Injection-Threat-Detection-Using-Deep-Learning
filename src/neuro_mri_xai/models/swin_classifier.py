# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Swin Transformer classification backbone."""

from __future__ import annotations

import timm
import torch.nn as nn

from neuro_mri_xai.config import Config


def build_swin_classifier(config: Config, pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        config.model.backbone,
        pretrained=pretrained,
        num_classes=config.model.num_classes,
    )


def get_backbone_and_head_params(model: nn.Module) -> tuple[list, list]:
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("head") or name.startswith("fc") or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params
