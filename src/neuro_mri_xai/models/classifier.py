# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Generic timm classifiers (Swin, ConvNeXt, DenseNet) and Grad-CAM targets."""

from __future__ import annotations

import timm
import torch.nn as nn

from neuro_mri_xai.config import Config
from neuro_mri_xai.models.swin_classifier import unwrap_model

BENCHMARK_BACKBONES: dict[str, str] = {
    "swin": "swin_base_patch4_window7_224",
    "convnext": "convnext_base.fb_in22k_ft_in1k",
    "densenet": "densenet121",
}


def build_timm_classifier(
    config: Config,
    backbone: str | None = None,
    pretrained: bool = True,
) -> nn.Module:
    """Create a timm classification model for any supported backbone."""
    name = backbone or config.model.backbone
    use_pretrained = config.model.pretrained if pretrained else False
    kwargs: dict = {
        "pretrained": use_pretrained,
        "num_classes": config.model.num_classes,
    }
    if name.startswith("swin"):
        kwargs["drop_path_rate"] = config.model.drop_path_rate
    return timm.create_model(name, **kwargs)


def apply_partial_freeze(model: nn.Module) -> tuple[int, int]:
    """Freeze all params; unfreeze classification head and final backbone stage."""
    backbone = unwrap_model(model)

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(backbone, "head"):
        for param in backbone.head.parameters():
            param.requires_grad = True

    if hasattr(backbone, "layers"):
        for param in backbone.layers[-1].parameters():
            param.requires_grad = True
    elif hasattr(backbone, "stages"):
        for param in backbone.stages[-1].parameters():
            param.requires_grad = True
    elif hasattr(backbone, "features"):
        for param in backbone.features[-1].parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return a layer suitable for Grad-CAM (Swin, ConvNeXt, or DenseNet)."""
    from neuro_mri_xai.models.swin_classifier import get_swin_target_layers

    backbone = unwrap_model(model)
    if hasattr(backbone, "layers"):
        layer, _ = get_swin_target_layers(model)
        return layer

    if hasattr(backbone, "stages"):
        last_stage = backbone.stages[-1]
        blocks = getattr(last_stage, "blocks", None)
        if blocks is not None:
            return blocks[-1]
        return last_stage

    if hasattr(backbone, "features"):
        return backbone.features[-1]

    raise ValueError("Could not determine Grad-CAM target layer for this backbone")
