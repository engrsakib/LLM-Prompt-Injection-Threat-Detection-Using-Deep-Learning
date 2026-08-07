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
