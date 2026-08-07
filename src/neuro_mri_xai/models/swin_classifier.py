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


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip PEFT / wrapper layers to reach the timm backbone."""
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "base_model"):
            current = current.base_model  # type: ignore[assignment]
            continue
        if hasattr(current, "model") and isinstance(getattr(current, "model"), nn.Module):
            inner = current.model
            if inner is not current:
                current = inner
                continue
        break
    return current


def build_swin_classifier(config: Config, pretrained: bool = True) -> nn.Module:
    use_pretrained = config.model.pretrained if pretrained else False
    return timm.create_model(
        config.model.backbone,
        pretrained=use_pretrained,
        num_classes=config.model.num_classes,
        drop_path_rate=config.model.drop_path_rate,
    )


def get_swin_target_layers(model: nn.Module) -> tuple[nn.Module, nn.Module | None]:
    """Return (gradcam_layer, attention_layer) for timm Swin models."""
    backbone = unwrap_model(model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose Swin 'layers' attribute")

    last_stage = layers[-1]
    blocks = getattr(last_stage, "blocks", None)
    if blocks is None:
        raise ValueError("Swin stage missing 'blocks'")

    last_block = blocks[-1]
    gradcam_layer = getattr(last_block, "norm2", last_block)
    attention_layer = getattr(last_block, "attn", None)
    return gradcam_layer, attention_layer


def get_backbone_and_head_params(model: nn.Module) -> tuple[list, list]:
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_lora = "lora_" in name
        is_head = (
            name.startswith("head")
            or name.startswith("fc")
            or "classifier" in name
            or name.endswith(".head.weight")
            or name.endswith(".head.bias")
        )
        if is_head and not is_lora:
            head_params.append(param)
        elif is_lora or (not is_head):
            backbone_params.append(param)
    return backbone_params, head_params
