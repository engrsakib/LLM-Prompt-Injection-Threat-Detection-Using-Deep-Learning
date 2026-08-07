"""Model registry and factory."""

from __future__ import annotations

import torch.nn as nn

from neuro_mri_xai.config import Config
from neuro_mri_xai.models.lora import apply_lora, get_trainable_param_count
from neuro_mri_xai.models.swin_classifier import build_swin_classifier


def build_model(config: Config, pretrained: bool = True) -> nn.Module:
    model = build_swin_classifier(config, pretrained=pretrained)
    if config.model.use_lora:
        model = apply_lora(model, config)
        trainable, total = get_trainable_param_count(model)
        print(f"LoRA enabled: {trainable:,} / {total:,} trainable parameters")
    return model
