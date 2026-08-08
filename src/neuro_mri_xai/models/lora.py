# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""LoRA fine-tuning via PEFT for timm Swin models."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from neuro_mri_xai.config import Config


def discover_lora_targets(model: nn.Module, suffixes: list[str]) -> list[str]:
    """Find module names whose final component matches a LoRA target suffix."""
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in suffixes or any(name.endswith(f".{s}") for s in suffixes):
            targets.append(name)
    return targets


def freeze_backbone_except_lora(model: nn.Module, modules_to_save: list[str] | None = None) -> None:
    """Freeze all parameters except LoRA adapters and optional modules_to_save."""
    save_patterns = modules_to_save or []
    for name, param in model.named_parameters():
        keep = "lora_" in name or any(
            name.startswith(f"{p}.") or name == p or name.endswith(f".{p}.weight")
            for p in save_patterns
        )
        param.requires_grad = keep


def inject_lora_into_timm(model: nn.Module, config: Config) -> nn.Module:
    """Apply LoRA adapters to timm Swin linear layers."""
    targets = discover_lora_targets(model, config.model.lora_target_modules)
    if not targets:
        targets = config.model.lora_target_modules

    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        target_modules=targets if targets else config.model.lora_target_modules,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        modules_to_save=config.model.lora_modules_to_save or None,
    )

    peft_model = get_peft_model(model, lora_config)
    freeze_backbone_except_lora(peft_model, config.model.lora_modules_to_save)

    for name, param in peft_model.named_parameters():
        if any(
            name.startswith(f"{m}.") or name == m or "lora_" in name
            for m in config.model.lora_modules_to_save
        ):
            param.requires_grad = True

    return peft_model


def is_peft_model(model: nn.Module) -> bool:
    """Return True when ``model`` is a top-level ``PeftModel`` wrapper."""
    try:
        from peft import PeftModel

        return isinstance(model, PeftModel)
    except ImportError:
        return False


def unwrap_peft_base(model: nn.Module) -> nn.Module:
    """Return the raw base module suitable for ``PeftModel.from_pretrained``."""
    if not is_peft_model(model):
        return model

    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        if base is not model and is_peft_model(base):
            return unwrap_peft_base(base)
        if hasattr(base, "model") and isinstance(base.model, nn.Module):
            return base.model
        return base

    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and isinstance(base.model, nn.Module):
            return base.model
        return base

    return model


def apply_lora(model: nn.Module, config: Config) -> nn.Module:
    return inject_lora_into_timm(model, config)


def save_lora_adapter(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
        return
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, path / "adapter_weights.pt")


def load_lora_adapter(model: nn.Module, path: str | Path) -> nn.Module:
    path = Path(path)
    if (path / "adapter_config.json").exists():
        from peft import PeftModel

        if is_peft_model(model):
            if hasattr(model, "load_adapter"):
                model.load_adapter(str(path), adapter_name="default")
                return model
            model = unwrap_peft_base(model)

        return PeftModel.from_pretrained(model, str(path))

    weights_file = path / "adapter_weights.pt"
    if weights_file.exists():
        state = torch.load(weights_file, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
    return model


def get_trainable_param_count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
