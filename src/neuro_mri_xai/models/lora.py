"""LoRA fine-tuning via PEFT for timm Swin models."""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from peft import LoraConfig, get_peft_model

from neuro_mri_xai.config import Config


def apply_lora(model: nn.Module, config: Config) -> nn.Module:
    lora_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.lora_target_modules,
        lora_dropout=config.model.lora_dropout,
        bias="none",
    )
    return get_peft_model(model, lora_config)


def save_lora_adapter(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
    else:
        import torch
        torch.save(model.state_dict(), path / "adapter_weights.pt")


def load_lora_adapter(model: nn.Module, path: str | Path) -> nn.Module:
    path = Path(path)
    if (path / "adapter_config.json").exists():
        from peft import PeftModel
        return PeftModel.from_pretrained(model, path)
    import torch
    state = torch.load(path / "adapter_weights.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


def get_trainable_param_count(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
