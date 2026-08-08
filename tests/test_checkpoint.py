"""Tests for checkpoint and LoRA adapter loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuro_mri_xai.config import Config, load_config
from neuro_mri_xai.data.constants import EXPECTED_CLASS_NAMES
from neuro_mri_xai.evaluation.checkpoint import load_checkpoint_model
from neuro_mri_xai.models import build_model
from neuro_mri_xai.models.lora import is_peft_model, load_lora_adapter, unwrap_peft_base


@pytest.fixture
def config() -> Config:
    cfg = load_config()
    cfg.model.use_lora = True
    cfg.model.num_classes = len(EXPECTED_CLASS_NAMES)
    return cfg


def test_load_lora_adapter_does_not_double_wrap(config: Config, tmp_path: Path) -> None:
    pytest.importorskip("peft")
    from peft import PeftModel

    config.model.use_lora = False
    base = build_model(config, pretrained=False)
    assert not is_peft_model(base)

    config.model.use_lora = True
    lora_model = build_model(config, pretrained=False)
    assert is_peft_model(lora_model)

    adapter_dir = tmp_path / "lora_adapter"
    lora_model.save_pretrained(adapter_dir)

    reloaded = load_lora_adapter(lora_model, adapter_dir)
    assert isinstance(reloaded, PeftModel)
    inner = reloaded.base_model.model
    assert not isinstance(inner, PeftModel)


def test_unwrap_peft_base_returns_timm_backbone(config: Config) -> None:
    pytest.importorskip("peft")

    lora_model = build_model(config, pretrained=False)
    base = unwrap_peft_base(lora_model)
    assert not isinstance(base, type(lora_model))
    assert hasattr(base, "head")


def test_load_checkpoint_model_single_peft_wrap(
    config: Config,
    tmp_path: Path,
) -> None:
    pytest.importorskip("peft")

    model = build_model(config, pretrained=False)
    adapter_dir = tmp_path / "lora_adapter"
    model.save_pretrained(adapter_dir)

    ckpt_path = tmp_path / "best_swin.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": EXPECTED_CLASS_NAMES,
            "use_lora": True,
            "backbone": config.model.backbone,
            "lora_adapter_dir": str(adapter_dir),
        },
        ckpt_path,
    )

    loaded, class_names = load_checkpoint_model(ckpt_path, config)
    assert class_names == EXPECTED_CLASS_NAMES
    assert is_peft_model(loaded)

    x = torch.randn(1, 3, 224, 224)
    out = loaded(x)
    assert out.shape == (1, len(EXPECTED_CLASS_NAMES))
