"""Smoke tests for model forward passes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from neuro_mri_xai.config import Config, load_config
from neuro_mri_xai.data.constants import EXPECTED_CLASS_NAMES
from neuro_mri_xai.models import build_model
from neuro_mri_xai.models.lora import apply_lora, get_trainable_param_count
from neuro_mri_xai.models.sam_roi import _otsu_bbox, overlay_heatmap_on_mask
from neuro_mri_xai.models.swin_classifier import (
    apply_swin_partial_freeze,
    get_swin_target_layers,
    unwrap_model,
)


@pytest.fixture
def config():
    cfg = load_config()
    cfg.model.use_lora = False
    cfg.model.num_classes = 8
    return cfg


def test_swin_forward_pass(config: Config):
    pytest.importorskip("timm")
    model = build_model(config, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 8)


def test_lora_reduces_trainable_params(config: Config):
    pytest.importorskip("timm")
    pytest.importorskip("peft")
    base = build_model(config, pretrained=False)
    _, total = get_trainable_param_count(base)
    config.model.use_lora = True
    lora_model = apply_lora(base, config)
    trainable, _ = get_trainable_param_count(lora_model)
    assert trainable < total


def test_otsu_bbox_fallback():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 200
    x1, y1, x2, y2 = _otsu_bbox(img)
    assert x2 > x1 and y2 > y1


def test_sam_overlay_shape():
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    heatmap = np.random.rand(224, 224).astype(np.float32)
    mask = np.ones((224, 224), dtype=np.uint8) * 255
    out = overlay_heatmap_on_mask(img, heatmap, mask)
    assert out.shape == img.shape


def test_unwrap_model_and_target_layers(config: Config):
    pytest.importorskip("timm")
    model = build_model(config, pretrained=False)
    backbone = unwrap_model(model)
    assert hasattr(backbone, "layers")
    gradcam_layer, attn_layer = get_swin_target_layers(model)
    assert gradcam_layer is not None


def test_apply_swin_partial_freeze_leaves_last_stage_and_head_trainable(config: Config):
    pytest.importorskip("timm")
    model = build_model(config, pretrained=False)
    trainable_before, total = get_trainable_param_count(model)
    assert trainable_before == total

    trainable_after, _ = apply_swin_partial_freeze(model)
    assert 0 < trainable_after < trainable_before

    for name, param in model.named_parameters():
        in_last_stage = name.startswith("layers.3") or ".layers.3." in name
        in_head = name.startswith("head") or ".head." in name
        if in_last_stage or in_head:
            assert param.requires_grad, f"Expected trainable: {name}"
        else:
            assert not param.requires_grad, f"Expected frozen: {name}"


def test_explain_sample_return_keys(tmp_path, config: Config):
    pytest.importorskip("timm")
    from neuro_mri_xai.explainability.pipeline import explain_sample

    data_root = tmp_path / "data"
    class_dir = data_root / "Normal"
    class_dir.mkdir(parents=True)
    img_path = class_dir / "sample.jpg"
    Image.new("RGB", (224, 224), color=(128, 64, 32)).save(img_path)

    config.sam.enabled = False
    model = build_model(config, pretrained=False)
    result = explain_sample(model, img_path, config, EXPECTED_CLASS_NAMES, tmp_path / "xai")
    assert "prediction" in result
    assert "confidence" in result
    assert "gradcam_path" in result
    assert "attention_path" in result
