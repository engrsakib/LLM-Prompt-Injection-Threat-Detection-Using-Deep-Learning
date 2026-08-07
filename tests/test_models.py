"""Smoke tests for model forward passes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from neuro_mri_xai.config import Config, ModelConfig, load_config
from neuro_mri_xai.models import build_model
from neuro_mri_xai.models.lora import apply_lora, get_trainable_param_count
from neuro_mri_xai.models.sam_roi import _otsu_bbox, overlay_heatmap_on_mask


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
