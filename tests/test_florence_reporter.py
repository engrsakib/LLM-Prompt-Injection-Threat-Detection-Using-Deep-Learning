"""Tests for Florence-2 processor loading compatibility."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from neuro_mri_xai.models import florence_reporter as fr


@pytest.fixture(autouse=True)
def reset_florence_patch_flag() -> None:
    fr._tokenizer_compat_patched = False
    yield
    fr._tokenizer_compat_patched = False


def test_prepare_florence_inputs_aligns_float_dtype_to_model() -> None:
    pytest.importorskip("torch")
    model = SimpleNamespace()
    model.dtype = torch.float16
    model.parameters = lambda: iter([torch.tensor(1.0, dtype=torch.float16)])

    raw = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "pixel_values": torch.randn(1, 3, 224, 224, dtype=torch.float32),
    }
    prepared = fr._prepare_florence_inputs(raw, model, torch.device("cpu"))

    assert prepared["input_ids"].dtype == torch.long
    assert prepared["pixel_values"].dtype == torch.float16


def test_patch_adds_additional_special_tokens_property() -> None:
    pytest.importorskip("transformers")
    import transformers.tokenization_utils_base as tok_utils

    fr._patch_tokenizer_additional_special_tokens()
    assert isinstance(tok_utils.PreTrainedTokenizerBase.additional_special_tokens, property)


def test_additional_special_tokens_getter_fallback() -> None:
    stub = SimpleNamespace(all_special_tokens=["<s>", "</s>"])
    assert fr._additional_special_tokens_getter(stub) == ["<s>", "</s>"]
    assert fr._additional_special_tokens_getter(SimpleNamespace()) == []


def test_ensure_instance_sets_fallback_list() -> None:
    tokenizer = SimpleNamespace()
    fr._ensure_instance_additional_special_tokens(tokenizer)
    assert tokenizer._additional_special_tokens == []


def test_load_florence_processor_applies_patch_and_uses_slow_tokenizer() -> None:
    pytest.importorskip("transformers")
    slow_tokenizer = SimpleNamespace(_additional_special_tokens=[])
    processor = SimpleNamespace(tokenizer=slow_tokenizer)
    model_id = "microsoft/Florence-2-base"

    with (
        mock.patch.object(fr, "_patch_tokenizer_additional_special_tokens") as mock_patch,
        mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=slow_tokenizer,
        ) as mock_tokenizer,
        mock.patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=processor,
        ) as mock_processor,
    ):
        result = fr._load_florence_processor(model_id)

    mock_patch.assert_called_once()
    mock_tokenizer.assert_called_once_with(model_id, use_fast=False, trust_remote_code=True)
    mock_processor.assert_called_once_with(
        model_id,
        tokenizer=slow_tokenizer,
        trust_remote_code=True,
    )
    assert result is processor


def test_load_florence_processor_falls_back_to_use_fast_false() -> None:
    pytest.importorskip("transformers")
    slow_tokenizer = SimpleNamespace(_additional_special_tokens=[])
    processor = SimpleNamespace(tokenizer=slow_tokenizer)
    model_id = "microsoft/Florence-2-base"

    with (
        mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=slow_tokenizer,
        ),
        mock.patch(
            "transformers.AutoProcessor.from_pretrained",
            side_effect=[RuntimeError("tokenizer override ignored"), processor],
        ) as mock_processor,
    ):
        result = fr._load_florence_processor(model_id)

    assert mock_processor.call_count == 2
    mock_processor.assert_any_call(model_id, use_fast=False, trust_remote_code=True)
    assert result is processor
