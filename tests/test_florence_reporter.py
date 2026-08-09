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


def test_normalize_florence_task_defaults_and_vqa() -> None:
    assert fr._normalize_florence_task("") == fr.DEFAULT_FLORENCE_CAPTION_TASK
    assert fr._normalize_florence_task("  ") == fr.DEFAULT_FLORENCE_CAPTION_TASK
    assert fr._normalize_florence_task("<CAPTION>") == "<CAPTION>"
    assert fr._normalize_florence_task("<VQA>What is shown?") == "<VQA>What is shown?"


def test_count_florence_image_tokens() -> None:
    processor = SimpleNamespace(image_token_id=42)
    input_ids = torch.tensor([[1, 42, 42, 3, 42]])
    assert fr._count_florence_image_tokens(processor, input_ids) == 3


def test_validate_florence_inputs_raises_on_zero_image_tokens() -> None:
    processor = SimpleNamespace(image_token_id=99)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.randn(1, 3, 224, 224),
    }
    with pytest.raises(ValueError, match="0 image tokens"):
        fr._validate_florence_inputs(inputs, processor)


def test_task_prompt_with_image_marker() -> None:
    assert fr._task_prompt_with_image_marker("<CAPTION>") == "<image><CAPTION>"
    assert fr._task_prompt_with_image_marker("<image><CAPTION>") == "<image><CAPTION>"


def test_build_florence_processor_inputs_uses_image_prefixed_prompt() -> None:
    token_id = 7
    image = mock.Mock()
    calls: list[str | list[str]] = []

    class StubProcessor:
        image_token_id = token_id
        image_token = "<image>"
        num_image_tokens = 2
        image_seq_length = 2

        def __call__(self, *, text, images, return_tensors, truncation):
            calls.append(text)
            has_image_prefix = (
                text == "<image><CAPTION>"
                or (isinstance(text, list) and text == ["<image><CAPTION>"])
            )
            ids = torch.tensor([[token_id, token_id, 1, 2]]) if has_image_prefix else torch.tensor([[1, 2, 3]])
            return {
                "input_ids": ids,
                "pixel_values": torch.randn(1, 3, 224, 224),
            }

    processor = StubProcessor()
    inputs = fr._build_florence_processor_inputs(processor, "<CAPTION>", image)
    assert "<image><CAPTION>" in calls or ["<image><CAPTION>"] in calls
    assert fr._count_florence_image_tokens(processor, inputs["input_ids"]) > 0


def test_build_florence_processor_inputs_with_stub_processor() -> None:
    token_id = 7
    image = mock.Mock()

    class StubProcessor:
        image_token_id = token_id
        image_token = "<image>"
        num_image_tokens = 2
        image_seq_length = 2

        def __call__(self, *, text, images, return_tensors, truncation):
            assert truncation is False
            return {
                "input_ids": torch.tensor([[token_id, token_id, 1, 2]]),
                "pixel_values": torch.randn(1, 3, 224, 224),
            }

    processor = StubProcessor()
    inputs = fr._build_florence_processor_inputs(processor, "<CAPTION>", image)
    assert fr._count_florence_image_tokens(processor, inputs["input_ids"]) > 0
    assert inputs["pixel_values"].shape == (1, 3, 224, 224)


def test_manual_florence_prompt_inputs_adds_image_tokens() -> None:
    token_id = 11
    image = mock.Mock()

    class StubTokenizer:
        bos_token = "<s>"
        eos_token = "</s>"

        def __call__(self, texts, return_tensors, truncation):
            assert truncation is False
            assert texts[0].startswith("<image><image>")
            return {"input_ids": torch.tensor([[token_id, token_id, 5, 6]])}

        def convert_tokens_to_ids(self, token: str) -> int:
            return token_id if token == "<image>" else 0

    class StubImageProcessor:
        def __call__(self, img, return_tensors):
            return {"pixel_values": torch.randn(1, 3, 8, 8)}

    processor = SimpleNamespace(
        tokenizer=StubTokenizer(),
        image_token="<image>",
        num_image_tokens=2,
        image_seq_length=2,
        image_token_id=token_id,
        _construct_prompts=lambda tasks: list(tasks),
        image_processor=StubImageProcessor(),
    )

    manual = fr._manual_florence_prompt_inputs(
        processor,
        "<MORE_DETAILED_CAPTION>",
        image,
    )
    assert manual is not None
    assert fr._count_florence_image_tokens(processor, manual["input_ids"]) == 2


def test_generate_diagnostic_text_falls_back_when_caption_unavailable() -> None:
    config = SimpleNamespace(get_class_names=lambda: ["Normal", "MS"])
    image = mock.Mock()

    with mock.patch.object(fr, "generate_caption", return_value=None):
        text = fr.generate_diagnostic_text(image, "Normal", 0.91, config)

    assert "Predicted diagnosis: Normal" in text
    assert "Florence-2 caption unavailable" in text
    assert "DISCLAIMER" in text


def test_generate_diagnostic_text_never_raises_on_unexpected_error() -> None:
    config = SimpleNamespace(get_class_names=lambda: ["Normal", "MS"])
    image = mock.Mock()

    with mock.patch.object(fr, "generate_caption", side_effect=RuntimeError("CUDA OOM")):
        text = fr.generate_diagnostic_text(image, "Normal", 0.91, config)

    assert "Predicted diagnosis: Normal" in text
    assert "Florence-2 caption unavailable" in text


def test_validate_florence_inputs_raises_on_token_count_mismatch() -> None:
    processor = SimpleNamespace(image_token_id=99, num_image_tokens=65, image_seq_length=65)
    inputs = {
        "input_ids": torch.tensor([[99, 99, 1, 2]]),
        "pixel_values": torch.randn(1, 3, 224, 224),
    }
    with pytest.raises(ValueError, match="image token mismatch"):
        fr._validate_florence_inputs(inputs, processor)


def test_load_florence_model_prefers_auto_model_for_causal_lm() -> None:
    pytest.importorskip("transformers")
    model_stub = SimpleNamespace(to=lambda device: model_stub, eval=lambda: model_stub)
    model_id = "microsoft/Florence-2-base"

    with mock.patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        return_value=model_stub,
    ) as mock_auto:
        result = fr._load_florence_model(model_id, torch.float32, "cpu")

    mock_auto.assert_called_once_with(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    assert result is model_stub
