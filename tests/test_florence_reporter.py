"""Tests for Florence-2 processor loading compatibility."""

from __future__ import annotations

from unittest import mock

import pytest

from neuro_mri_xai.models.florence_reporter import _load_florence_processor


def test_load_florence_processor_prefers_slow_tokenizer() -> None:
    pytest.importorskip("transformers")
    slow_tokenizer = object()
    processor = object()
    model_id = "microsoft/Florence-2-base"

    with (
        mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=slow_tokenizer,
        ) as mock_tokenizer,
        mock.patch(
            "transformers.AutoProcessor.from_pretrained",
            return_value=processor,
        ) as mock_processor,
    ):
        result = _load_florence_processor(model_id)

    mock_tokenizer.assert_called_once_with(model_id, use_fast=False, trust_remote_code=True)
    mock_processor.assert_called_once_with(
        model_id,
        tokenizer=slow_tokenizer,
        trust_remote_code=True,
    )
    assert result is processor


def test_load_florence_processor_falls_back_to_use_fast_false() -> None:
    pytest.importorskip("transformers")
    slow_tokenizer = object()
    processor = object()
    model_id = "microsoft/Florence-2-base"

    with (
        mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=slow_tokenizer,
        ),
        mock.patch(
            "transformers.AutoProcessor.from_pretrained",
            side_effect=[TypeError("tokenizer kw unsupported"), processor],
        ) as mock_processor,
    ):
        result = _load_florence_processor(model_id)

    assert mock_processor.call_count == 2
    mock_processor.assert_any_call(model_id, use_fast=False, trust_remote_code=True)
    assert result is processor
