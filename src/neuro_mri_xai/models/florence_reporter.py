# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Florence-2 natural language captioning for MRI reports."""

from __future__ import annotations

import logging

import torch
from PIL import Image

from neuro_mri_xai.config import Config

logger = logging.getLogger(__name__)

_florence_model = None
_florence_processor = None
_tokenizer_compat_patched = False


def _additional_special_tokens_getter(tokenizer: object) -> list[str]:
    """Safe accessor compatible with slow and fast Hugging Face tokenizers."""
    stored = getattr(tokenizer, "_additional_special_tokens", None)
    if stored is not None:
        return list(stored)

    getter = getattr(tokenizer, "get_additional_special_tokens", None)
    if callable(getter):
        try:
            return list(getter())
        except Exception:
            pass

    all_special = getattr(tokenizer, "all_special_tokens", None)
    if all_special:
        return list(all_special)

    return []


def _patch_tokenizer_additional_special_tokens() -> None:
    """Monkey-patch tokenizer classes used by Florence-2 remote processor code."""
    global _tokenizer_compat_patched
    if _tokenizer_compat_patched:
        return

    import transformers.tokenization_utils_base as tok_utils

    prop = property(_additional_special_tokens_getter)
    patched: set[int] = set()

    def _apply(cls: type) -> None:
        cls_id = id(cls)
        if cls_id in patched:
            return
        try:
            cls.additional_special_tokens = prop
            patched.add(cls_id)
        except (TypeError, AttributeError):
            logger.debug("Could not patch additional_special_tokens on %s", cls)

    _apply(tok_utils.PreTrainedTokenizerBase)

    optional_modules = (
        "transformers.tokenization_utils_fast",
        "transformers.tokenization_utils",
    )
    optional_names = ("PreTrainedTokenizerFast", "TokenizersBackend")

    for module_name in optional_modules:
        try:
            module = __import__(module_name, fromlist=list(optional_names))
        except ImportError:
            continue
        for name in optional_names:
            cls = getattr(module, name, None)
            if isinstance(cls, type):
                _apply(cls)

    _tokenizer_compat_patched = True


def _ensure_instance_additional_special_tokens(tokenizer: object) -> None:
    """Ensure tokenizer instances never raise on `.additional_special_tokens`."""
    try:
        _ = tokenizer.additional_special_tokens
    except AttributeError:
        try:
            tokenizer._additional_special_tokens = []
        except Exception:
            logger.debug("Could not set _additional_special_tokens on %s", type(tokenizer))


def _load_florence_processor(model_id: str):
    """Load Florence-2 processor with tokenizer compatibility patches applied."""
    from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

    _patch_tokenizer_additional_special_tokens()
    trust = {"trust_remote_code": True}

    slow_tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **trust)
    _ensure_instance_additional_special_tokens(slow_tokenizer)

    try:
        processor = AutoProcessor.from_pretrained(model_id, tokenizer=slow_tokenizer, **trust)
        if getattr(processor, "tokenizer", None) is not None:
            _ensure_instance_additional_special_tokens(processor.tokenizer)
        return processor
    except Exception as exc:
        logger.warning(
            "AutoProcessor.from_pretrained(tokenizer=...) failed (%s); retrying fallbacks",
            exc,
        )

    try:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=False, **trust)
        if getattr(processor, "tokenizer", None) is not None:
            _ensure_instance_additional_special_tokens(processor.tokenizer)
        return processor
    except Exception as exc:
        logger.warning("AutoProcessor use_fast=False failed (%s); assembling manually", exc)

    image_processor = AutoImageProcessor.from_pretrained(model_id, **trust)
    try:
        from transformers.models.florence2.processing_florence2 import Florence2Processor
    except ImportError:
        try:
            from transformers import Florence2Processor
        except ImportError as import_exc:
            raise RuntimeError(
                f"Failed to load Florence-2 processor for {model_id}"
            ) from import_exc

    processor = Florence2Processor(image_processor=image_processor, tokenizer=slow_tokenizer)
    _ensure_instance_additional_special_tokens(processor.tokenizer)
    return processor


def _load_florence(config: Config):
    global _florence_model, _florence_processor
    if _florence_model is not None:
        return _florence_model, _florence_processor

    from transformers import Florence2ForConditionalGeneration

    _patch_tokenizer_additional_special_tokens()

    model_id = config.florence.model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = _load_florence_processor(model_id)
    model = Florence2ForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    _florence_model, _florence_processor = model, processor
    return model, processor


def unload_florence() -> None:
    global _florence_model, _florence_processor
    _florence_model = _florence_processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_caption(
    image: Image.Image, config: Config, task: str = "<MORE_DETAILED_CAPTION>"
) -> str:
    model, processor = _load_florence(config)
    device = next(model.parameters()).device
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            num_beams=3,
        )
    caption = processor.batch_decode(generated, skip_special_tokens=False)[0]
    for token in ("<MORE_DETAILED_CAPTION>", "<CAPTION>", "<DETAILED_CAPTION>"):
        caption = caption.replace(token, "").strip()
    return caption


def generate_clinical_vqa(
    image: Image.Image,
    questions: list[str],
    config: Config,
) -> list[str]:
    """Answer clinical VQA prompts using Florence-2."""
    answers: list[str] = []
    for question in questions:
        task = f"<VQA>{question}"
        answers.append(generate_caption(image, config, task=task))
    return answers


def generate_diagnostic_text(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    config: Config,
) -> str:
    class_names = config.get_class_names()
    class_context = ""
    if predicted_class in class_names:
        class_context = (
            f" (class index {class_names.index(predicted_class) + 1}/{len(class_names)})"
        )

    caption = generate_caption(image, config)
    disclaimer = (
        "DISCLAIMER: This AI-generated report is for research and interpretability "
        "purposes only. It is not a substitute for professional medical diagnosis."
    )
    return (
        f"Predicted diagnosis: {predicted_class}{class_context} "
        f"(confidence: {confidence:.1%})\n\n"
        f"Visual description: {caption}\n\n{disclaimer}"
    )
