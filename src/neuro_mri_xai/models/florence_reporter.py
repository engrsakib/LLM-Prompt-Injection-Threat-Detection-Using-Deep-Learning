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

FLORENCE_CAPTION_TASKS = frozenset(
    {"<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"}
)
FLORENCE_VQA_PREFIX = "<VQA>"
DEFAULT_FLORENCE_CAPTION_TASK = "<MORE_DETAILED_CAPTION>"
FLORENCE_IMAGE_TOKEN = "<image>"
DEFAULT_FLORENCE_IMAGE_SEQ_LENGTH = 577

_florence_model = None
_florence_processor = None
_tokenizer_compat_patched = False


def _normalize_florence_task(task: str) -> str:
    """Ensure a valid Florence-2 task prompt string."""
    normalized = (task or "").strip()
    if not normalized:
        logger.warning("Empty Florence task prompt; using %s", DEFAULT_FLORENCE_CAPTION_TASK)
        return DEFAULT_FLORENCE_CAPTION_TASK

    if normalized.startswith(FLORENCE_VQA_PREFIX):
        return normalized

    if normalized in FLORENCE_CAPTION_TASKS:
        return normalized

    if normalized.startswith("<") and normalized.endswith(">"):
        return normalized

    logger.warning(
        "Unrecognized Florence task %r; using %s",
        normalized,
        DEFAULT_FLORENCE_CAPTION_TASK,
    )
    return DEFAULT_FLORENCE_CAPTION_TASK


def _resolve_image_token_id(processor: object) -> int | None:
    token_id = getattr(processor, "image_token_id", None)
    if isinstance(token_id, int):
        return token_id

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None

    token_id = getattr(tokenizer, "image_token_id", None)
    if isinstance(token_id, int):
        return token_id

    image_token = getattr(processor, "image_token", None) or getattr(tokenizer, "image_token", None)
    if isinstance(image_token, str):
        try:
            return int(tokenizer.convert_tokens_to_ids(image_token))
        except Exception:
            pass

    for candidate in (FLORENCE_IMAGE_TOKEN, "<|image|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(candidate)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if isinstance(token_id, int) and token_id != unk_id:
                return token_id
        except Exception:
            continue
    return None


def _count_florence_image_tokens(processor: object, input_ids: torch.Tensor) -> int:
    """Count image placeholder tokens in tokenized input_ids."""
    image_token_id = _resolve_image_token_id(processor)
    if image_token_id is None:
        return 0
    return int((input_ids == image_token_id).sum().item())


def _sync_florence_processor(processor: object, model: torch.nn.Module) -> None:
    """Align processor image-token metadata with the loaded model."""
    image_processor = getattr(processor, "image_processor", None)
    image_seq_length = getattr(image_processor, "image_seq_length", None)
    if image_seq_length is None:
        image_seq_length = getattr(processor, "image_seq_length", None)

    if image_seq_length is not None:
        processor.image_seq_length = image_seq_length
        if hasattr(processor, "num_image_tokens"):
            processor.num_image_tokens = image_seq_length

    model_config = getattr(model, "config", None)
    if model_config is not None:
        extra_tokens = getattr(model_config, "num_additional_image_tokens", None)
        if extra_tokens is not None and hasattr(processor, "num_additional_image_tokens"):
            processor.num_additional_image_tokens = int(extra_tokens)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        image_token = getattr(tokenizer, "image_token", None)
        if image_token is not None and not hasattr(processor, "image_token"):
            processor.image_token = image_token

        image_token_id = _resolve_image_token_id(processor)
        if image_token_id is not None:
            processor.image_token_id = image_token_id


def _ensure_processor_image_metadata(
    processor: object,
    model: torch.nn.Module | None = None,
) -> None:
    """Ensure processor exposes image token count/id required for multimodal prompts."""
    if model is not None:
        _sync_florence_processor(processor, model)

    tokenizer = getattr(processor, "tokenizer", None)
    if not getattr(processor, "image_token", None):
        token = getattr(tokenizer, "image_token", None) if tokenizer is not None else None
        processor.image_token = token or FLORENCE_IMAGE_TOKEN

    num_image_tokens = getattr(processor, "num_image_tokens", None) or getattr(
        processor, "image_seq_length", None
    )
    if num_image_tokens is None and model is not None:
        model_config = getattr(model, "config", None)
        if model_config is not None:
            num_image_tokens = getattr(model_config, "image_seq_length", None)
    if num_image_tokens is None:
        image_processor = getattr(processor, "image_processor", None)
        num_image_tokens = getattr(image_processor, "image_seq_length", None)
    if num_image_tokens is None:
        num_image_tokens = DEFAULT_FLORENCE_IMAGE_SEQ_LENGTH

    processor.num_image_tokens = int(num_image_tokens)
    processor.image_seq_length = int(num_image_tokens)

    image_token_id = _resolve_image_token_id(processor)
    if image_token_id is not None:
        processor.image_token_id = image_token_id


def _manual_florence_prompt_inputs(
    processor: object,
    task: str,
    image: Image.Image,
    pixel_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | None:
    """Build Florence inputs using native prompt + explicit image placeholder tokens."""
    tokenizer = getattr(processor, "tokenizer", None)
    image_token = getattr(processor, "image_token", FLORENCE_IMAGE_TOKEN)
    num_image_tokens = getattr(processor, "num_image_tokens", None) or getattr(
        processor, "image_seq_length", None
    )
    if tokenizer is None or not isinstance(num_image_tokens, int) or num_image_tokens <= 0:
        return None

    construct_prompts = getattr(processor, "_construct_prompts", None)
    if callable(construct_prompts):
        prompt_strings = construct_prompts([task])
    else:
        prompt_strings = [task]

    bos_token = getattr(tokenizer, "bos_token", "") or ""
    eos_token = getattr(tokenizer, "eos_token", "") or ""
    expanded = [
        image_token * num_image_tokens + bos_token + prompt + eos_token
        for prompt in prompt_strings
    ]
    tokenized = tokenizer(expanded, return_tensors="pt", truncation=False)

    if pixel_values is None:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]

    fallback: dict[str, torch.Tensor] = dict(tokenized)
    if pixel_values is not None:
        fallback["pixel_values"] = pixel_values
    if _count_florence_image_tokens(processor, fallback["input_ids"]) > 0:
        return fallback
    return None


def _build_florence_processor_inputs(
    processor: object,
    task: str,
    image: Image.Image,
    model: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Build processor inputs with valid task prompt and image placeholder tokens."""
    normalized_task = _normalize_florence_task(task)
    _ensure_processor_image_metadata(processor, model)

    def _call_processor(text: str | list[str], images: Image.Image | list[Image.Image]) -> object:
        return processor(
            text=text,
            images=images,
            return_tensors="pt",
            truncation=False,
        )

    raw_inputs = _call_processor(normalized_task, image)
    input_ids = raw_inputs.get("input_ids")
    pixel_values = raw_inputs.get("pixel_values")
    if input_ids is not None and _count_florence_image_tokens(processor, input_ids) > 0:
        return dict(raw_inputs)

    logger.warning(
        "Florence processor returned 0 image tokens for task %r; retrying list form",
        normalized_task,
    )
    raw_inputs = _call_processor([normalized_task], [image])
    input_ids = raw_inputs.get("input_ids")
    pixel_values = raw_inputs.get("pixel_values")
    if input_ids is not None and _count_florence_image_tokens(processor, input_ids) > 0:
        return dict(raw_inputs)

    manual = _manual_florence_prompt_inputs(
        processor,
        normalized_task,
        image,
        pixel_values=pixel_values,
    )
    if manual is not None:
        logger.info("Built Florence inputs via explicit image-token prompt for %r", normalized_task)
        return manual

    return dict(raw_inputs)


def _validate_florence_inputs(inputs: dict[str, torch.Tensor], processor: object) -> None:
    """Verify pixel_values and image placeholder tokens before generation."""
    pixel_values = inputs.get("pixel_values")
    input_ids = inputs.get("input_ids")

    if pixel_values is None:
        raise ValueError("Florence inputs missing pixel_values")
    if pixel_values.ndim != 4:
        raise ValueError(
            f"Expected pixel_values rank 4, got shape {tuple(pixel_values.shape)}"
        )
    if pixel_values.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1 for pixel_values, got {pixel_values.shape[0]}"
        )
    if input_ids is None:
        raise ValueError("Florence inputs missing input_ids")

    image_token_count = _count_florence_image_tokens(processor, input_ids)
    if image_token_count == 0:
        raise ValueError(
            "Florence input_ids contain 0 image tokens; "
            f"input_ids shape={tuple(input_ids.shape)}, "
            f"pixel_values shape={tuple(pixel_values.shape)}"
        )


def _model_compute_dtype(model: torch.nn.Module) -> torch.dtype:
    """Return the floating-point dtype used by model weights."""
    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype):
        return model_dtype
    return next(model.parameters()).dtype


def _prepare_florence_inputs(
    inputs: object,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move Florence inputs to model device and align float tensors with model dtype."""
    compute_dtype = _model_compute_dtype(model)
    prepared: dict[str, torch.Tensor] = {}

    for key, value in dict(inputs).items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point():
            prepared[key] = value.to(device=device, dtype=compute_dtype)
        else:
            prepared[key] = value.to(device=device)

    return prepared


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


def _load_florence_model(model_id: str, dtype: torch.dtype, device: str) -> torch.nn.Module:
    """Load Florence-2 with AutoModelForCausalLM (remote code) and class fallbacks."""
    from transformers import AutoModelForCausalLM

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        logger.info("Loaded Florence-2 weights via AutoModelForCausalLM")
        return model.to(device).eval()
    except Exception as exc:
        logger.warning(
            "AutoModelForCausalLM.from_pretrained failed for %s (%s); trying Florence2ForConditionalGeneration",
            model_id,
            exc,
        )

    try:
        from transformers import Florence2ForConditionalGeneration

        model = Florence2ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        logger.info("Loaded Florence-2 weights via Florence2ForConditionalGeneration")
        return model.to(device).eval()
    except Exception as exc:
        raise RuntimeError(f"Failed to load Florence-2 model from {model_id}") from exc


def _load_florence(config: Config):
    global _florence_model, _florence_processor
    if _florence_model is not None:
        return _florence_model, _florence_processor

    _patch_tokenizer_additional_special_tokens()

    model_id = config.florence.model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = _load_florence_processor(model_id)
    model = _load_florence_model(model_id, dtype=dtype, device=device)
    _ensure_processor_image_metadata(processor, model)

    _florence_model, _florence_processor = model, processor
    return model, processor


def unload_florence() -> None:
    global _florence_model, _florence_processor
    _florence_model = _florence_processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _decode_florence_caption(processor: object, generated: torch.Tensor, task: str) -> str:
    caption = processor.batch_decode(generated, skip_special_tokens=False)[0]
    for token in (
        task,
        DEFAULT_FLORENCE_CAPTION_TASK,
        "<MORE_DETAILED_CAPTION>",
        "<CAPTION>",
        "<DETAILED_CAPTION>",
    ):
        caption = caption.replace(token, "").strip()
    return caption


def _template_diagnostic_text(
    predicted_class: str,
    confidence: float,
    config: Config,
    *,
    caption: str | None = None,
    florence_unavailable: bool = False,
) -> str:
    class_names = config.get_class_names()
    class_context = ""
    if predicted_class in class_names:
        class_context = (
            f" (class index {class_names.index(predicted_class) + 1}/{len(class_names)})"
        )

    disclaimer = (
        "DISCLAIMER: This AI-generated report is for research and interpretability "
        "purposes only. It is not a substitute for professional medical diagnosis."
    )
    if florence_unavailable or not caption:
        visual = (
            "Visual description: Florence-2 caption unavailable; refer to classifier "
            "prediction and explainability figures above."
        )
    else:
        visual = f"Visual description: {caption}"

    return (
        f"Predicted diagnosis: {predicted_class}{class_context} "
        f"(confidence: {confidence:.1%})\n\n"
        f"{visual}\n\n{disclaimer}"
    )


def generate_caption(
    image: Image.Image, config: Config, task: str = "<MORE_DETAILED_CAPTION>"
) -> str | None:
    """Generate a Florence-2 caption, or None when tokenization/generation is unavailable."""
    try:
        model, processor = _load_florence(config)
        device = next(model.parameters()).device
        if image.mode != "RGB":
            image = image.convert("RGB")
        normalized_task = _normalize_florence_task(task)
        raw_inputs = _build_florence_processor_inputs(
            processor,
            normalized_task,
            image,
            model=model,
        )
        _validate_florence_inputs(raw_inputs, processor)
        inputs = _prepare_florence_inputs(raw_inputs, model, device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )
        return _decode_florence_caption(processor, generated, normalized_task)
    except ValueError as exc:
        logger.warning("Florence caption skipped due to invalid inputs (%s)", exc)
        return None
    except Exception as exc:
        logger.warning("Florence caption generation failed (%s)", exc)
        return None


def generate_clinical_vqa(
    image: Image.Image,
    questions: list[str],
    config: Config,
) -> list[str]:
    """Answer clinical VQA prompts using Florence-2."""
    answers: list[str] = []
    for question in questions:
        task = f"{FLORENCE_VQA_PREFIX}{question.strip()}"
        caption = generate_caption(image, config, task=task)
        answers.append(
            caption
            or "Answer unavailable (Florence-2 input validation or generation failed)."
        )
    return answers


def generate_diagnostic_text(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    config: Config,
) -> str:
    caption = generate_caption(image, config)
    if caption is None:
        return _template_diagnostic_text(
            predicted_class,
            confidence,
            config,
            florence_unavailable=True,
        )
    return _template_diagnostic_text(
        predicted_class,
        confidence,
        config,
        caption=caption,
    )
