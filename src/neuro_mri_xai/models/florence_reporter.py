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
DEFAULT_FLORENCE_IMAGE_TOKEN_ID = 51289

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


def _resolve_model_image_token_id(model: torch.nn.Module | None) -> int | None:
    if model is None:
        return None
    config = getattr(model, "config", None)
    if config is not None:
        token_id = getattr(config, "image_token_id", None)
        if isinstance(token_id, int):
            return token_id
    return DEFAULT_FLORENCE_IMAGE_TOKEN_ID


def _ensure_image_token_on_tokenizer(tokenizer: object, model: torch.nn.Module | None = None) -> int:
    """Ensure tokenizer/processor expose Florence `<image>` token id (51289 by default)."""
    target_id = _resolve_model_image_token_id(model) or DEFAULT_FLORENCE_IMAGE_TOKEN_ID
    image_token = getattr(tokenizer, "image_token", None) or FLORENCE_IMAGE_TOKEN

    unk_id = getattr(tokenizer, "unk_token_id", None)
    current_id = None
    try:
        current_id = int(tokenizer.convert_tokens_to_ids(image_token))
    except Exception:
        current_id = None

    if current_id is None or (unk_id is not None and current_id == unk_id):
        add_tokens = getattr(tokenizer, "add_special_tokens", None)
        if callable(add_tokens):
            add_tokens({"additional_special_tokens": [image_token]})
        current_id = int(tokenizer.convert_tokens_to_ids(image_token))

    if current_id != target_id:
        # Prefer model config id — Florence-2 expects a fixed placeholder id in input_ids.
        logger.debug(
            "Tokenizer image token id %s differs from model config %s; using model id",
            current_id,
            target_id,
        )
        current_id = target_id

    tokenizer.image_token = image_token
    tokenizer.image_token_id = current_id
    return current_id


def _resolve_image_token_id(processor: object, model: torch.nn.Module | None = None) -> int | None:
    model_id = _resolve_model_image_token_id(model)
    if model_id is not None:
        return model_id

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


def _count_florence_image_tokens(
    processor: object,
    input_ids: torch.Tensor,
    model: torch.nn.Module | None = None,
) -> int:
    """Count image placeholder tokens in tokenized input_ids."""
    image_token_id = _resolve_image_token_id(processor, model)
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

        image_token_id = _resolve_image_token_id(processor, model)
        if image_token_id is not None:
            processor.image_token_id = image_token_id


def _pixel_values_from_image(processor: object, image: Image.Image) -> torch.Tensor:
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("Florence processor missing image_processor")
    return image_processor(image, return_tensors="pt")["pixel_values"]


@torch.no_grad()
def _infer_image_feature_token_count(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
) -> int | None:
    """Infer placeholder count from vision features (matches get_placeholder_mask)."""
    try:
        if hasattr(model, "get_image_features"):
            features = model.get_image_features(pixel_values)
        elif hasattr(model, "model") and hasattr(model.model, "get_image_features"):
            features = model.model.get_image_features(pixel_values)
        elif hasattr(model, "encode_image"):
            features = model.encode_image(pixel_values)
        else:
            return None
        if isinstance(features, tuple):
            features = features[0]
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            tensor = features.pooler_output
        elif hasattr(features, "last_hidden_state"):
            tensor = features.last_hidden_state
        elif isinstance(features, torch.Tensor):
            tensor = features
        else:
            return None
        if tensor.dim() < 2:
            return None
        return int(tensor.shape[0] * tensor.shape[1])
    except Exception as exc:
        logger.debug("Could not infer Florence image feature count (%s)", exc)
        return None


def _resolve_num_image_tokens(
    processor: object,
    model: torch.nn.Module | None,
    pixel_values: torch.Tensor | None,
) -> int:
    """Resolve how many `<image>` placeholders the vision tower expects."""
    if model is not None and pixel_values is not None:
        inferred = _infer_image_feature_token_count(model, pixel_values)
        if inferred is not None and inferred > 0:
            return inferred

    for source in (
        getattr(processor, "num_image_tokens", None),
        getattr(processor, "image_seq_length", None),
        getattr(getattr(processor, "image_processor", None), "image_seq_length", None),
    ):
        if isinstance(source, int) and source > 0:
            return source

    if model is not None:
        model_config = getattr(model, "config", None)
        if model_config is not None:
            seq_len = getattr(model_config, "image_seq_length", None)
            if isinstance(seq_len, int) and seq_len > 0:
                return seq_len

    return DEFAULT_FLORENCE_IMAGE_SEQ_LENGTH


def _ensure_processor_image_metadata(
    processor: object,
    model: torch.nn.Module | None = None,
) -> None:
    """Ensure processor exposes image token count/id required for multimodal prompts."""
    if model is not None:
        _sync_florence_processor(processor, model)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        token_id = _ensure_image_token_on_tokenizer(tokenizer, model)
        processor.image_token = getattr(tokenizer, "image_token", FLORENCE_IMAGE_TOKEN)
        processor.image_token_id = token_id
    elif not getattr(processor, "image_token", None):
        processor.image_token = FLORENCE_IMAGE_TOKEN

    num_image_tokens = _resolve_num_image_tokens(processor, model, pixel_values=None)
    processor.num_image_tokens = int(num_image_tokens)
    processor.image_seq_length = int(num_image_tokens)

    image_token_id = _resolve_image_token_id(processor, model)
    if image_token_id is not None:
        processor.image_token_id = image_token_id


def _construct_florence_prompt_text(processor: object, task: str) -> str:
    construct_prompts = getattr(processor, "_construct_prompts", None)
    if callable(construct_prompts):
        return construct_prompts([task])[0]
    return task


def _build_input_ids_with_image_placeholders(
    processor: object,
    model: torch.nn.Module,
    task: str,
    num_image_tokens: int,
) -> torch.Tensor:
    """Insert ``num_image_tokens`` copies of ``model.config.image_token_id`` before the text prompt."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Florence processor missing tokenizer")

    image_token_id = _ensure_image_token_on_tokenizer(tokenizer, model)
    processor.image_token_id = image_token_id

    prompt_text = _construct_florence_prompt_text(processor, task)
    bos_token = getattr(tokenizer, "bos_token", "") or ""
    eos_token = getattr(tokenizer, "eos_token", "") or ""
    text_ids = tokenizer(
        bos_token + prompt_text + eos_token,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=False,
    )["input_ids"][0]

    prefix = torch.full((num_image_tokens,), image_token_id, dtype=torch.long)
    return torch.cat([prefix, text_ids], dim=0).unsqueeze(0)


def _task_prompt_with_image_marker(task: str) -> str:
    """Prefix task with a single <image> marker when the processor expects it in text."""
    if task.startswith(FLORENCE_IMAGE_TOKEN):
        return task
    return f"{FLORENCE_IMAGE_TOKEN}{task}"


def _inputs_have_image_tokens(
    processor: object,
    inputs: object,
    model: torch.nn.Module | None = None,
) -> bool:
    input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
    if input_ids is None:
        return False
    return _count_florence_image_tokens(processor, input_ids, model) > 0


def _manual_florence_prompt_inputs(
    processor: object,
    task: str,
    image: Image.Image,
    model: torch.nn.Module,
    pixel_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | None:
    """Build Florence inputs with explicit image placeholder token ids in input_ids."""
    if pixel_values is None:
        try:
            pixel_values = _pixel_values_from_image(processor, image)
        except Exception as exc:
            logger.debug("Manual Florence prompt could not build pixel_values (%s)", exc)
            return None

    num_image_tokens = _resolve_num_image_tokens(processor, model, pixel_values)
    processor.num_image_tokens = int(num_image_tokens)
    processor.image_seq_length = int(num_image_tokens)

    try:
        input_ids = _build_input_ids_with_image_placeholders(
            processor,
            model,
            task,
            num_image_tokens,
        )
    except Exception as exc:
        logger.debug("Direct Florence input_ids construction failed (%s)", exc)
        return None

    if _count_florence_image_tokens(processor, input_ids, model) <= 0:
        return None

    return {"input_ids": input_ids, "pixel_values": pixel_values}


def _build_florence_processor_inputs(
    processor: object,
    task: str,
    image: Image.Image,
    model: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    """Build processor inputs with valid task prompt and image placeholder tokens."""
    if model is None:
        raise ValueError("Florence model is required to build multimodal inputs")

    normalized_task = _normalize_florence_task(task)
    _ensure_processor_image_metadata(processor, model)

    def _call_processor(text: str | list[str], images: Image.Image | list[Image.Image]) -> object:
        return processor(
            text=text,
            images=images,
            return_tensors="pt",
            truncation=False,
        )

    pixel_values: torch.Tensor | None = None
    try:
        pixel_values = _pixel_values_from_image(processor, image)
    except Exception as exc:
        logger.debug("Could not precompute pixel_values (%s)", exc)

    # Native Florence2Processor expects task-only text (e.g. "<MORE_DETAILED_CAPTION>").
    for prompt in (normalized_task, [normalized_task]):
        try:
            images_arg: Image.Image | list[Image.Image] = image if isinstance(prompt, str) else [image]
            raw_inputs = _call_processor(prompt, images_arg)
            if pixel_values is None:
                pixel_values = raw_inputs.get("pixel_values")
            if _inputs_have_image_tokens(processor, raw_inputs, model):
                result = dict(raw_inputs)
                if pixel_values is not None and result.get("pixel_values") is None:
                    result["pixel_values"] = pixel_values
                return result
        except Exception as exc:
            logger.debug("Florence processor call failed for %r (%s)", prompt, exc)

    manual = _manual_florence_prompt_inputs(
        processor,
        normalized_task,
        image,
        model,
        pixel_values=pixel_values,
    )
    if manual is not None:
        logger.info(
            "Built Florence inputs via explicit image-token ids for %r",
            normalized_task,
        )
        return manual

    raise ValueError(
        f"Could not construct Florence inputs with image placeholder tokens for task {normalized_task!r}"
    )


def _validate_florence_inputs(
    inputs: dict[str, torch.Tensor],
    processor: object,
    model: torch.nn.Module | None = None,
) -> None:
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

    image_token_count = _count_florence_image_tokens(processor, input_ids, model)
    if image_token_count == 0:
        raise ValueError(
            "Florence input_ids contain 0 image tokens; "
            f"input_ids shape={tuple(input_ids.shape)}, "
            f"pixel_values shape={tuple(pixel_values.shape)}"
        )

    expected_tokens = _resolve_num_image_tokens(processor, model, pixel_values)
    if expected_tokens > 0 and image_token_count != expected_tokens:
        raise ValueError(
            f"Florence image token mismatch: found {image_token_count} placeholder "
            f"tokens in input_ids but vision encoder expects {expected_tokens} to match "
            f"image features"
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


def _remap_florence_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reconcile fine-tuned checkpoint prefixes with Hugging Face Florence-2 modules."""
    if not state_dict:
        return state_dict

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("model.language_model."):
            new_key = "language_model.model." + key[len("model.language_model.") :]
        elif key.startswith("model.vision_tower."):
            new_key = "vision_tower." + key[len("model.vision_tower.") :]
        elif key.startswith("model.image_projection."):
            new_key = "image_projection." + key[len("model.image_projection.") :]
        elif key.startswith("language_model.") and not key.startswith("language_model.model."):
            suffix = key[len("language_model.") :]
            if suffix.startswith(("layers.", "embed_tokens.", "norm.", "lm_head.")):
                new_key = "language_model.model." + suffix
        remapped[new_key] = value
    return remapped


def _load_florence_state_dict_with_remap(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load a state dict after prefix remapping; log missing/unexpected keys."""
    remapped = _remap_florence_state_dict_keys(state_dict)
    load_result = model.load_state_dict(remapped, strict=False)
    missing = getattr(load_result, "missing_keys", None) or []
    unexpected = getattr(load_result, "unexpected_keys", None) or []
    if missing or unexpected:
        logger.warning(
            "Florence state dict load: missing=%d unexpected=%d (after key remap)",
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.debug("Florence missing keys (first 10): %s", missing[:10])
        if unexpected:
            logger.debug("Florence unexpected keys (first 10): %s", unexpected[:10])


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


def template_diagnostic_text(
    predicted_class: str,
    confidence: float,
    config: Config,
    *,
    caption: str | None = None,
    florence_unavailable: bool = False,
) -> str:
    """Rule-based diagnostic report text (Florence-independent fallback)."""
    return _template_diagnostic_text(
        predicted_class,
        confidence,
        config,
        caption=caption,
        florence_unavailable=florence_unavailable,
    )


def _safe_unload_florence_after_failure() -> None:
    try:
        unload_florence()
    except Exception:
        logger.debug("Florence unload after failure raised; ignoring", exc_info=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        _validate_florence_inputs(raw_inputs, processor, model)
        inputs = _prepare_florence_inputs(raw_inputs, model, device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )
        return _decode_florence_caption(processor, generated, normalized_task)
    except Exception as exc:
        logger.warning("Florence caption generation failed (%s); returning no caption", exc)
        _safe_unload_florence_after_failure()
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
    """Generate diagnostic narrative; never raises — falls back to template text on any error."""
    try:
        caption = generate_caption(image, config)
        if caption is None:
            return template_diagnostic_text(
                predicted_class,
                confidence,
                config,
                florence_unavailable=True,
            )
        return template_diagnostic_text(
            predicted_class,
            confidence,
            config,
            caption=caption,
        )
    except Exception as exc:
        logger.warning(
            "Florence diagnostic text generation failed (%s); using template fallback",
            exc,
        )
        _safe_unload_florence_after_failure()
        return template_diagnostic_text(
            predicted_class,
            confidence,
            config,
            florence_unavailable=True,
        )
