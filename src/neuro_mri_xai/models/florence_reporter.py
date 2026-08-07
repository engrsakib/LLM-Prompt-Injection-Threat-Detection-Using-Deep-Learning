"""Florence-2 natural language captioning for MRI reports."""

from __future__ import annotations

import torch
from PIL import Image

from neuro_mri_xai.config import Config

_florence_model = None
_florence_processor = None


def _load_florence(config: Config):
    global _florence_model, _florence_processor
    if _florence_model is not None:
        return _florence_model, _florence_processor

    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(config.florence.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.florence.model_id, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)
    model.eval()
    _florence_model, _florence_processor = model, processor
    return model, processor


def unload_florence() -> None:
    global _florence_model, _florence_processor
    _florence_model = _florence_processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_caption(image: Image.Image, config: Config, task: str = "<MORE_DETAILED_CAPTION>") -> str:
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


def generate_diagnostic_text(
    image: Image.Image,
    predicted_class: str,
    confidence: float,
    config: Config,
) -> str:
    caption = generate_caption(image, config)
    disclaimer = (
        "DISCLAIMER: This AI-generated report is for research and interpretability "
        "purposes only. It is not a substitute for professional medical diagnosis."
    )
    return (
        f"Predicted diagnosis: {predicted_class} (confidence: {confidence:.1%})\n\n"
        f"Visual description: {caption}\n\n{disclaimer}"
    )
