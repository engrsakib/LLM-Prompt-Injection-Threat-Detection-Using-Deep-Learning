"""Grad-CAM, attention rollout, and SAM-constrained XAI overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from neuro_mri_xai.config import Config, load_config
from neuro_mri_xai.dataset import get_transforms
from neuro_mri_xai.models import build_model
from neuro_mri_xai.models.sam_roi import extract_brain_mask, overlay_heatmap_on_mask, unload_sam
from neuro_mri_xai.utils.paths import ensure_dir


def _find_gradcam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    inner = model.base_model if hasattr(model, "base_model") else model
    if hasattr(inner, "model"):
        inner = inner.model
    if hasattr(inner, "layers") and len(inner.layers) > 0:
        last_stage = inner.layers[-1]
        if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
            return last_stage.blocks[-1].norm1
    for _, module in reversed(list(inner.named_modules())):
        if "norm" in type(module).__name__.lower():
            return module
    raise RuntimeError("Could not find Grad-CAM target layer")


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.target_layer = _find_gradcam_target_layer(model)
        self.activations = self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd(_m, _i, output):
            self.activations = output.detach()
        def bwd(_m, _gi, grad_output):
            self.gradients = grad_output[0].detach()
        self._hooks.append(self.target_layer.register_forward_hook(fwd))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        output[0, class_idx].backward(retain_graph=True)
        grads, acts = self.gradients, self.activations
        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks failed")
        if acts.dim() == 3:
            b, tokens, c = acts.shape
            side = int(tokens**0.5)
            if side * side == tokens:
                acts = acts.view(b, side, side, c).permute(0, 3, 1, 2)
                grads = grads.view(b, side, side, c).permute(0, 3, 1, 2)
        weights = grads.mean(dim=(2, 3), keepdim=True) if grads.dim() == 4 else grads.mean(dim=1, keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def compute_attention_rollout(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    x = input_tensor.clone().requires_grad_(True)
    output = model(x)
    class_idx = int(output.argmax(dim=1).item())
    model.zero_grad(set_to_none=True)
    output[0, class_idx].backward()
    saliency = x.grad.abs().sum(dim=1).squeeze().detach().cpu().numpy()
    return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)


def display_gradcam(
    image_path: str | Path,
    heatmap: np.ndarray,
    output_path: str | Path | None = None,
    alpha: float = 0.4,
    title: str = "Grad-CAM Medical Explanation",
) -> Path | None:
    img = np.array(Image.open(image_path).convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    jet = cm.get_cmap("jet")
    colored = jet(heatmap_resized)[:, :, :3]
    overlay = (alpha * colored + (1 - alpha) * img / 255.0).clip(0, 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(title)
    plt.tight_layout()
    if output_path:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path
    plt.show()
    plt.close(fig)
    return None


def load_checkpoint_model(checkpoint_path: str | Path, config: Config) -> tuple[torch.nn.Module, list[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names", config.classes)
    config.model.num_classes = len(class_names)
    model = build_model(config, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    return model.to(device).eval(), class_names


def explain_sample(
    model: torch.nn.Module,
    image_path: str | Path,
    config: Config,
    class_names: list[str],
    output_dir: str | Path | None = None,
) -> dict:
    device = next(model.parameters()).device
    image_path = Path(image_path)
    pil_image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_image)
    tensor = get_transforms(config.dataset.image_size, train=False)(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())

    gradcam = GradCAM(model)
    heatmap = gradcam(tensor, class_idx)
    gradcam.remove_hooks()
    attention = compute_attention_rollout(model, tensor)

    result = {
        "prediction": class_names[class_idx],
        "confidence": confidence,
        "gradcam_path": None,
        "attention_path": None,
        "sam_overlay_path": None,
    }
    if output_dir:
        out = ensure_dir(output_dir)
        stem = image_path.stem
        gradcam_path = out / f"{stem}_gradcam.png"
        attention_path = out / f"{stem}_attention.png"
        display_gradcam(image_path, heatmap, gradcam_path, alpha=config.explainability.alpha)
        display_gradcam(image_path, attention, attention_path, alpha=config.explainability.alpha, title="Attention Saliency")
        result["gradcam_path"] = str(gradcam_path)
        result["attention_path"] = str(attention_path)
        if config.sam.enabled:
            mask = extract_brain_mask(image_rgb, config)
            sam_path = out / f"{stem}_sam_overlay.png"
            Image.fromarray(overlay_heatmap_on_mask(image_rgb, heatmap, mask, config.explainability.alpha)).save(sam_path)
            result["sam_overlay_path"] = str(sam_path)
            unload_sam()
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate XAI visualizations")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    model, class_names = load_checkpoint_model(args.checkpoint, config)
    result = explain_sample(model, args.image, config, class_names, args.output_dir or config.explainability.figures_dir)
    print(result)


if __name__ == "__main__":
    main()
