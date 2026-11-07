"""Backward-compatible DistilBERT wrapper (thin alias over :class:`HFEncoderClassifier`)."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.hf_encoder import HFEncoderClassifier


class DistilBERTWrapper:
    """Kept for compatibility with earlier notebooks; new code should use the registry."""

    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 7, device: str | None = None) -> None:
        self.classifier = HFEncoderClassifier(checkpoint=model_name, num_labels=num_labels)
        self.tokenizer = self.classifier.tokenizer
        self.model = self.classifier.model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)

    def predict(self, texts: list[str], batch_size: int = 16, max_length: int = 256) -> torch.Tensor:
        encoder = self.classifier.get_encoder()
        logits = []
        self.classifier.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = encoder.encode_batch(texts[start : start + batch_size], max_length).to(self.device)
                logits.append(self.classifier(batch.input_ids, batch.attention_mask).cpu())
        return torch.cat(logits, dim=0)

    def save(self, path: str) -> None:
        self.classifier.save(Path(path))
