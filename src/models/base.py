"""Shared classification interface used by every backbone in the study.

All models:
- consume a batch dict with ``input_ids`` and ``attention_mask`` (LongTensor [N, T]),
- return logits of shape ``[N, NUM_CLASSES]`` whose column order is ``LABEL_ORDER``
  (index == ``intent_label``: 0 benign ... 6 indirect_injection),
- can be saved to / loaded from a directory.

Tokenisation is handled by a :class:`BatchEncoder` so the training loop never has to
know whether it is dealing with a Hugging Face tokenizer or a train-only vocabulary.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from src.data.constants import LABEL_ORDER, NUM_CLASSES

META_FILENAME = "classifier_meta.json"


@dataclass
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    truncated: torch.Tensor  # bool [N] - sequence exceeded max_length

    def to(self, device: torch.device | str) -> "EncodedBatch":
        return EncodedBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            truncated=self.truncated,
        )


class BatchEncoder(Protocol):
    """Turns raw text into model-ready tensors."""

    def encode_batch(self, texts: list[str], max_length: int) -> EncodedBatch: ...

    def save(self, directory: Path) -> None: ...


class BaseClassifier(nn.Module, ABC):
    """Abstract seven-class classifier."""

    kind: str = "base"
    num_labels: int = NUM_CLASSES
    label_order: tuple[str, ...] = LABEL_ORDER

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return logits ``[N, num_labels]``."""

    @abstractmethod
    def get_encoder(self) -> BatchEncoder:
        """Return the tokenizer / vocab used to encode text for this model."""

    @abstractmethod
    def save(self, directory: str | Path) -> None: ...

    # Shared helpers -----------------------------------------------------
    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float(), dim=-1)

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"params_total": int(total), "params_trainable": int(trainable)}

    def write_meta(self, directory: Path, extra: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "num_labels": self.num_labels,
            "label_order": list(self.label_order),
        }
        payload.update(self.count_parameters())
        if extra:
            payload.update(extra)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / META_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_meta(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / META_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing {META_FILENAME} in {directory}")
    return json.loads(path.read_text(encoding="utf-8"))
