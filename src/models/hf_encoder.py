"""Generic Hugging Face encoder classifier (BERT family, XLNet, ModernBERT, ...)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from src.data.constants import NUM_CLASSES
from src.models.base import BaseClassifier, EncodedBatch

HF_META_FILENAME = "hf_classifier.json"


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Make sure the tokenizer has a pad token. Returns True if one was added."""
    if tokenizer.pad_token_id is not None:
        return False
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return True


class HFTokenizerEncoder:
    """Adapts a Hugging Face tokenizer to the :class:`BatchEncoder` protocol."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        ensure_pad_token(tokenizer)

    def encode_batch(self, texts: list[str], max_length: int) -> EncodedBatch:
        # verbose=False silences the "> model_max_length" warning; truncation is applied below.
        raw = self.tokenizer(texts, truncation=False, padding=False, add_special_tokens=True, verbose=False)
        lengths = torch.tensor([len(ids) for ids in raw["input_ids"]], dtype=torch.long)
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        return EncodedBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            truncated=lengths > max_length,
        )

    def save(self, directory: Path) -> None:
        self.tokenizer.save_pretrained(str(directory))


class HFEncoderClassifier(BaseClassifier):
    kind = "hf_encoder"

    def __init__(
        self,
        checkpoint: str,
        num_labels: int = NUM_CLASSES,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        token: str | None = None,
        torch_dtype: torch.dtype | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.revision = revision
        self.num_labels = num_labels
        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "token": token,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **load_kwargs)
        added = ensure_pad_token(self.tokenizer)
        kwargs = dict(load_kwargs)
        if torch_dtype is not None:
            kwargs["dtype"] = torch_dtype
        if model_kwargs:
            kwargs.update(model_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            **kwargs,
        )
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.id2label = {i: name for i, name in enumerate(self.label_order)}
        self.model.config.label2id = {name: i for i, name in enumerate(self.label_order)}
        self._encoder = HFTokenizerEncoder(self.tokenizer)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def get_encoder(self) -> HFTokenizerEncoder:
        return self._encoder

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory))
        self.tokenizer.save_pretrained(str(directory))
        (directory / HF_META_FILENAME).write_text(
            json.dumps({"checkpoint": self.checkpoint, "revision": self.revision}, indent=2),
            encoding="utf-8",
        )
        self.write_meta(directory, {"checkpoint": self.checkpoint, "revision": self.revision})

    @classmethod
    def load(cls, directory: str | Path, **kwargs: Any) -> "HFEncoderClassifier":
        directory = Path(directory)
        meta = json.loads((directory / HF_META_FILENAME).read_text(encoding="utf-8"))
        obj = cls(checkpoint=str(directory), local_files_only=True, **kwargs)
        obj.checkpoint = meta.get("checkpoint", str(directory))
        obj.revision = meta.get("revision")
        return obj
