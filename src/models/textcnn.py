"""Kim (2014) style TextCNN with padding-aware max pooling."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.data.constants import NUM_CLASSES
from src.models.base import BaseClassifier
from src.models.vocab import PAD_ID, Vocab

CONFIG_FILENAME = "model_config.json"
WEIGHTS_FILENAME = "model.pt"


class TextCNN(BaseClassifier):
    kind = "textcnn"

    def __init__(
        self,
        vocab: Vocab | None = None,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if vocab is None and vocab_size is None:
            raise ValueError("Provide either a Vocab or vocab_size")
        self.vocab = vocab
        size = len(vocab) if vocab is not None else int(vocab_size)
        self.config = {
            "vocab_size": size,
            "embed_dim": embed_dim,
            "num_classes": num_classes,
            "kernel_sizes": list(kernel_sizes),
            "num_filters": num_filters,
            "dropout": dropout,
        }
        self.num_labels = num_classes
        self.embedding = nn.Embedding(size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != PAD_ID).long()
        emb = self.embedding(input_ids)  # [N, T, E]
        emb = emb * attention_mask.unsqueeze(-1).to(emb.dtype)
        x = emb.transpose(1, 2)  # [N, E, T]
        pooled = []
        for conv in self.convs:
            feat = torch.relu(conv(x))  # [N, F, T']
            feat = feat[:, :, : input_ids.shape[1]]
            mask = attention_mask[:, : feat.shape[2]].unsqueeze(1).bool()
            feat = feat.masked_fill(~mask, float("-inf"))
            pooled.append(feat.max(dim=2).values)
        out = torch.cat(pooled, dim=1)
        out = torch.nan_to_num(out, neginf=0.0)
        return self.fc(self.dropout(out))

    def get_encoder(self) -> Vocab:
        if self.vocab is None:
            raise RuntimeError("TextCNN was built without a Vocab; cannot encode text")
        return self.vocab

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_FILENAME).write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        torch.save(self.state_dict(), directory / WEIGHTS_FILENAME)
        if self.vocab is not None:
            self.vocab.save(directory)
        self.write_meta(directory)

    @classmethod
    def load(cls, directory: str | Path, map_location: str = "cpu") -> "TextCNN":
        directory = Path(directory)
        cfg = json.loads((directory / CONFIG_FILENAME).read_text(encoding="utf-8"))
        vocab = Vocab.load(directory)
        model = cls(
            vocab=vocab,
            embed_dim=cfg["embed_dim"],
            num_classes=cfg["num_classes"],
            kernel_sizes=tuple(cfg["kernel_sizes"]),
            num_filters=cfg["num_filters"],
            dropout=cfg["dropout"],
        )
        model.load_state_dict(torch.load(directory / WEIGHTS_FILENAME, map_location=map_location))
        return model
