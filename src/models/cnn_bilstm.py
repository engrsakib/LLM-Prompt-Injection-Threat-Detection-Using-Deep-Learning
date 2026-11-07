"""Hybrid CNN -> BiLSTM classifier (local n-gram features followed by sequence modelling)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.constants import NUM_CLASSES
from src.models.base import BaseClassifier
from src.models.vocab import PAD_ID, Vocab

CONFIG_FILENAME = "model_config.json"
WEIGHTS_FILENAME = "model.pt"


class CNNBiLSTM(BaseClassifier):
    kind = "cnn_bilstm"

    def __init__(
        self,
        vocab: Vocab | None = None,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        conv_channels: int = 128,
        kernel_size: int = 3,
        hidden_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if vocab is None and vocab_size is None:
            raise ValueError("Provide either a Vocab or vocab_size")
        self.vocab = vocab
        size = len(vocab) if vocab is not None else int(vocab_size)
        self.config = {
            "vocab_size": size,
            "embed_dim": embed_dim,
            "conv_channels": conv_channels,
            "kernel_size": kernel_size,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout": dropout,
        }
        self.num_labels = num_classes
        self.embedding = nn.Embedding(size, embed_dim, padding_idx=PAD_ID)
        self.conv = nn.Conv1d(embed_dim, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lstm = nn.LSTM(conv_channels, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden_dim, num_classes)  # mean-pool + max-pool

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != PAD_ID).long()
        mask = attention_mask.bool()
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        emb = self.dropout(self.embedding(input_ids))  # [N, T, E]
        conv = torch.relu(self.conv(emb.transpose(1, 2)))[:, :, : input_ids.shape[1]]  # [N, C, T]
        conv = conv.transpose(1, 2) * mask.unsqueeze(-1).to(conv.dtype)
        packed = pack_padded_sequence(conv, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=input_ids.shape[1])  # [N, T, 2H]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(out.dtype)
        mean_pool = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
        max_pool = out.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)
        return self.fc(self.dropout(torch.cat([mean_pool, max_pool], dim=1)))

    def get_encoder(self) -> Vocab:
        if self.vocab is None:
            raise RuntimeError("Model built without a Vocab")
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
    def load(cls, directory: str | Path, map_location: str = "cpu") -> "CNNBiLSTM":
        directory = Path(directory)
        cfg = json.loads((directory / CONFIG_FILENAME).read_text(encoding="utf-8"))
        vocab = Vocab.load(directory)
        model = cls(
            vocab=vocab,
            embed_dim=cfg["embed_dim"],
            conv_channels=cfg["conv_channels"],
            kernel_size=cfg["kernel_size"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
        )
        model.load_state_dict(torch.load(directory / WEIGHTS_FILENAME, map_location=map_location))
        return model
