"""BiLSTM encoder with additive attention pooling over non-padded positions."""

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


class BiLSTMAttention(BaseClassifier):
    kind = "bilstm_attention"

    def __init__(
        self,
        vocab: Vocab | None = None,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
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
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "dropout": dropout,
        }
        self.num_labels = num_classes
        self.embedding = nn.Embedding(size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != PAD_ID).long()
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        emb = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=input_ids.shape[1])  # [N, T, 2H]
        scores = self.attn(out).squeeze(-1)  # [N, T]
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (out * weights).sum(dim=1)
        return self.fc(self.dropout(context))

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
    def load(cls, directory: str | Path, map_location: str = "cpu") -> "BiLSTMAttention":
        directory = Path(directory)
        cfg = json.loads((directory / CONFIG_FILENAME).read_text(encoding="utf-8"))
        vocab = Vocab.load(directory)
        model = cls(
            vocab=vocab,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_classes=cfg["num_classes"],
            dropout=cfg["dropout"],
        )
        model.load_state_dict(torch.load(directory / WEIGHTS_FILENAME, map_location=map_location))
        return model
