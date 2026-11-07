"""Train-only vocabulary for the custom CNN / RNN classifiers.

The vocabulary must be fitted on the training split only; it is saved alongside the
checkpoint so evaluation and reload use exactly the same token ids.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch

from src.models.base import EncodedBatch

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
VOCAB_FILENAME = "vocab.json"

# Keep case and punctuation: obfuscated / encoded attacks rely on them.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text))


class Vocab:
    def __init__(self, itos: list[str] | None = None, lowercase: bool = False) -> None:
        self.lowercase = lowercase
        self.itos: list[str] = itos or [PAD_TOKEN, UNK_TOKEN]
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    # Construction ---------------------------------------------------------
    @classmethod
    def fit(
        cls,
        texts: list[str],
        min_freq: int = 2,
        max_size: int = 50_000,
        lowercase: bool = False,
    ) -> "Vocab":
        counter: Counter[str] = Counter()
        for text in texts:
            toks = tokenize(text.lower() if lowercase else text)
            counter.update(toks)
        itos = [PAD_TOKEN, UNK_TOKEN]
        for tok, freq in counter.most_common():
            if freq < min_freq or len(itos) >= max_size:
                break
            itos.append(tok)
        return cls(itos=itos, lowercase=lowercase)

    def __len__(self) -> int:
        return len(self.itos)

    # Encoding --------------------------------------------------------------
    def encode(self, text: str, max_length: int) -> tuple[list[int], bool]:
        toks = tokenize(text.lower() if self.lowercase else text)
        truncated = len(toks) > max_length
        ids = [self.stoi.get(t, UNK_ID) for t in toks[:max_length]]
        if not ids:
            ids = [UNK_ID]
        return ids, truncated

    def encode_batch(self, texts: list[str], max_length: int) -> EncodedBatch:
        encoded = [self.encode(t, max_length) for t in texts]
        width = max(len(ids) for ids, _ in encoded)
        input_ids = torch.full((len(encoded), width), PAD_ID, dtype=torch.long)
        attention = torch.zeros((len(encoded), width), dtype=torch.long)
        truncated = torch.zeros(len(encoded), dtype=torch.bool)
        for row, (ids, trunc) in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[row, : len(ids)] = 1
            truncated[row] = trunc
        return EncodedBatch(input_ids=input_ids, attention_mask=attention, truncated=truncated)

    # Persistence -----------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"itos": self.itos, "lowercase": self.lowercase}
        (directory / VOCAB_FILENAME).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "Vocab":
        payload = json.loads((Path(directory) / VOCAB_FILENAME).read_text(encoding="utf-8"))
        return cls(itos=payload["itos"], lowercase=bool(payload.get("lowercase", False)))
