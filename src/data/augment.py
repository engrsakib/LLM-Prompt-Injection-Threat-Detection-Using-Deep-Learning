"""Data augmentation helpers (Phase 2 — not yet implemented)."""

from __future__ import annotations

import random


def synonym_replace(text: str, n: int = 1) -> str:
    """Placeholder for synonym replacement augmentation."""
    return text


def token_noise(text: str, p: float = 0.05) -> str:
    """Apply simple character-level noise for robustness testing."""
    chars = list(text)
    for index in range(len(chars)):
        if random.random() < p:
            chars[index] = ""
    return "".join(chars)


def paraphrase(text: str) -> str:
    """Placeholder for T5/Pegasus paraphrase augmentation."""
    return text
