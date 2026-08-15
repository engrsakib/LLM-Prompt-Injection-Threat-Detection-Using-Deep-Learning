"""Text normalization utilities for prompt-injection samples."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Control chars except common whitespace (tab, newline, carriage return)
_CONTROL_CHAR_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)
_ZERO_WIDTH_PATTERN = re.compile(
    r"[\u200b-\u200d\ufeff\u2060\u180e]"
)
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class CleaningConfig:
    lowercase: bool = False
    normalize_unicode: bool = True
    remove_control_chars: bool = True
    remove_zero_width: bool = True
    collapse_whitespace: bool = True
    strip_text: bool = True


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalization for consistent unicode representation."""
    return unicodedata.normalize("NFKC", text)


def remove_control_characters(text: str) -> str:
    """Remove non-printable control characters."""
    return _CONTROL_CHAR_PATTERN.sub("", text)


def remove_zero_width_characters(text: str) -> str:
    """Remove zero-width and invisible unicode characters."""
    return _ZERO_WIDTH_PATTERN.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Collapse repeated whitespace to a single space."""
    return _MULTI_SPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str, config: CleaningConfig | None = None) -> str:
    """Run the full cleaning pipeline on a single prompt."""
    if config is None:
        config = CleaningConfig()

    if text is None:
        return ""

    cleaned = str(text)

    if config.normalize_unicode:
        cleaned = normalize_unicode(cleaned)

    if config.remove_control_chars:
        cleaned = remove_control_characters(cleaned)

    if config.remove_zero_width:
        cleaned = remove_zero_width_characters(cleaned)

    if config.collapse_whitespace:
        cleaned = collapse_whitespace(cleaned)

    if config.strip_text:
        cleaned = cleaned.strip()

    if config.lowercase:
        cleaned = cleaned.lower()

    return cleaned


def cleaning_fingerprint(text: str) -> str:
    """Normalized fingerprint used for exact duplicate detection."""
    cfg = CleaningConfig(
        lowercase=True,
        normalize_unicode=True,
        remove_control_chars=True,
        remove_zero_width=True,
        collapse_whitespace=True,
        strip_text=True,
    )
    return clean_text(text, cfg)
