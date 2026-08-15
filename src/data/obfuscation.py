"""Obfuscation-aware text normalization for prompt-injection samples."""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field

from src.data.cleaning import CleaningConfig, clean_text

_BASE64_PATTERN = re.compile(
    r"(?:[A-Za-z0-9+/]{4}){2,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"
)
_HEX_PATTERN = re.compile(r"\b(?:0x)?([0-9a-fA-F]{8,})\b")
_LEET_PATTERN = re.compile(r"(?i)(1|3|4|5|7|0)(?=.*[a-z])")
_URL_ENCODING_PATTERN = re.compile(r"(?:%[0-9a-fA-F]{2})+")
_MIXED_SCRIPT_PATTERN = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u4E00-\u9FFF\u0590-\u05FF]"
)
_REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")


@dataclass
class ObfuscationProfile:
    detected: bool = False
    types: list[str] = field(default_factory=list)
    decoded_fragments: list[str] = field(default_factory=list)


def _try_base64_decode(fragment: str) -> str | None:
    padded = fragment + "=" * (-len(fragment) % 4)
    try:
        decoded = base64.b64decode(padded, validate=True)
        text = decoded.decode("utf-8")
        if text.isprintable() or " " in text:
            return text
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    return None


def detect_obfuscation(text: str) -> ObfuscationProfile:
    """Detect common obfuscation patterns without altering the text."""
    profile = ObfuscationProfile()
    if not text:
        return profile

    if _BASE64_PATTERN.search(text):
        profile.types.append("base64")
    if _HEX_PATTERN.search(text):
        profile.types.append("hex")
    if _URL_ENCODING_PATTERN.search(text):
        profile.types.append("url_encoding")
    if _LEET_PATTERN.search(text):
        profile.types.append("leetspeak")
    if _MIXED_SCRIPT_PATTERN.search(text):
        profile.types.append("mixed_script")
    if _REPEATED_CHAR_PATTERN.search(text):
        profile.types.append("char_repetition")

    profile.detected = bool(profile.types)
    return profile


def obfuscation_aware_clean(
    text: str,
    intent_label: int | None = None,
    base_config: CleaningConfig | None = None,
) -> tuple[str, ObfuscationProfile]:
    """
    Clean text while preserving attack signal for obfuscated prompts.

    Rules:
    - Never lowercase obfuscation-class samples or detected obfuscation
    - Decode base64 fragments only when valid and append as metadata-friendly suffix
    - Remove zero-width/control chars but keep encoded payload structure
    """
    profile = detect_obfuscation(text)
    is_obfuscation_class = intent_label == 4

    config = base_config or CleaningConfig(
        lowercase=False,
        normalize_unicode=True,
        remove_control_chars=True,
        remove_zero_width=True,
        collapse_whitespace=True,
        strip_text=True,
    )

    # Preserve case when obfuscation is likely part of the attack signal.
    if is_obfuscation_class or profile.detected:
        config = CleaningConfig(
            lowercase=False,
            normalize_unicode=config.normalize_unicode,
            remove_control_chars=config.remove_control_chars,
            remove_zero_width=config.remove_zero_width,
            collapse_whitespace=config.collapse_whitespace,
            strip_text=config.strip_text,
        )

    cleaned = clean_text(text, config)

    if profile.detected and "base64" in profile.types:
        for match in _BASE64_PATTERN.findall(text):
            if len(match) < 8:
                continue
            decoded = _try_base64_decode(match)
            if decoded:
                profile.decoded_fragments.append(decoded)

    # Attach decoded context without replacing original payload (preserves signal).
    if profile.decoded_fragments and (is_obfuscation_class or profile.detected):
        unique_fragments = list(dict.fromkeys(profile.decoded_fragments))
        cleaned = f"{cleaned} [decoded_context: {' | '.join(unique_fragments[:2])}]"

    return cleaned, profile


def obfuscation_features(text: str, intent_label: int | None = None) -> dict:
    """Return obfuscation metadata columns for dataframe export."""
    cleaned, profile = obfuscation_aware_clean(text, intent_label=intent_label)
    return {
        "text_obfuscation_aware": cleaned,
        "obfuscation_detected": profile.detected,
        "obfuscation_types": ",".join(profile.types),
        "obfuscation_decoded_count": len(profile.decoded_fragments),
    }
