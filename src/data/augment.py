"""Train-only augmentation pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

_SECURITY_SYNONYMS = {
    "ignore": ["disregard", "override", "bypass"],
    "instruction": ["directive", "command", "prompt"],
    "system": ["internal", "core", "hidden"],
    "prompt": ["instruction", "query", "input"],
    "reveal": ["expose", "leak", "disclose"],
    "secret": ["hidden", "confidential", "private"],
    "admin": ["root", "superuser", "privileged"],
}


@dataclass
class AugmentConfig:
    enabled: bool = True
    synonym_replace_prob: float = 0.25
    token_noise_prob: float = 0.10
    word_swap_prob: float = 0.15
    max_augmentations_per_row: int = 1
    augment_minority_only: bool = True
    minority_threshold_ratio: float = 0.75


def synonym_replace(text: str, n: int = 1, rng: random.Random | None = None) -> str:
    """Replace up to n security-related terms with synonyms."""
    rng = rng or random.Random()
    words = text.split()
    replaceable = [
        i
        for i, word in enumerate(words)
        if word.lower().strip(".,!?") in _SECURITY_SYNONYMS
    ]
    if not replaceable:
        return text

    chosen = rng.sample(replaceable, k=min(n, len(replaceable)))
    for idx in chosen:
        token = words[idx].lower().strip(".,!?")
        synonym = rng.choice(_SECURITY_SYNONYMS[token])
        words[idx] = synonym
    return " ".join(words)


def token_noise(text: str, p: float = 0.05, rng: random.Random | None = None) -> str:
    """Apply light character-level noise for robustness."""
    rng = rng or random.Random()
    chars = list(text)
    for idx in range(len(chars)):
        if rng.random() < p and chars[idx].isalpha():
            chars[idx] = chars[idx] + chars[idx]
    return "".join(chars)


def random_word_swap(text: str, rng: random.Random | None = None) -> str:
    """Swap two adjacent words to simulate paraphrase variation."""
    rng = rng or random.Random()
    words = text.split()
    if len(words) < 3:
        return text
    idx = rng.randint(0, len(words) - 2)
    words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return " ".join(words)


def paraphrase(text: str, rng: random.Random | None = None) -> str:
    """Rule-based paraphrase combining synonym replace and word swap."""
    rng = rng or random.Random()
    out = synonym_replace(text, n=1, rng=rng)
    if rng.random() < 0.5:
        out = random_word_swap(out, rng=rng)
    return out


def _is_minority(row: pd.Series, class_counts: dict[int, int]) -> bool:
    label = int(row["intent_label"])
    if not class_counts:
        return False
    max_count = max(class_counts.values())
    return class_counts.get(label, 0) < max_count * 0.75


def augment_row(
    row: pd.Series,
    config: AugmentConfig,
    rng: random.Random,
    class_counts: dict[int, int],
) -> list[dict]:
    """Generate augmented variants for one training row."""
    if not config.enabled:
        return []

    if config.augment_minority_only and not _is_minority(row, class_counts):
        return []

    text = row.get("text_obfuscation_aware", row.get("text_clean", row.get("text", "")))
    variants: list[dict] = []

    for _ in range(config.max_augmentations_per_row):
        augmented_text = text
        method = rng.choice(["synonym", "noise", "paraphrase"])

        if method == "synonym" and rng.random() < config.synonym_replace_prob:
            augmented_text = synonym_replace(augmented_text, n=1, rng=rng)
        elif method == "noise" and rng.random() < config.token_noise_prob:
            augmented_text = token_noise(augmented_text, p=0.04, rng=rng)
        elif method == "paraphrase":
            augmented_text = paraphrase(augmented_text, rng=rng)

        if augmented_text != text:
            new_row = row.to_dict()
            new_row["text_original_aug_source"] = text
            new_row["text"] = augmented_text
            new_row["text_clean"] = augmented_text
            new_row["text_obfuscation_aware"] = augmented_text
            new_row["is_augmented"] = True
            new_row["augment_method"] = method
            variants.append(new_row)

    return variants


def augment_training_frame(
    df: pd.DataFrame,
    config: AugmentConfig,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Apply train-only augmentation and return combined dataframe."""
    rng = random.Random(seed)
    class_counts = {
        int(k): int(v)
        for k, v in df["intent_label"].astype(int).value_counts().to_dict().items()
    }

    base = df.copy()
    base["is_augmented"] = False
    base["augment_method"] = "none"

    generated: list[dict] = []
    for _, row in base.iterrows():
        generated.extend(augment_row(row, config, rng, class_counts))

    if generated:
        aug_df = pd.DataFrame(generated)
        combined = pd.concat([base, aug_df], ignore_index=True)
    else:
        combined = base

    report = {
        "enabled": config.enabled,
        "input_rows": int(len(df)),
        "generated_rows": int(len(generated)),
        "output_rows": int(len(combined)),
        "methods": pd.Series([r.get("augment_method") for r in generated]).value_counts().to_dict()
        if generated
        else {},
    }
    return combined, report
