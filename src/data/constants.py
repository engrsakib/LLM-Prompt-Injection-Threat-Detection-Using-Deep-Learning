"""Schema constants for the Threat Matrix dataset."""

from __future__ import annotations

INTENT_LABELS: dict[int, str] = {
    0: "benign",
    1: "direct_injection",
    2: "system_extraction",
    3: "role_hijack",
    4: "obfuscation",
    5: "tool_abuse",
    6: "indirect_injection",
}

INTENT_TO_LABEL: dict[str, int] = {v: k for k, v in INTENT_LABELS.items()}

BINARY_BENIGN = 0
BINARY_MALICIOUS = 1

# Severity range measured on the local snapshot (data/raw/*.jsonl): values are {1, 2, 3}.
# Earlier documentation claimed 1-10; that was not supported by the data.
MIN_SEVERITY = 1
MAX_SEVERITY = 3

# Ordered label list used by every classifier head (index == intent_label).
LABEL_ORDER: tuple[str, ...] = tuple(INTENT_LABELS[i] for i in range(len(INTENT_LABELS)))
NUM_CLASSES = len(LABEL_ORDER)
ATTACK_LABELS: tuple[int, ...] = tuple(i for i in INTENT_LABELS if i != 0)

REQUIRED_FIELDS = (
    "text",
    "label",
    "binary_label",
    "intent",
    "intent_label",
    "technique",
    "severity",
)

OPTIONAL_FIELDS = (
    "technique_label",
    "surface",
    "surface_label",
    "source",
    "ambiguity",
)

SPLITS = ("train", "validation", "test")

DATASET_NAME = "neuralchemy/prompt-injection-Threat-Matrix"
DATASET_LICENSE = "CC BY-NC 4.0"
DATASET_URL = "https://huggingface.co/datasets/neuralchemy/prompt-injection-Threat-Matrix"
