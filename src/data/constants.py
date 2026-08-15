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

MIN_SEVERITY = 1
MAX_SEVERITY = 10

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
