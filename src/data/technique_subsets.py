"""Technique-specific subset definitions for Phase 3."""

from __future__ import annotations

# Canonical subset keys used in exports and reports.
TECHNIQUE_SUBSETS: dict[str, dict] = {
    "encoding": {
        "description": "Encoding / obfuscation attacks (Base64, hex, unicode tricks)",
        "technique_keywords": (
            "encoding",
            "obfuscation",
            "base64",
            "hex",
            "unicode",
            "cipher",
            "encoded",
        ),
        "intent_labels": (4,),  # obfuscation
    },
    "role_play": {
        "description": "Role-play / persona hijacking attacks",
        "technique_keywords": (
            "role",
            "persona",
            "character",
            "roleplay",
            "role-play",
            "role_play",
            "jailbreak",
        ),
        "intent_labels": (3,),  # role_hijack
    },
    "tool_abuse": {
        "description": "Malicious tool / function-call abuse",
        "technique_keywords": (
            "tool",
            "function",
            "api",
            "call",
            "tool_abuse",
            "tool-abuse",
            "plugin",
        ),
        "intent_labels": (5,),  # tool_abuse
    },
}
