#!/usr/bin/env python3
"""Verify GPLv3 copyright header on modified Python source files."""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_LINES = [
    "Copyright (C) 2026 Md. Nazmus Sakib",
    "GNU General Public License",
    "free software: you can redistribute it and/or modify",
]

CHECK_PREFIXES = ("src/", "scripts/")


def needs_header(path: Path) -> bool:
    normalized = str(path).replace("\\", "/")
    return normalized.startswith(CHECK_PREFIXES) and path.suffix == ".py"


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line for line in REQUIRED_LINES if line not in text]


def main() -> None:
    failures: list[str] = []
    for raw in sys.argv[1:]:
        path = Path(raw)
        if not path.exists() or not needs_header(path):
            continue
        missing = check_file(path)
        if missing:
            failures.append(f"{path}: missing {missing!r}")

    if failures:
        print("License header check failed:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        sys.exit(1)

    print("License header check passed.")


if __name__ == "__main__":
    main()
