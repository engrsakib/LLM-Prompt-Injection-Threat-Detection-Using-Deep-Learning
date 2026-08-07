#!/usr/bin/env python3
"""Map changed source files to associated pytest modules."""

from __future__ import annotations

import sys
from pathlib import Path

# Explicit source file -> test file mapping
FILE_TO_TEST: dict[str, str] = {
    "dataset.py": "tests/test_dataset.py",
    "paths.py": "tests/test_paths.py",
    "config.py": "tests/test_paths.py",
}

# Source names that share model/integration tests
MODEL_RELATED = {
    "train.py",
    "evaluate.py",
    "explainability.py",
    "report.py",
    "lora.py",
    "sam_roi.py",
    "swin_classifier.py",
    "florence_reporter.py",
    "__init__.py",
    "seed.py",
    "plotting.py",
}


def select_tests(changed_files: list[str]) -> list[str]:
    selected: set[str] = set()
    run_all = False

    for raw in changed_files:
        path = Path(raw.replace("\\", "/"))
        if path.suffix != ".py":
            continue

        parts = path.parts
        if parts[0] == "tests":
            selected.add(str(path).replace("\\", "/"))
            continue

        if parts[0] == "scripts":
            selected.add("tests/test_paths.py")
            continue

        if parts[0] != "src":
            continue

        name = path.name
        if name in FILE_TO_TEST:
            selected.add(FILE_TO_TEST[name])
        elif "models" in parts or name in MODEL_RELATED:
            selected.add("tests/test_models.py")
        else:
            run_all = True

    if run_all or (not selected and changed_files):
        return ["tests"]

    return sorted(selected)


def main() -> None:
    changed = [a for a in sys.argv[1:] if a.strip()]
    tests = select_tests(changed)
    print(" ".join(tests))


if __name__ == "__main__":
    main()
