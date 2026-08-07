# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""CLI entry point for XAI visualizations."""

from __future__ import annotations

import argparse

from neuro_mri_xai.config import load_config
from neuro_mri_xai.evaluation.checkpoint import load_checkpoint_model
from neuro_mri_xai.explainability.pipeline import explain_sample


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate XAI visualizations")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    model, class_names = load_checkpoint_model(args.checkpoint, config)
    result = explain_sample(
        model, args.image, config, class_names,
        args.output_dir or config.explainability.figures_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
