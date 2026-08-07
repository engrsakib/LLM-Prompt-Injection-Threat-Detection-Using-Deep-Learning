# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Command-line entry point for model training."""

from __future__ import annotations

import argparse

from neuro_mri_xai.config import load_config
from neuro_mri_xai.training.trainer import Trainer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Swin MRI classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs

    trainer = Trainer(config)
    best = trainer.train()
    print(f"Best model: {best} (val_acc={trainer.state.best_val_acc:.4f})")


if __name__ == "__main__":
    main()
