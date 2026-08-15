"""CLI entrypoint for Phase-1 data preparation."""

from __future__ import annotations

import argparse
import logging
import sys

from src.data.pipeline import PipelineConfig, run_prepare_pipeline


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Threat Matrix dataset for training (Phase 1)."
    )
    parser.add_argument(
        "--config",
        default="configs/data.yaml",
        help="Path to data pipeline YAML config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    config = PipelineConfig.from_yaml(args.config)
    report = run_prepare_pipeline(config)

    print("Data preparation completed successfully.")
    print(f"Snapshot ID: {report['snapshot']['snapshot_id']}")
    print("Processed splits:")
    for split, count in report["distribution_summary"].items():
        print(f"  - {split}: {count} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
