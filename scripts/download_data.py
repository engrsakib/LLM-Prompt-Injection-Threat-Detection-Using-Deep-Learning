#!/usr/bin/env python3
"""Download and prepare Threat Matrix dataset."""

from src.data.pipeline import PipelineConfig, run_prepare_pipeline

if __name__ == "__main__":
    config = PipelineConfig.from_yaml("configs/data.yaml")
    report = run_prepare_pipeline(config)
    print(f"Done. Snapshot ID: {report['snapshot']['snapshot_id']}")
