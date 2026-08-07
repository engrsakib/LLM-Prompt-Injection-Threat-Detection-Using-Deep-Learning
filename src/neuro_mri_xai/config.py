# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Configuration loading and typed access."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from neuro_mri_xai.data.constants import EXPECTED_CLASS_NAMES, NUM_CLASSES
from neuro_mri_xai.utils.paths import get_data_root, get_project_root, resolve_path


@dataclass
class DatasetConfig:
    source: str = "kagglehub"
    kagglehub_handle: str = "engrsakib02/neurological-disorders-mri-dataset-for-xai"
    kaggle_dataset: str = "engrsakib02/neurological-disorders-mri-dataset-for-xai"
    kaggle_data_subdir: str = "data"
    gdrive_path: str = ""
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 2
    data_dir: Path = field(default_factory=lambda: Path("data"))
    class_names: list[str] = field(default_factory=lambda: list(EXPECTED_CLASS_NAMES))


@dataclass
class ModelConfig:
    backbone: str = "swin_base_patch4_window7_224"
    num_classes: int = 8
    pretrained: bool = True
    drop_path_rate: float = 0.1
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(default_factory=lambda: ["qkv", "proj"])
    lora_modules_to_save: list[str] = field(default_factory=lambda: ["head"])
    lora_dropout: float = 0.1


@dataclass
class SamConfig:
    enabled: bool = True
    checkpoint: str = "sam_vit_b_01ec64.pth"
    model_type: str = "vit_b"
    weights_dir: str = "weights"


@dataclass
class FlorenceConfig:
    enabled: bool = True
    model_id: str = "microsoft/Florence-2-base"
    use_lora: bool = False


@dataclass
class TrainingConfig:
    epochs: int = 20
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 0.01
    early_stopping_patience: int = 5
    checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("outputs/logs"))
    use_amp: bool = True
    use_cosine_scheduler: bool = True


@dataclass
class EvaluationConfig:
    figures_dir: Path = field(default_factory=lambda: Path("outputs/figures"))
    run_sklearn_baselines: bool = True


@dataclass
class ExplainabilityConfig:
    figures_dir: Path = field(default_factory=lambda: Path("outputs/figures"))
    alpha: float = 0.4


@dataclass
class ReportConfig:
    output_dir: Path = field(default_factory=lambda: Path("outputs/reports"))


@dataclass
class VramConfig:
    max_batch_size: int = 16
    sam_on_cpu: bool = False
    sequential_models: bool = True
    empty_cache_between: bool = True


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sam: SamConfig = field(default_factory=SamConfig)
    florence: FlorenceConfig = field(default_factory=FlorenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    vram: VramConfig = field(default_factory=VramConfig)
    classes: list[str] = field(default_factory=list)
    project_root: Path = field(default_factory=get_project_root)

    def sam_checkpoint_path(self) -> Path:
        return resolve_path(Path(self.sam.weights_dir) / self.sam.checkpoint, self.project_root)

    def get_class_names(self) -> list[str]:
        if self.dataset.class_names:
            return list(self.dataset.class_names)
        if self.classes:
            return list(self.classes)
        return list(EXPECTED_CLASS_NAMES)


def _merge_dataclass(cls, data: dict[str, Any]):
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in fields}
    return cls(**filtered)


def load_config(
    config_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Config:
    project_root = get_project_root()
    if config_path is None:
        config_path = project_root / "configs" / "default.yaml"
    config_path = Path(config_path)

    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    cfg = Config(project_root=project_root)
    cfg.dataset = _merge_dataclass(DatasetConfig, raw.get("dataset", {}))
    cfg.model = _merge_dataclass(ModelConfig, raw.get("model", {}))
    cfg.sam = _merge_dataclass(SamConfig, raw.get("sam", {}))
    cfg.florence = _merge_dataclass(FlorenceConfig, raw.get("florence", {}))
    cfg.training = _merge_dataclass(TrainingConfig, raw.get("training", {}))
    cfg.evaluation = _merge_dataclass(EvaluationConfig, raw.get("evaluation", {}))
    cfg.explainability = _merge_dataclass(ExplainabilityConfig, raw.get("explainability", {}))
    cfg.report = _merge_dataclass(ReportConfig, raw.get("report", {}))
    cfg.vram = _merge_dataclass(VramConfig, raw.get("vram", {}))

    dataset_raw = raw.get("dataset", {})
    if dataset_raw.get("class_names"):
        cfg.dataset.class_names = list(dataset_raw["class_names"])
    elif raw.get("classes"):
        cfg.dataset.class_names = list(raw["classes"])
    else:
        cfg.dataset.class_names = list(EXPECTED_CLASS_NAMES)

    cfg.classes = cfg.get_class_names()
    cfg.model.num_classes = NUM_CLASSES

    cfg.dataset.data_dir = get_data_root(raw)
    if data_dir:
        from neuro_mri_xai.utils.cli import apply_data_dir_override

        apply_data_dir_override(cfg, data_dir)
    cfg.training.checkpoint_dir = resolve_path(cfg.training.checkpoint_dir, project_root)
    cfg.training.log_dir = resolve_path(cfg.training.log_dir, project_root)
    cfg.evaluation.figures_dir = resolve_path(cfg.evaluation.figures_dir, project_root)
    cfg.explainability.figures_dir = resolve_path(cfg.explainability.figures_dir, project_root)
    cfg.report.output_dir = resolve_path(cfg.report.output_dir, project_root)

    if os.environ.get("NEURO_MRI_USE_LORA"):
        cfg.model.use_lora = os.environ["NEURO_MRI_USE_LORA"].lower() in ("1", "true", "yes")
    if os.environ.get("NEURO_MRI_SAM_ENABLED"):
        cfg.sam.enabled = os.environ["NEURO_MRI_SAM_ENABLED"].lower() in ("1", "true", "yes")
    if os.environ.get("NEURO_MRI_SEQUENTIAL_VRAM"):
        cfg.vram.sequential_models = os.environ["NEURO_MRI_SEQUENTIAL_VRAM"].lower() in (
            "1",
            "true",
            "yes",
        )

    return cfg
