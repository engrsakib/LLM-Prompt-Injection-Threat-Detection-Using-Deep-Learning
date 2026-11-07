"""Shared training loop for every registered classifier.

Features: AdamW, linear / cosine warmup schedules, gradient accumulation, AMP (fp16 / bf16 on
CUDA), early stopping on validation macro-F1, periodic + best checkpoints with optimizer /
scheduler / RNG state for resumption, ``history.jsonl`` per epoch, truncation-rate accounting,
wall-time + hardware capture. The validation split is used ONLY for early stopping / model
selection; the test split is never seen by this module.
"""

from __future__ import annotations

import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.constants import NUM_CLASSES
from src.eval.evaluate import multiclass_metrics
from src.models.base import BaseClassifier, EncodedBatch

STATE_FILE = "training_state.pt"
HISTORY_FILE = "history.jsonl"


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    grad_accum: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    scheduler: str = "linear"  # linear | cosine | constant
    max_length: int = 256
    precision: str = "fp32"  # fp32 | fp16 | bf16
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    early_stopping_metric: str = "f1_macro"
    eval_every_steps: int | None = None  # None -> once per epoch
    max_steps: int | None = None  # for smoke runs
    num_workers: int = 0
    log_every: int = 50
    seed: int = 42
    label_smoothing: float = 0.0
    class_weights: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    best_metric: float | None
    best_epoch: int | None
    epochs_completed: int
    global_step: int
    stopped_early: bool
    wall_time_sec: float
    train_truncation_rate: float
    history: list[dict[str, Any]] = field(default_factory=list)
    hardware: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TextLabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None):
        self.texts = list(texts)
        self.labels = None if labels is None else [int(x) for x in labels]
        if self.labels is not None and len(self.labels) != len(self.texts):
            raise ValueError("texts / labels length mismatch")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], (-1 if self.labels is None else self.labels[idx])


def make_collate(model: BaseClassifier, max_length: int):
    encoder = model.get_encoder()

    def collate(batch):
        texts, labels = zip(*batch)
        enc: EncodedBatch = encoder.encode_batch(list(texts), max_length=max_length)
        return enc, torch.tensor(labels, dtype=torch.long)

    return collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_mem_gb"] = round(props.total_memory / 1e9, 2)
        info["gpu_count"] = torch.cuda.device_count()
    return info


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast_dtype(precision: str, device: torch.device):
    if device.type != "cuda":
        return None
    if precision == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def build_scheduler(optimizer, kind: str, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        if kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)  # linear

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def predict_probabilities(
    model: BaseClassifier,
    texts: Iterable[str],
    batch_size: int = 64,
    max_length: int = 256,
    device: torch.device | None = None,
    precision: str = "fp32",
) -> tuple[np.ndarray, float]:
    """Return (probs [N, C] float64, truncation_rate)."""
    device = device or pick_device()
    texts = list(texts)
    model.eval()
    model.to(device)
    loader = DataLoader(TextLabelDataset(texts, None), batch_size=batch_size, shuffle=False, collate_fn=make_collate(model, max_length))
    chunks: list[np.ndarray] = []
    truncated = 0
    amp_dtype = _autocast_dtype(precision, device)
    for enc, _ in loader:
        enc = enc.to(device)
        truncated += int(enc.truncated.sum().item())
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(enc.input_ids, enc.attention_mask)
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64))
    probs = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, NUM_CLASSES))
    # renormalise tiny float drift so downstream validation passes
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs, (truncated / len(texts) if texts else 0.0)


class Trainer:
    def __init__(self, model: BaseClassifier, config: TrainConfig, output_dir: str | Path, device: torch.device | None = None):
        self.model = model
        self.cfg = config
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = device or pick_device()
        self.history_path = self.out / HISTORY_FILE
        self.checkpoint_dir = self.out / "checkpoint"
        self.last_dir = self.out / "last"

    # --- persistence -------------------------------------------------------
    def _save_state(self, directory: Path, optimizer, scheduler, scaler, epoch: int, step: int, best: dict, patience: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory)
        trainable_state = {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}
        torch.save(
            {
                "model_trainable_state": trainable_state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "global_step": step,
                "best": best,
                "patience_counter": patience,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                "config": self.cfg.to_dict(),
            },
            directory / STATE_FILE,
        )

    def _load_state(self, directory: Path, optimizer, scheduler, scaler) -> dict:
        state = torch.load(directory / STATE_FILE, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(state["model_trainable_state"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected parameters in resume state: {unexpected[:5]}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        return state

    def _append_history(self, record: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # --- evaluation ---------------------------------------------------------
    def evaluate(self, texts: list[str], labels: list[int]) -> dict[str, Any]:
        probs, trunc = predict_probabilities(
            self.model, texts, batch_size=self.cfg.eval_batch_size, max_length=self.cfg.max_length, device=self.device, precision=self.cfg.precision
        )
        metrics = multiclass_metrics(np.asarray(labels), probs)
        metrics["truncation_rate"] = trunc
        return metrics

    # --- main loop ----------------------------------------------------------
    def fit(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
        resume: bool = False,
    ) -> TrainResult:
        cfg = self.cfg
        set_seed(cfg.seed)
        model = self.model.to(self.device)
        loader = DataLoader(
            TextLabelDataset(train_texts, train_labels),
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=make_collate(model, cfg.max_length),
            num_workers=cfg.num_workers,
            generator=torch.Generator().manual_seed(cfg.seed),
            drop_last=False,
        )
        steps_per_epoch = max(1, math.ceil(len(loader) / cfg.grad_accum))
        total_steps = steps_per_epoch * cfg.epochs
        if cfg.max_steps:
            total_steps = min(total_steps, cfg.max_steps)
        warmup = int(cfg.warmup_ratio * total_steps)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scheduler = build_scheduler(optimizer, cfg.scheduler, total_steps, warmup)
        amp_dtype = _autocast_dtype(cfg.precision, self.device)
        scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
        weight = torch.tensor(cfg.class_weights, dtype=torch.float32, device=self.device) if cfg.class_weights else None
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)

        start_epoch, global_step, patience = 0, 0, 0
        best: dict[str, Any] = {"metric": None, "epoch": None}
        if resume and (self.last_dir / STATE_FILE).exists():
            state = self._load_state(self.last_dir, optimizer, scheduler, scaler)
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state["global_step"])
            best = state["best"]
            patience = int(state.get("patience_counter", 0))
            self._append_history({"type": "resume", "from_epoch": start_epoch, "global_step": global_step})
        if (not resume or start_epoch == 0) and self.history_path.exists():
            self.history_path.unlink()

        hw = hardware_info()
        t0 = time.time()
        truncated_total, seen_total = 0, 0
        stopped_early = False
        epochs_completed = start_epoch

        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            running, n_batches = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, (enc, labels) in enumerate(loader):
                enc = enc.to(self.device)
                labels = labels.to(self.device)
                truncated_total += int(enc.truncated.sum().item())
                seen_total += labels.numel()
                with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(enc.input_ids, enc.attention_mask)
                    loss = loss_fn(logits.float(), labels) / cfg.grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                running += float(loss.item()) * cfg.grad_accum
                n_batches += 1
                if (batch_idx + 1) % cfg.grad_accum == 0 or (batch_idx + 1) == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if cfg.log_every and global_step % cfg.log_every == 0:
                        self._append_history({"type": "step", "epoch": epoch, "global_step": global_step, "loss": running / n_batches, "lr": scheduler.get_last_lr()[0]})
                    if cfg.max_steps and global_step >= cfg.max_steps:
                        break
            val = self.evaluate(val_texts, val_labels)
            metric = val.get(cfg.early_stopping_metric)
            record = {
                "type": "epoch",
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": running / max(1, n_batches),
                "lr": scheduler.get_last_lr()[0],
                "val_f1_macro": val["f1_macro"],
                "val_accuracy": val["accuracy"],
                "val_balanced_accuracy": val["balanced_accuracy"],
                "val_truncation_rate": val["truncation_rate"],
                "elapsed_sec": time.time() - t0,
            }
            self._append_history(record)
            epochs_completed = epoch + 1
            improved = metric is not None and (best["metric"] is None or metric > best["metric"] + 1e-9)
            if improved:
                best = {"metric": float(metric), "epoch": epoch}
                patience = 0
                self._save_state(self.checkpoint_dir, optimizer, scheduler, scaler, epoch, global_step, best, patience)
            else:
                patience += 1
            self._save_state(self.last_dir, optimizer, scheduler, scaler, epoch, global_step, best, patience)
            if cfg.max_steps and global_step >= cfg.max_steps:
                break
            if cfg.early_stopping_patience and patience >= cfg.early_stopping_patience:
                stopped_early = True
                break

        if not self.checkpoint_dir.exists():  # e.g. zero epochs; still persist something
            self._save_state(self.checkpoint_dir, optimizer, scheduler, scaler, epochs_completed - 1, global_step, best, patience)

        history = [json.loads(line) for line in self.history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return TrainResult(
            best_metric=best["metric"],
            best_epoch=best["epoch"],
            epochs_completed=epochs_completed,
            global_step=global_step,
            stopped_early=stopped_early,
            wall_time_sec=time.time() - t0,
            train_truncation_rate=(truncated_total / seen_total) if seen_total else 0.0,
            history=history,
            hardware=hw,
        )
