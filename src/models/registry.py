"""Model registry for the 16-backbone prompt-injection study.

Every entry records *what we intend to run* and *what has actually been verified*.
Static fields below are the declared plan; ``derive_status`` upgrades the status from
artifacts on disk (smoke runs, trained checkpoints, evaluation files). A registered entry
never implies a trained model.

Status vocabulary (ordered): planned < implemented < smoke-tested < trained < evaluated,
plus ``blocked`` (access / resource blocker recorded in ``access_note``).

Checkpoint verification (Hugging Face API, 2026-11-07):
* ``ehsanaul/securebert-base`` (the identifier previously listed in the README) returns
  401/not found. The public SecureBERT release is ``ehsanaghaei/SecureBERT`` (RoBERTa-base
  continued pre-training on cybersecurity text, license bigscience-openrail-m). We register
  the corrected identifier and keep the original in ``access_note`` for traceability.
* ``meta-llama/Llama-3.1-8B`` is gated (manual approval); registered as ``blocked`` until a
  token with approved access is supplied through ``HF_TOKEN`` (env / Kaggle Secret).
* ``mistralai/Mistral-7B-v0.3`` is the ungated official *base* checkpoint (Apache-2.0).
* ``Qwen/Qwen3-8B`` is the post-trained (chat / thinking) release; ``Qwen/Qwen3-8B-Base``
  is the base model. We pin the user-requested ``Qwen/Qwen3-8B`` and document the difference.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.data.constants import NUM_CLASSES

STATUS_ORDER = ("planned", "implemented", "smoke-tested", "trained", "evaluated")
VALID_STATUSES = STATUS_ORDER + ("blocked",)
REGISTRY_JSON = Path("docs") / "registry.json"


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    family: str
    loader: str  # hf_encoder | textcnn | bilstm_attention | cnn_bilstm | decoder_lora
    checkpoint: str | None
    revision: str | None
    head: str
    compute_tier: str  # cpu_ok | gpu_small | gpu_medium | gpu_large_qlora
    thesis_role: str
    license: str
    access_note: str
    status: str
    params_hub: int | None = None  # as reported by the Hub (safetensors metadata)
    params_total: int | None = None  # measured when the model is instantiated
    params_trainable: int | None = None
    min_transformers: str = "4.46.0"
    hparams: dict[str, Any] = field(default_factory=dict)
    gated: bool = False
    experimental: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _enc(lr: float, bs: int, precision: str = "fp32", epochs: int = 4, max_length: int = 256) -> dict[str, Any]:
    return {
        "learning_rate": lr,
        "batch_size": bs,
        "grad_accum": 1,
        "epochs": epochs,
        "max_length": max_length,
        "precision": precision,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "scheduler": "linear",
    }


def _custom(lr: float = 1e-3, bs: int = 64, epochs: int = 10) -> dict[str, Any]:
    return {
        "learning_rate": lr,
        "batch_size": bs,
        "grad_accum": 1,
        "epochs": epochs,
        "max_length": 256,
        "precision": "fp32",
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "scheduler": "linear",
        "vocab_min_freq": 2,
        "vocab_max_size": 50000,
    }


def _llm(bs: int = 4, accum: int = 8) -> dict[str, Any]:
    return {
        "learning_rate": 1e-4,
        "batch_size": bs,
        "grad_accum": accum,
        "epochs": 2,
        "max_length": 256,
        "precision": "bf16",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "quantization": "4bit",
        "gradient_checkpointing": True,
    }


REGISTRY: list[ModelSpec] = [
    ModelSpec("distilbert", "DistilBERT", "distilled transformer encoder", "hf_encoder",
              "distilbert-base-uncased", None, "linear (pre-classifier + classifier)", "cpu_ok",
              "Primary baseline; aligned with the dataset's published DistilBERT baseline",
              "Apache-2.0", "ungated", "implemented", hparams=_enc(5e-5, 32)),
    ModelSpec("bert", "BERT", "transformer encoder", "hf_encoder",
              "bert-base-uncased", None, "pooler + linear", "gpu_small",
              "Canonical text-classification baseline", "Apache-2.0", "ungated", "implemented",
              hparams=_enc(3e-5, 32)),
    ModelSpec("roberta", "RoBERTa", "transformer encoder", "hf_encoder",
              "roberta-base", None, "classification head (dense+tanh+out)", "gpu_small",
              "Robust pre-training for adversarial text", "MIT", "ungated", "implemented",
              hparams=_enc(2e-5, 32)),
    ModelSpec("deberta_v3", "DeBERTa-v3", "disentangled-attention encoder", "hf_encoder",
              "microsoft/deberta-v3-base", None, "pooler + linear", "gpu_small",
              "State-of-the-art base-size encoder candidate", "MIT", "ungated; requires sentencepiece",
              "implemented", hparams=_enc(2e-5, 16, epochs=4)),
    ModelSpec("securebert", "SecureBERT", "cybersecurity RoBERTa encoder", "hf_encoder",
              "ehsanaghaei/SecureBERT", "3a47918dd874e5c769efd152b51c5756a953fb67",
              "classification head (dense+tanh+out)", "gpu_small",
              "Domain-specific security embeddings",
              "bigscience-openrail-m",
              "CORRECTED identifier: README previously listed `ehsanaul/securebert-base` (HF API: 401/not public). "
              "`ehsanaghaei/SecureBERT` is the public release; HF lists `cisco-ai/SecureBERT2.0-base` as a newer version.",
              "implemented", hparams=_enc(2e-5, 32)),
    ModelSpec("albert", "ALBERT", "parameter-shared encoder", "hf_encoder",
              "albert-base-v2", None, "pooler + linear", "cpu_ok",
              "Parameter-efficient comparison", "Apache-2.0", "ungated; requires sentencepiece",
              "implemented", hparams=_enc(3e-5, 32)),
    ModelSpec("electra_small", "ELECTRA-small", "discriminator-pretrained encoder", "hf_encoder",
              "google/electra-small-discriminator", None, "classification head", "cpu_ok",
              "Fast compute-efficient encoder", "Apache-2.0", "ungated", "implemented",
              hparams=_enc(1e-4, 32)),
    ModelSpec("textcnn", "TextCNN", "custom convolutional", "textcnn", None, None,
              "linear over max-pooled conv features", "cpu_ok", "Classical deep-learning baseline",
              "MIT (this repo)", "custom implementation; vocab fitted on train only",
              "implemented", hparams=_custom()),
    ModelSpec("bilstm_attention", "BiLSTM + Attention", "custom recurrent", "bilstm_attention", None, None,
              "additive attention pooling + linear", "cpu_ok", "Sequential / obfuscation patterns",
              "MIT (this repo)", "custom implementation; vocab fitted on train only",
              "implemented", hparams=_custom()),
    ModelSpec("cnn_bilstm", "CNN-BiLSTM", "custom hybrid", "cnn_bilstm", None, None,
              "mean+max pooled BiLSTM + linear", "cpu_ok", "Local + long-range feature fusion",
              "MIT (this repo)", "custom implementation; vocab fitted on train only",
              "implemented", hparams=_custom()),
    ModelSpec("distilroberta", "DistilRoBERTa", "distilled transformer encoder", "hf_encoder",
              "distilroberta-base", None, "classification head (dense+tanh+out)", "cpu_ok",
              "Accuracy / speed trade-off", "Apache-2.0", "ungated", "implemented",
              hparams=_enc(5e-5, 32)),
    ModelSpec("xlnet", "XLNet", "permutation language model", "hf_encoder",
              "xlnet-base-cased", None, "sequence summary + logits proj", "gpu_small",
              "Bidirectional context modelling", "MIT", "ungated; requires sentencepiece",
              "implemented", hparams=_enc(2e-5, 16)),
    # --- Additional experimental backbones (research candidates, not presumed superior) ---
    ModelSpec("modernbert_large", "ModernBERT-large", "long-context encoder", "hf_encoder",
              "answerdotai/ModernBERT-large", "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
              "linear", "gpu_medium", "Modern encoder candidate (8k context)", "Apache-2.0",
              "ungated; requires transformers>=4.48", "implemented", params_hub=395_881_664,
              min_transformers="4.48.0", hparams=_enc(3e-5, 16, precision="bf16", epochs=3),
              experimental=True),
    ModelSpec("mistral_7b", "Mistral-7B (v0.3 base)", "decoder LLM + LoRA head", "decoder_lora",
              "mistralai/Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1",
              "score (linear) on last non-pad token; LoRA adapters", "gpu_large_qlora",
              "Large decoder candidate (QLoRA)", "Apache-2.0",
              "ungated official BASE checkpoint (not Instruct). Needs ~16GB VRAM with 4-bit QLoRA; "
              "never loaded on the local CPU machine.",
              "implemented", params_hub=7_248_023_552, min_transformers="4.46.0",
              hparams=_llm(), experimental=True),
    ModelSpec("qwen3_8b", "Qwen3-8B", "decoder LLM + LoRA head", "decoder_lora",
              "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218",
              "score (linear) on last non-pad token; LoRA adapters", "gpu_large_qlora",
              "Large decoder candidate (QLoRA)", "Apache-2.0",
              "ungated. NOTE: `Qwen/Qwen3-8B` is the POST-TRAINED (chat/thinking) release; the base model is "
              "`Qwen/Qwen3-8B-Base`. Requires transformers>=4.51. Never loaded on the local CPU machine.",
              "implemented", params_hub=8_190_735_360, min_transformers="4.51.0",
              hparams=_llm(), experimental=True),
    ModelSpec("llama31_8b", "Llama-3.1-8B", "decoder LLM + LoRA head", "decoder_lora",
              "meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
              "score (linear) on last non-pad token; LoRA adapters", "gpu_large_qlora",
              "Large decoder candidate (QLoRA)", "Llama 3.1 Community License",
              "BLOCKED: gated repository (manual approval). Provide HF_TOKEN with approved access via "
              "environment variable / Kaggle Secret; the code path is identical to Mistral/Qwen3.",
              "blocked", params_hub=8_030_261_248, min_transformers="4.43.0",
              hparams=_llm(), gated=True, experimental=True),
]

assert len(REGISTRY) == 16, "registry must contain exactly 16 backbones"
_BY_ID = {spec.model_id: spec for spec in REGISTRY}
assert len(_BY_ID) == 16, "duplicate model ids"


def get_spec(model_id: str) -> ModelSpec:
    try:
        return _BY_ID[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model_id '{model_id}'. Known: {sorted(_BY_ID)}") from exc


def list_model_ids() -> list[str]:
    return [s.model_id for s in REGISTRY]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(
    spec: ModelSpec,
    train_texts: list[str] | None = None,
    hparams: dict[str, Any] | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    token: str | None = None,
    checkpoint_override: str | None = None,
):
    """Instantiate the classifier described by ``spec``.

    Custom models need ``train_texts`` (train split only) to fit their vocabulary.
    """
    hp = dict(spec.hparams)
    if hparams:
        hp.update(hparams)
    token = token or os.environ.get("HF_TOKEN") or None
    checkpoint = checkpoint_override or spec.checkpoint

    if spec.loader == "hf_encoder":
        from src.models.hf_encoder import HFEncoderClassifier

        return HFEncoderClassifier(
            checkpoint=checkpoint,
            num_labels=NUM_CLASSES,
            revision=None if checkpoint_override else spec.revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )

    if spec.loader == "decoder_lora":
        from src.models.decoder_classifier import DecoderLoRAClassifier

        return DecoderLoRAClassifier(
            checkpoint=checkpoint,
            num_labels=NUM_CLASSES,
            revision=None if checkpoint_override else spec.revision,
            lora_r=int(hp.get("lora_r", 16)),
            lora_alpha=int(hp.get("lora_alpha", 32)),
            lora_dropout=float(hp.get("lora_dropout", 0.05)),
            quantization=hp.get("quantization"),
            torch_dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(hp.get("precision", "bf16")),
            gradient_checkpointing=bool(hp.get("gradient_checkpointing", True)),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )

    # Custom models
    from src.models.vocab import Vocab

    if train_texts is None:
        raise ValueError(f"{spec.model_id} needs train_texts to fit its vocabulary")
    vocab = Vocab.fit(
        train_texts,
        min_freq=int(hp.get("vocab_min_freq", 2)),
        max_size=int(hp.get("vocab_max_size", 50_000)),
    )
    if spec.loader == "textcnn":
        from src.models.textcnn import TextCNN

        return TextCNN(vocab=vocab, embed_dim=int(hp.get("embed_dim", 128)), num_filters=int(hp.get("num_filters", 100)))
    if spec.loader == "bilstm_attention":
        from src.models.bilstm_attention import BiLSTMAttention

        return BiLSTMAttention(vocab=vocab, embed_dim=int(hp.get("embed_dim", 128)), hidden_dim=int(hp.get("hidden_dim", 128)))
    if spec.loader == "cnn_bilstm":
        from src.models.cnn_bilstm import CNNBiLSTM

        return CNNBiLSTM(vocab=vocab, embed_dim=int(hp.get("embed_dim", 128)), hidden_dim=int(hp.get("hidden_dim", 128)))
    raise ValueError(f"Unknown loader '{spec.loader}'")


def load_model(spec: ModelSpec, directory: str | Path, **kwargs: Any):
    """Reload a saved checkpoint produced by ``BaseClassifier.save``."""
    if spec.loader == "hf_encoder":
        from src.models.hf_encoder import HFEncoderClassifier

        return HFEncoderClassifier.load(directory, **kwargs)
    if spec.loader == "decoder_lora":
        from src.models.decoder_classifier import DecoderLoRAClassifier

        return DecoderLoRAClassifier.load(directory, **kwargs)
    if spec.loader == "textcnn":
        from src.models.textcnn import TextCNN

        return TextCNN.load(directory)
    if spec.loader == "bilstm_attention":
        from src.models.bilstm_attention import BiLSTMAttention

        return BiLSTMAttention.load(directory)
    if spec.loader == "cnn_bilstm":
        from src.models.cnn_bilstm import CNNBiLSTM

        return CNNBiLSTM.load(directory)
    raise ValueError(f"Unknown loader '{spec.loader}'")


# ---------------------------------------------------------------------------
# Status derivation from artifacts
# ---------------------------------------------------------------------------
def derive_status(spec: ModelSpec, runs_dir: str | Path = "outputs/runs", smoke_dir: str | Path = "outputs/smoke") -> dict[str, Any]:
    """Combine the declared status with evidence on disk.

    * ``outputs/smoke/<model_id>*/run_summary.json`` -> smoke-tested
    * ``outputs/runs/<model_id>/seed_*/checkpoint`` -> trained
    * ``outputs/runs/<model_id>/seed_*/metrics_test.json`` -> evaluated
    """
    runs_dir, smoke_dir = Path(runs_dir), Path(smoke_dir)
    evidence: dict[str, Any] = {"smoke_runs": [], "trained_seeds": [], "evaluated_seeds": []}
    status = spec.status
    if status == "blocked":
        return {"model_id": spec.model_id, "status": status, "evidence": evidence}

    for path in sorted(smoke_dir.glob(f"{spec.model_id}*/run_summary.json")):
        evidence["smoke_runs"].append(str(path.as_posix()))
    model_runs = runs_dir / spec.model_id
    for seed_dir in sorted(model_runs.glob("seed_*")) if model_runs.exists() else []:
        if (seed_dir / "checkpoint").exists() and (seed_dir / "run_summary.json").exists():
            evidence["trained_seeds"].append(seed_dir.name)
        if (seed_dir / "metrics_test.json").exists():
            evidence["evaluated_seeds"].append(seed_dir.name)

    rank = STATUS_ORDER.index(status) if status in STATUS_ORDER else 0
    if evidence["smoke_runs"]:
        rank = max(rank, STATUS_ORDER.index("smoke-tested"))
    if evidence["trained_seeds"]:
        rank = max(rank, STATUS_ORDER.index("trained"))
    if evidence["evaluated_seeds"]:
        rank = max(rank, STATUS_ORDER.index("evaluated"))
    return {"model_id": spec.model_id, "status": STATUS_ORDER[rank], "evidence": evidence}


def export_registry(path: str | Path = REGISTRY_JSON, runs_dir: str | Path = "outputs/runs", smoke_dir: str | Path = "outputs/smoke") -> list[dict[str, Any]]:
    rows = []
    for spec in REGISTRY:
        row = spec.to_dict()
        derived = derive_status(spec, runs_dir=runs_dir, smoke_dir=smoke_dir)
        row["status"] = derived["status"]
        row["evidence"] = derived["evidence"]
        rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def write_registry_markdown(rows: list[dict[str, Any]], path: str | Path = Path("docs") / "MODEL_REGISTRY.md") -> Path:
    """Render ``docs/MODEL_REGISTRY.md`` from exported rows (declared plan + on-disk evidence)."""
    from datetime import UTC, datetime

    lines = [
        "# Model Registry (16 backbones)",
        "",
        f"_Generated by `python -m src.models.registry` on {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}. "
        "Statuses are derived from artifacts on disk (`outputs/smoke`, `outputs/runs`); a registered entry never implies a trained model._",
        "",
        "Status vocabulary: `planned` < `implemented` < `smoke-tested` < `trained` < `evaluated`; `blocked` = access/resource blocker (see note).",
        "",
        "| # | model_id | Model | Family | Checkpoint (revision) | Params (Hub / measured) | Head | Tier | Status | Role |",
        "|:-:|----------|-------|--------|-----------------------|------------------------|------|------|--------|------|",
    ]
    for i, r in enumerate(rows, 1):
        ckpt = f"`{r['checkpoint']}`" if r["checkpoint"] else "_custom (this repo)_"
        if r.get("revision"):
            ckpt += f"<br/>`{r['revision'][:12]}`"
        params = []
        if r.get("params_hub"):
            params.append(f"hub {r['params_hub'] / 1e6:,.0f}M")
        if r.get("params_total"):
            params.append(f"measured {r['params_total'] / 1e6:,.1f}M total / {r['params_trainable'] / 1e6:,.1f}M trainable")
        lines.append(
            f"| {i} | `{r['model_id']}` | {r['display_name']} | {r['family']} | {ckpt} | {'; '.join(params) or '-'} | {r['head']} | `{r['compute_tier']}` | **{r['status']}** | {r['thesis_role']} |"
        )
    lines += ["", "## Access, licence and verification notes", ""]
    for r in rows:
        lines.append(f"- **{r['display_name']}** (`{r['model_id']}`) - licence: {r['license']}; min transformers {r['min_transformers']}; {r['access_note']}")
    lines += [
        "",
        "## Evidence on disk",
        "",
        "| model_id | smoke runs | trained seeds | evaluated seeds |",
        "|----------|-----------|---------------|-----------------|",
    ]
    for r in rows:
        ev = r.get("evidence", {})
        lines.append(f"| `{r['model_id']}` | {len(ev.get('smoke_runs', []))} | {', '.join(ev.get('trained_seeds', [])) or '-'} | {', '.join(ev.get('evaluated_seeds', [])) or '-'} |")
    lines += [
        "",
        "## Excluded by design",
        "",
        "GPT-4 Attack Validator, Lakera Guard and MetaSecAlign are **not** registered: they are external / paid / "
        "generative systems that would send data outside the environment. They remain disabled and out of scope.",
        "",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def check_registry(online: bool = False, token: str | None = None) -> list[dict[str, Any]]:
    """Config-level verification: version requirements and (optionally) Hub metadata.

    Never downloads model weights. With ``--online`` it queries the Hub API for the resolved
    commit sha, gating flag and parameter count so revisions can be pinned.
    """
    import transformers
    from packaging.version import Version

    results = []
    tv = Version(transformers.__version__)
    for spec in REGISTRY:
        item: dict[str, Any] = {
            "model_id": spec.model_id,
            "checkpoint": spec.checkpoint,
            "declared_status": spec.status,
            "transformers_ok": tv >= Version(spec.min_transformers),
        }
        if online and spec.checkpoint:
            try:
                from huggingface_hub import model_info

                info = model_info(spec.checkpoint, revision=spec.revision, token=token or os.environ.get("HF_TOKEN"))
                item["resolved_sha"] = info.sha
                item["gated"] = bool(getattr(info, "gated", False))
                st = getattr(info, "safetensors", None)
                item["params_hub"] = int(st.total) if st is not None and getattr(st, "total", None) else None
                item["hub_ok"] = True
            except Exception as exc:  # noqa: BLE001 - we want the message, not a crash
                item["hub_ok"] = False
                item["hub_error"] = f"{type(exc).__name__}: {exc}"
        results.append(item)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect / verify the 16-model registry.")
    parser.add_argument("--check", action="store_true", help="Run config-level checks.")
    parser.add_argument("--online", action="store_true", help="Query the Hugging Face Hub API (no weight downloads).")
    parser.add_argument("--export", default=str(REGISTRY_JSON), help="Write registry JSON with derived statuses.")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--smoke-dir", default="outputs/smoke")
    parser.add_argument("--markdown", default=str(Path("docs") / "MODEL_REGISTRY.md"), help="Write the Markdown registry table ('' to skip).")
    args = parser.parse_args(argv)

    rows = export_registry(args.export, runs_dir=args.runs_dir, smoke_dir=args.smoke_dir)
    if args.markdown:
        # measured parameter counts from smoke/run configs, when available
        for row in rows:
            for cfg in sorted(Path(args.smoke_dir).glob(f"{row['model_id']}*/config.json")) + sorted(Path(args.runs_dir).glob(f"{row['model_id']}/seed_*/config.json")):
                try:
                    model_cfg = json.loads(cfg.read_text(encoding="utf-8"))["model"]
                    row["params_total"], row["params_trainable"] = model_cfg.get("params_total"), model_cfg.get("params_trainable")
                    break
                except (KeyError, json.JSONDecodeError):
                    continue
        print(f"Wrote {write_registry_markdown(rows, args.markdown)}")
    print(f"{'model_id':<18}{'status':<14}{'tier':<18}checkpoint")
    for row in rows:
        print(f"{row['model_id']:<18}{row['status']:<14}{row['compute_tier']:<18}{row['checkpoint'] or '(custom)'}")
    print(f"\nWrote {args.export}")

    if args.check:
        results = check_registry(online=args.online)
        for item in results:
            flag = "OK " if item["transformers_ok"] else "VER"
            extra = ""
            if "hub_ok" in item:
                extra = f" sha={item.get('resolved_sha')} gated={item.get('gated')} params={item.get('params_hub')}" if item["hub_ok"] else f" HUB_ERROR={item['hub_error']}"
            print(f"[{flag}] {item['model_id']:<18}{extra}")
        bad = [r for r in results if not r["transformers_ok"]]
        if bad:
            print(f"\n{len(bad)} model(s) need a newer transformers version.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
