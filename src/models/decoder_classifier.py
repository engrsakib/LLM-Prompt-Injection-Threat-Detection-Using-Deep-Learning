"""Decoder-only LLM (Mistral / Qwen3 / Llama) sequence classifier with LoRA or QLoRA.

Design notes
------------
* ``AutoModelForSequenceClassification`` for decoder models pools the hidden state of the
  **last non-pad token** (requires ``config.pad_token_id``). We use right padding and verify
  padding invariance in ``tests/test_models.py``.
* The classification head (``score``) is randomly initialised, so it is added to
  ``modules_to_save`` and therefore trained in full precision and stored with the adapter.
* Probabilities are the softmax of the seven-way head - never generated prose.
* QLoRA (4-bit) requires ``bitsandbytes`` on a CUDA device; otherwise we fall back to
  plain LoRA in the requested dtype and record the fallback in ``load_info``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.constants import NUM_CLASSES
from src.models.base import BaseClassifier
from src.models.hf_encoder import HFTokenizerEncoder, ensure_pad_token

DECODER_META_FILENAME = "decoder_classifier.json"
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def _resolve_dtype(name: str | None) -> torch.dtype | None:
    if name in (None, "auto"):
        return None
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


class DecoderLoRAClassifier(BaseClassifier):
    kind = "decoder_lora"

    def __init__(
        self,
        checkpoint: str,
        num_labels: int = NUM_CLASSES,
        revision: str | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: tuple[str, ...] | list[str] | None = None,
        quantization: str | None = None,  # None | "4bit" | "8bit"
        torch_dtype: str | None = "bfloat16",
        gradient_checkpointing: bool = True,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        token: str | None = None,
        device_map: str | dict | None = None,
        adapter_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

        self.checkpoint = checkpoint
        self.revision = revision
        self.num_labels = num_labels
        self.load_info: dict[str, Any] = {"requested_quantization": quantization}

        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "token": token,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **load_kwargs)
        self.tokenizer.padding_side = "right"
        added = ensure_pad_token(self.tokenizer)

        quant_config = None
        use_cuda = torch.cuda.is_available()
        if quantization in ("4bit", "8bit"):
            if use_cuda and bitsandbytes_available():
                from transformers import BitsAndBytesConfig

                if quantization == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                else:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                self.load_info["quantization"] = quantization
            else:
                self.load_info["quantization"] = None
                self.load_info["fallback_reason"] = (
                    "bitsandbytes/CUDA unavailable; loaded without quantization"
                )
        else:
            self.load_info["quantization"] = None

        model_kwargs: dict[str, Any] = dict(load_kwargs)
        dtype = _resolve_dtype(torch_dtype)
        if dtype is not None and not use_cuda and dtype != torch.float32:
            # CPU bf16/fp16 training is unsupported for most kernels; keep fp32 and record it.
            self.load_info["dtype_fallback"] = f"{torch_dtype}->float32 (no CUDA)"
            dtype = torch.float32
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = device_map or "auto"
        elif device_map is not None:
            model_kwargs["device_map"] = device_map

        base = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels, **model_kwargs
        )
        if added:
            base.resize_token_embeddings(len(self.tokenizer))
        base.config.pad_token_id = self.tokenizer.pad_token_id
        base.config.id2label = {i: n for i, n in enumerate(self.label_order)}
        base.config.label2id = {n: i for i, n in enumerate(self.label_order)}

        if quant_config is not None:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=gradient_checkpointing)
        elif gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
            try:
                base.gradient_checkpointing_enable()
                base.config.use_cache = False
            except (ValueError, AttributeError):  # tiny test configs may not support it
                pass

        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
        else:
            lora = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(target_modules or DEFAULT_TARGET_MODULES),
                modules_to_save=["score"],
            )
            self.model = get_peft_model(base, lora)
        self.lora_config = {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": list(target_modules or DEFAULT_TARGET_MODULES),
        }
        self._encoder = HFTokenizerEncoder(self.tokenizer)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def get_encoder(self) -> HFTokenizerEncoder:
        return self._encoder

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Saves LoRA adapters + the `score` head (modules_to_save) + adapter_config.json
        self.model.save_pretrained(str(directory))
        self.tokenizer.save_pretrained(str(directory))
        (directory / DECODER_META_FILENAME).write_text(
            json.dumps(
                {
                    "checkpoint": self.checkpoint,
                    "revision": self.revision,
                    "num_labels": self.num_labels,
                    "lora": self.lora_config,
                    "load_info": self.load_info,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.write_meta(directory, {"checkpoint": self.checkpoint, "load_info": self.load_info})

    @classmethod
    def load(
        cls,
        directory: str | Path,
        base_checkpoint: str | None = None,
        **kwargs: Any,
    ) -> "DecoderLoRAClassifier":
        """Reload base weights (from hub/cache or ``base_checkpoint``) and attach the saved adapter."""
        directory = Path(directory)
        meta = json.loads((directory / DECODER_META_FILENAME).read_text(encoding="utf-8"))
        lora = meta.get("lora", {})
        return cls(
            checkpoint=base_checkpoint or meta["checkpoint"],
            num_labels=int(meta.get("num_labels", NUM_CLASSES)),
            revision=meta.get("revision"),
            lora_r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=tuple(lora.get("target_modules", DEFAULT_TARGET_MODULES)),
            adapter_dir=directory,
            **kwargs,
        )
