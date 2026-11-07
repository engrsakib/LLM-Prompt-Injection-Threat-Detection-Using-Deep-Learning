"""Model package: registry, shared interface and backbone implementations.

Heavy dependencies (transformers / peft) are imported lazily inside the builders so that
``src.models.registry`` can be inspected without a GPU or model downloads.
"""

from src.models.base import BaseClassifier, EncodedBatch
from src.models.registry import REGISTRY, ModelSpec, build_model, get_spec, list_model_ids, load_model

__all__ = [
    "BaseClassifier",
    "EncodedBatch",
    "REGISTRY",
    "ModelSpec",
    "build_model",
    "get_spec",
    "list_model_ids",
    "load_model",
]
