# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Feature extraction and explainability (XAI) modules."""

from neuro_mri_xai.explainability.attention_rollout import compute_attention_rollout
from neuro_mri_xai.explainability.gradcam import GradCAM, display_gradcam
from neuro_mri_xai.explainability.pipeline import explain_sample
from neuro_mri_xai.explainability.sam_overlay import create_sam_overlay, save_sam_overlay

__all__ = [
    "GradCAM",
    "display_gradcam",
    "compute_attention_rollout",
    "create_sam_overlay",
    "save_sam_overlay",
    "explain_sample",
]
