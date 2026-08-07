# Copyright (C) 2026 Md. Nazmus Sakib
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Canonical neurological MRI class labels and dataset constants."""

from __future__ import annotations

EXPECTED_CLASS_NAMES: list[str] = [
    "AD_MildDemented",
    "AD_ModerateDemented",
    "AD_VeryMildDemented",
    "BT_glioma",
    "BT_meningioma",
    "BT_pituitary",
    "MS",
    "Normal",
]

NUM_CLASSES: int = len(EXPECTED_CLASS_NAMES)

DEFAULT_KAGGLE_DATA_SUBDIR: str = "data"
