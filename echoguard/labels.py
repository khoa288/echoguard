"""Canonical ESC-50 danger labels and probability aggregation."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

DANGER_CATEGORIES = frozenset(
    {
        "siren",
        "car_horn",
        "glass_breaking",
        "thunderstorm",
        "crying_baby",
        "dog",
        "door_wood_knock",
        "clock_alarm",
    }
)

ESC50_CATEGORIES = (
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep",
    "crow", "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
    "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
    "crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps",
    "laughing", "brushing_teeth", "snoring", "drinking_sipping", "door_wood_knock",
    "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening",
    "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick",
    "glass_breaking", "helicopter", "chainsaw", "siren", "car_horn", "engine",
    "train", "church_bells", "airplane", "fireworks", "hand_saw",
)
EMERGENCY_CLASS_INDICES = tuple(
    index for index, name in enumerate(ESC50_CATEGORIES) if name in DANGER_CATEGORIES
)
EMERGENCY_CLASS_SET_VERSION = "esc50-danger-v1"


def emergency_probability(
    probabilities: Sequence[float] | np.ndarray,
    indices: Iterable[int] = EMERGENCY_CLASS_INDICES,
) -> float | np.ndarray:
    """Sum emergency-class probabilities for one vector or a batch."""
    values = np.asarray(probabilities, dtype=float)
    if values.shape[-1] != len(ESC50_CATEGORIES):
        raise ValueError(f"Expected 50 class probabilities, got shape {values.shape}")
    return values[..., tuple(indices)].sum(axis=-1)
