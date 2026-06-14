"""Small audio-window utilities shared by scripts and notebooks."""

from __future__ import annotations

import numpy as np


def pad_or_trim(waveform: np.ndarray, samples: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size >= samples:
        return waveform[:samples]
    return np.pad(waveform, (0, samples - waveform.size))


def extract_window(
    waveform: np.ndarray, center_seconds: float, sample_rate: int, duration_seconds: float
) -> np.ndarray:
    size = int(round(sample_rate * duration_seconds))
    start = int(round(center_seconds * sample_rate)) - size // 2
    left = max(0, -start)
    right = max(0, start + size - len(waveform))
    chunk = np.asarray(waveform, dtype=np.float32)[max(0, start) : min(len(waveform), start + size)]
    return np.pad(chunk, (left, right))[:size]
