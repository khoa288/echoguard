"""Stage-3 50-class expert identities and normalized output summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from echoguard.labels import ESC50_CATEGORIES, emergency_probability

STAGE3_CHECKPOINTS: Mapping[str, str] = {
    "mn04": "mn04_as.pt",
    "dymn10": "dymn10_as.pt",
}


@dataclass(frozen=True)
class Stage3Output:
    probabilities: np.ndarray
    emergency_probability: float
    top_class: int
    entropy: float
    margin: float


def summarize_probabilities(probabilities: np.ndarray) -> Stage3Output:
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    if probs.size != len(ESC50_CATEGORIES):
        raise ValueError(f"Expected 50 probabilities, got {probs.size}")
    if np.any(probs < 0) or not np.isfinite(probs).all():
        raise ValueError("Probabilities must be finite and non-negative")
    total = probs.sum()
    if not np.isclose(total, 1.0, atol=1e-4):
        if total <= 0:
            raise ValueError("Probability sum must be positive")
        probs = probs / total
    ordered = np.sort(probs)
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return Stage3Output(
        probabilities=probs,
        emergency_probability=float(emergency_probability(probs)),
        top_class=int(np.argmax(probs)),
        entropy=entropy,
        margin=float(ordered[-1] - ordered[-2]),
    )


def predict_expert(predictor: Callable[[np.ndarray], np.ndarray], waveform: np.ndarray) -> Stage3Output:
    return summarize_probabilities(predictor(waveform))


def validate_checkpoint_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    paths = {name: root / filename for name, filename in STAGE3_CHECKPOINTS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage-3 checkpoints: " + ", ".join(missing))
    return paths
