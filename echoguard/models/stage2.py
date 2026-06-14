"""Stage-2 BC-ResNet model identities and normalized inference adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

STAGE2_CHECKPOINTS: Mapping[str, str] = {
    "bc1": "bcresnet1.pt",
    "bc2": "bcresnet2.pt",
    "bc3": "bcresnet3.pt",
    "bc8": "bcresnet8.pt",
}


@dataclass(frozen=True)
class Stage2Output:
    p_danger: float

    def __post_init__(self) -> None:
        if not 0 <= self.p_danger <= 1:
            raise ValueError("p_danger must be in [0, 1]")


def validate_checkpoint_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    paths = {name: root / filename for name, filename in STAGE2_CHECKPOINTS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage-2 checkpoints: " + ", ".join(missing))
    return paths


def predict_p_danger(predictor: Callable[[np.ndarray], float], waveform: np.ndarray) -> Stage2Output:
    return Stage2Output(float(predictor(waveform)))
