"""Versioned primitive compute costs used by path simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class CostModel:
    """Per-call MAC estimates. Override values with measured deployment costs."""

    version: str = "echoguard-notebook-v1"
    stage1: float = 189_746.752
    models: Mapping[str, float] = field(
        default_factory=lambda: {
            "bc1": 479_744.0,
            "bc2": 958_208.0,
            "bc3": 1_437_696.0,
            "bc8": 3_831_808.0,
            "mn04": 18_700_000.0,
            "dymn10": 67_500_000.0,
        }
    )

    def model(self, name: str) -> float:
        if name not in self.models:
            raise KeyError(f"Unknown model cost: {name}")
        return float(self.models[name])
