"""Versioned configuration for counterfactual logging and replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class BanditConfig:
    schema_version: str = "1.0"
    sample_rate: int = 32_000
    model_window_seconds: float = 5.0
    decision_hop_seconds: float = 0.5
    label_window_seconds: float = 5.0
    danger_threshold: float = 0.30
    uncertainty_low: float = 0.25
    uncertainty_high: float = 0.65
    expert_threshold: float = 0.50
    seeds: tuple[int, ...] = (2026,)
    checkpoints: Mapping[str, str] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
