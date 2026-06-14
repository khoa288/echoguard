"""Empirical conservative and calibration-based escalation shields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConservativeBudget:
    tolerance: float = 0.05
    confidence_scale: float = 1.96
    candidate_total: float = 0.0
    baseline_total: float = 0.0
    squared_total: float = 0.0
    observations: int = 0
    fallbacks: int = 0

    def allow(self, estimated_candidate_reward: float, baseline_reward: float) -> bool:
        projected = self.candidate_total + estimated_candidate_reward
        baseline = self.baseline_total + baseline_reward
        uncertainty = self.confidence_scale * np.sqrt(
            self.squared_total / max(1, self.observations)
        )
        allowed = projected - uncertainty >= (1 - self.tolerance) * baseline
        if not allowed:
            self.fallbacks += 1
        return bool(allowed)

    def observe(self, candidate_reward: float, baseline_reward: float) -> None:
        self.candidate_total += candidate_reward
        self.baseline_total += baseline_reward
        self.squared_total += candidate_reward * candidate_reward
        self.observations += 1


@dataclass(frozen=True)
class RiskShield:
    """Calibration quantile shield; intended for held-out calibration scores."""

    threshold: float

    @classmethod
    def fit(cls, miss_risk_scores: np.ndarray, target_quantile: float = 0.95) -> "RiskShield":
        values = np.asarray(miss_risk_scores, dtype=float)
        if values.size == 0:
            raise ValueError("Calibration scores cannot be empty")
        return cls(float(np.quantile(values, target_quantile, method="higher")))

    def should_escalate(self, estimated_miss_risk: float) -> bool:
        return bool(estimated_miss_risk > self.threshold)
