"""Asymmetric, decomposed emergency-detection reward."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    compute_reference_macs: float = 67_500_000.0
    correct_emergency: float = 2.0
    correct_safe: float = 0.2
    false_positive_penalty: float = 1.0
    false_negative_penalty: float = 20.0
    delay_penalty_per_second: float = 0.1
    compute_penalty: float = 0.1


@dataclass(frozen=True)
class RewardResult:
    utility: float
    error_penalty: float
    delay_penalty: float
    compute_penalty: float
    total: float


def calculate_reward(
    truth: int,
    prediction: int,
    cost_macs: float,
    delay_seconds: float = 0.0,
    config: RewardConfig | None = None,
) -> RewardResult:
    config = config or RewardConfig()
    utility = config.correct_emergency if truth == prediction == 1 else (
        config.correct_safe if truth == prediction == 0 else 0.0
    )
    error = (
        config.false_negative_penalty if truth == 1 and prediction == 0 else
        config.false_positive_penalty if truth == 0 and prediction == 1 else 0.0
    )
    delay = max(0.0, delay_seconds) * config.delay_penalty_per_second
    compute = cost_macs / config.compute_reference_macs * config.compute_penalty
    return RewardResult(utility, error, delay, compute, utility - error - delay - compute)
