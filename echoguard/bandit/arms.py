"""Deterministic complete inference paths (arms A0-A14)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from echoguard.costs import CostModel


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    name: str
    stage2: str | None = None
    stage3: str | None = None
    direct_stage3: bool = False


@dataclass(frozen=True)
class ArmResult:
    arm_id: str
    predicted_label: int
    score: float
    cost_macs: float
    invoked_stage2: bool
    invoked_stage3: bool
    stage2_model: str | None
    stage3_model: str | None
    escalated: bool


ARM_REGISTRY: Mapping[str, ArmSpec] = {
    "A0": ArmSpec("A0", "s1_stop_safe"),
    "A1": ArmSpec("A1", "bc1_only", stage2="bc1"),
    "A2": ArmSpec("A2", "bc2_only", stage2="bc2"),
    "A3": ArmSpec("A3", "bc3_only", stage2="bc3"),
    "A4": ArmSpec("A4", "bc8_only", stage2="bc8"),
    "A5": ArmSpec("A5", "mn04_direct", stage3="mn04", direct_stage3=True),
    "A6": ArmSpec("A6", "dymn10_direct", stage3="dymn10", direct_stage3=True),
    "A7": ArmSpec("A7", "bc1_to_mn04", "bc1", "mn04"),
    "A8": ArmSpec("A8", "bc2_to_mn04", "bc2", "mn04"),
    "A9": ArmSpec("A9", "bc3_to_mn04", "bc3", "mn04"),
    "A10": ArmSpec("A10", "bc8_to_mn04", "bc8", "mn04"),
    "A11": ArmSpec("A11", "bc1_to_dymn10", "bc1", "dymn10"),
    "A12": ArmSpec("A12", "bc2_to_dymn10", "bc2", "dymn10"),
    "A13": ArmSpec("A13", "bc3_to_dymn10", "bc3", "dymn10"),
    "A14": ArmSpec("A14", "bc8_to_dymn10", "bc8", "dymn10"),
}


def evaluate_arm(
    arm_id: str,
    row: Mapping[str, float],
    costs: CostModel | None = None,
    danger_threshold: float = 0.30,
    uncertainty_low: float = 0.25,
    uncertainty_high: float = 0.65,
    expert_threshold: float = 0.50,
) -> ArmResult:
    spec = ARM_REGISTRY[arm_id]
    costs = costs or CostModel()
    if arm_id == "A0":
        return ArmResult(arm_id, 0, 0.0, costs.stage1, False, False, None, None, False)
    if spec.direct_stage3:
        score = float(row[f"{spec.stage3}_emergency_probability"])
        return ArmResult(
            arm_id, int(score >= expert_threshold), score,
            costs.stage1 + costs.model(spec.stage3), False, True, None, spec.stage3, True,
        )
    p_danger = float(row[f"{spec.stage2}_p_danger"])
    base_cost = costs.stage1 + costs.model(spec.stage2)
    if spec.stage3 is None:
        return ArmResult(
            arm_id, int(p_danger >= danger_threshold), p_danger,
            base_cost, True, False, spec.stage2, None, False,
        )
    escalate = p_danger >= danger_threshold or uncertainty_low <= p_danger <= uncertainty_high
    if not escalate:
        return ArmResult(arm_id, 0, p_danger, base_cost, True, False, spec.stage2, None, False)
    expert_score = float(row[f"{spec.stage3}_emergency_probability"])
    return ArmResult(
        arm_id, int(expert_score >= expert_threshold), expert_score,
        base_cost + costs.model(spec.stage3), True, True, spec.stage2, spec.stage3, True,
    )
