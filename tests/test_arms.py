import pytest

from echoguard.bandit.arms import ARM_REGISTRY, evaluate_arm
from echoguard.costs import CostModel

ROW = {
    "bc1_p_danger": 0.1,
    "bc2_p_danger": 0.2,
    "bc3_p_danger": 0.4,
    "bc8_p_danger": 0.8,
    "mn04_emergency_probability": 0.7,
    "dymn10_emergency_probability": 0.2,
}


def test_arm_ids_are_stable_and_unique():
    assert list(ARM_REGISTRY) == [f"A{i}" for i in range(15)]


def test_conditional_arm_only_charges_expert_when_escalated():
    costs = CostModel()
    stopped = evaluate_arm("A7", ROW, costs)
    escalated = evaluate_arm("A9", ROW, costs)
    assert not stopped.invoked_stage3
    assert stopped.cost_macs == pytest.approx(costs.stage1 + costs.model("bc1"))
    assert escalated.invoked_stage3
    assert escalated.cost_macs == pytest.approx(
        costs.stage1 + costs.model("bc3") + costs.model("mn04")
    )


def test_direct_expert_uses_stage3_not_stage2():
    result = evaluate_arm("A5", ROW)
    assert result.invoked_stage3
    assert not result.invoked_stage2
    assert result.predicted_label == 1
