import numpy as np
import pytest

from echoguard.labels import EMERGENCY_CLASS_INDICES, emergency_probability
from echoguard.models.stage2 import STAGE2_CHECKPOINTS
from echoguard.models.stage3 import STAGE3_CHECKPOINTS, summarize_probabilities


def test_emergency_probability_sums_configured_indices():
    probabilities = np.arange(1, 51, dtype=float)
    probabilities /= probabilities.sum()
    assert emergency_probability(probabilities) == pytest.approx(
        probabilities[list(EMERGENCY_CLASS_INDICES)].sum()
    )


def test_stage_roles_are_unambiguous():
    assert set(STAGE2_CHECKPOINTS) == {"bc1", "bc2", "bc3", "bc8"}
    assert set(STAGE3_CHECKPOINTS) == {"mn04", "dymn10"}
    assert "mn04" not in STAGE2_CHECKPOINTS


def test_stage3_summary_normalizes_and_reports_margin():
    output = summarize_probabilities(np.arange(1, 51, dtype=float))
    assert output.probabilities.sum() == pytest.approx(1.0)
    assert output.top_class == 49
    assert output.margin > 0
