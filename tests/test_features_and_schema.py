import pandas as pd
import pytest

from echoguard.bandit.features import add_causal_history, level1_matrix


def test_history_is_causal_and_resets_per_stream():
    frame = pd.DataFrame(
        {
            "stream_id": ["x", "x", "x", "y"],
            "s1_c_trigger": [1, 0, 1, 1],
            "delta_db": [1.0, 3.0, 9.0, 100.0],
        }
    )
    result = add_causal_history(frame, history_windows=2)
    assert result["recent_s1_trigger_count"].tolist() == [0, 1, 1, 0]
    assert result.loc[3, "recent_acoustic_volatility"] == 0


def test_level1_matrix_rejects_missing_features():
    with pytest.raises(ValueError, match="Missing Level-1"):
        level1_matrix(pd.DataFrame({"true_emergency_label": [1], "bc1_p_danger": [0.9]}))
