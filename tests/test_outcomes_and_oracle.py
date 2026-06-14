import pandas as pd

from echoguard.bandit.oracle import cheapest_correct_actions
from echoguard.bandit.outcomes import event_detection_metrics, materialize_arm_outcomes


def primitive_frame():
    return pd.DataFrame(
        [
            {
                "stream_id": "s",
                "scenario": "quiet",
                "seed": 1,
                "window_index": 0,
                "time_sec": 0.0,
                "true_emergency_label": 0,
                "bc1_p_danger": 0.1,
                "bc2_p_danger": 0.1,
                "bc3_p_danger": 0.1,
                "bc8_p_danger": 0.1,
                "mn04_emergency_probability": 0.8,
                "dymn10_emergency_probability": 0.9,
            }
        ]
    )


def test_oracle_deterministically_prefers_cheapest_correct_arm():
    frame = materialize_arm_outcomes(primitive_frame())
    oracle = cheapest_correct_actions(frame)
    assert oracle.loc[0, "oracle_arm"] == "A0"


def test_event_ttd_uses_first_detection_after_onset():
    trace = pd.DataFrame(
        {
            "stream_id": ["s"] * 4,
            "time_sec": [0.0, 0.5, 1.0, 1.5],
            "predicted_label": [1, 0, 1, 1],
        }
    )
    events = pd.DataFrame(
        [{"stream_id": "s", "event_id": 1, "onset_sec": 0.75, "offset_sec": 1.5}]
    )
    result = event_detection_metrics(trace, events)
    assert result.loc[0, "ttd_sec"] == 0.25
