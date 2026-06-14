"""Materialize path counterfactuals and event-level timing metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .arms import ARM_REGISTRY, evaluate_arm


def materialize_arm_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for arm_id in ARM_REGISTRY:
        values = [evaluate_arm(arm_id, row) for row in frame.to_dict("records")]
        prefix = arm_id.lower()
        result[f"{prefix}_pred_label"] = [value.predicted_label for value in values]
        result[f"{prefix}_score"] = [value.score for value in values]
        result[f"{prefix}_cost_macs"] = [value.cost_macs for value in values]
        result[f"{prefix}_invokes_s3"] = [int(value.invoked_stage3) for value in values]
        result[f"{prefix}_false_negative"] = (
            (result["true_emergency_label"] == 1) & (result[f"{prefix}_pred_label"] == 0)
        ).astype(int)
        result[f"{prefix}_false_positive"] = (
            (result["true_emergency_label"] == 0) & (result[f"{prefix}_pred_label"] == 1)
        ).astype(int)
    return result


def event_detection_metrics(
    trace: pd.DataFrame,
    events: pd.DataFrame,
    prediction_column: str = "predicted_label",
) -> pd.DataFrame:
    records = []
    for event in events.to_dict("records"):
        candidates = trace[
            (trace["stream_id"] == event["stream_id"])
            & (trace["time_sec"] >= event["onset_sec"])
            & (trace["time_sec"] <= event["offset_sec"])
            & (trace[prediction_column] == 1)
        ]
        detected = not candidates.empty
        first = float(candidates["time_sec"].min()) if detected else np.nan
        records.append(
            {
                **event,
                "detected": detected,
                "detection_time_sec": first,
                "ttd_sec": first - float(event["onset_sec"]) if detected else np.nan,
                "missed": not detected,
            }
        )
    return pd.DataFrame(records)
