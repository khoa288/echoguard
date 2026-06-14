"""Ground-truth-assisted upper bounds for counterfactual analysis."""

from __future__ import annotations

import pandas as pd

from .arms import ARM_REGISTRY


def cheapest_correct_actions(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in frame.iterrows():
        feasible = []
        for arm_id in ARM_REGISTRY:
            prefix = arm_id.lower()
            if int(row[f"{prefix}_pred_label"]) == int(row["true_emergency_label"]):
                feasible.append(
                    (float(row[f"{prefix}_cost_macs"]), int(row[f"{prefix}_invokes_s3"]), arm_id)
                )
        selected = min(feasible) if feasible else (
            float(row["a6_cost_macs"]), int(row["a6_invokes_s3"]), "A6"
        )
        records.append(
            {
                "stream_id": row["stream_id"],
                "window_index": row["window_index"],
                "oracle_arm": selected[2],
                "oracle_cost_macs": selected[0],
                "oracle_feasible": bool(feasible),
            }
        )
    return pd.DataFrame(records)
