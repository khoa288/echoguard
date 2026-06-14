"""Chronological policy replay and deployment-oriented metrics."""

from __future__ import annotations

import pandas as pd

from .features import level1_matrix
from .policies import Policy
from .rewards import calculate_reward


def replay_policy(frame: pd.DataFrame, policy: Policy) -> tuple[pd.DataFrame, dict[str, float]]:
    ordered = frame.sort_values(["stream_id", "window_index"]).copy()
    contexts = level1_matrix(ordered)
    policy.reset()
    trace = []
    for context, (_, row) in zip(contexts, ordered.iterrows()):
        action = policy.select_action(context)
        prefix = action.lower()
        prediction = int(row[f"{prefix}_pred_label"])
        cost = float(row[f"{prefix}_cost_macs"])
        reward = calculate_reward(int(row["true_emergency_label"]), prediction, cost)
        policy.observe(context, action, reward.total)
        trace.append(
            {
                "stream_id": row["stream_id"],
                "scenario": row["scenario"],
                "window_index": row["window_index"],
                "time_sec": row["time_sec"],
                "true_emergency_label": int(row["true_emergency_label"]),
                "action": action,
                "predicted_label": prediction,
                "score": float(row[f"{prefix}_score"]),
                "cost_macs": cost,
                "invokes_s3": int(row[f"{prefix}_invokes_s3"]),
                "reward": reward.total,
            }
        )
    output = pd.DataFrame(trace)
    truth = output["true_emergency_label"].to_numpy()
    prediction = output["predicted_label"].to_numpy()
    tp = int(((truth == 1) & (prediction == 1)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    metrics = {
        "precision": precision,
        "recall": recall,
        "emergency_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "average_macs_per_window": float(output["cost_macs"].mean()),
        "s3_duty": float(output["invokes_s3"].mean()),
        "total_reward": float(output["reward"].sum()),
        "false_positives": float(fp),
        "false_negatives": float(fn),
    }
    return output, metrics
