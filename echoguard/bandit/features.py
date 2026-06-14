"""Causal feature construction with explicit leakage boundaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import LEVEL1_FEATURES

LEVEL2_EXTRA_FEATURES = [
    "s2_p_danger", "s2_threshold_margin", "s2_entropy",
    "s2_model_bc1", "s2_model_bc2", "s2_model_bc3", "s2_model_bc8",
    "recent_s2_p_danger",
]
LEVEL2_FEATURES = LEVEL1_FEATURES + LEVEL2_EXTRA_FEATURES


def add_causal_history(frame: pd.DataFrame, history_windows: int = 10) -> pd.DataFrame:
    result = frame.copy()
    groups = result.groupby("stream_id", sort=False)
    trigger = result["s1_c_trigger"].astype(float)
    result["recent_s1_trigger_count"] = groups[trigger.name].transform(
        lambda values: values.shift(1).rolling(history_windows, min_periods=1).sum()
    ).fillna(0.0)
    result["recent_acoustic_volatility"] = groups["delta_db"].transform(
        lambda values: values.shift(1).rolling(history_windows, min_periods=1).std(ddof=0)
    ).fillna(0.0)
    return result


def level1_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(LEVEL1_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing Level-1 features: {missing}")
    return frame[LEVEL1_FEATURES].fillna(0.0).to_numpy(dtype=float)


def level2_row(row: pd.Series, model: str, probability: float, recent: float = 0.0) -> np.ndarray:
    if model not in {"bc1", "bc2", "bc3", "bc8"}:
        raise ValueError(f"Not a Stage-2 model: {model}")
    values = [float(row.get(name, 0.0)) for name in LEVEL1_FEATURES]
    p = float(probability)
    entropy = -(p * np.log(max(p, 1e-12)) + (1 - p) * np.log(max(1 - p, 1e-12)))
    values.extend(
        [p, p - 0.5, entropy]
        + [float(model == candidate) for candidate in ("bc1", "bc2", "bc3", "bc8")]
        + [float(recent)]
    )
    return np.asarray(values, dtype=float)
