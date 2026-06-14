"""Counterfactual log schema and validation."""

from __future__ import annotations

import pandas as pd

ROW_KEY = ["stream_id", "scenario", "seed", "window_index", "time_sec"]
GROUND_TRUTH_COLUMNS = [
    "true_emergency_label",
    "true_event_classes_overlapping_window",
    "seconds_since_event_onset",
]
LEVEL1_FEATURES = [
    "rms_db",
    "noise_floor_db",
    "delta_db",
    "spectral_flux",
    "band_energy_0_300",
    "band_energy_300_1200",
    "band_energy_1200_3000",
    "band_energy_3000_nyquist",
    "sleep_flag",
    "s1_a_trigger",
    "s1_b_trigger",
    "s1_c_trigger",
    "recent_s1_trigger_count",
    "recent_acoustic_volatility",
    "time_since_last_alarm",
    "previous_alarm_state",
]
MODEL_OUTPUT_COLUMNS = [
    "bc1_p_danger", "bc2_p_danger", "bc3_p_danger", "bc8_p_danger",
    "mn04_emergency_probability", "mn04_top_class", "mn04_entropy", "mn04_margin",
    "dymn10_emergency_probability", "dymn10_top_class", "dymn10_entropy",
    "dymn10_margin",
]
PROVENANCE_COLUMNS = [
    "schema_version", "config_hash", "cost_model_version",
    "emergency_class_set_version",
]
REQUIRED_COLUMNS = (
    ROW_KEY + GROUND_TRUTH_COLUMNS + LEVEL1_FEATURES + MODEL_OUTPUT_COLUMNS + PROVENANCE_COLUMNS
)


def validate_counterfactual_log(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing counterfactual columns: {missing}")
    if frame.duplicated(ROW_KEY).any():
        raise ValueError("Counterfactual log contains duplicate row keys")
    probability_columns = [column for column in MODEL_OUTPUT_COLUMNS if "probability" in column or "p_danger" in column]
    for column in probability_columns:
        if not frame[column].between(0, 1).all():
            raise ValueError(f"{column} contains values outside [0, 1]")
    for _, stream in frame.groupby("stream_id", sort=False):
        if not stream.sort_values("window_index")["window_index"].is_monotonic_increasing:
            raise ValueError("Window indices must increase within each stream")
