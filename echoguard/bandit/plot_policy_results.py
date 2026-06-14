"""Plots from saved result tables; never reruns inference."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_pareto(summary_path: str | Path, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_csv(summary_path)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(frame["average_macs_per_window"], frame["recall"])
    for row in frame.to_dict("records"):
        axis.annotate(row["policy"], (row["average_macs_per_window"], row["recall"]))
    axis.set_xlabel("Average MACs per decision window")
    axis.set_ylabel("Emergency recall")
    axis.set_title("EchoGuard-Bandit compute/recall frontier")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
