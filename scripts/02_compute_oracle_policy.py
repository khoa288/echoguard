#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echoguard.bandit.oracle import cheapest_correct_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", default=Path("bandit_results/oracle_actions.parquet"), type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.log)
    oracle = cheapest_correct_actions(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    oracle.to_parquet(args.output, index=False)
    summary = oracle.groupby("oracle_arm").agg(windows=("oracle_arm", "size"), mean_cost=("oracle_cost_macs", "mean"))
    summary.to_csv(args.output.parent / "oracle_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
