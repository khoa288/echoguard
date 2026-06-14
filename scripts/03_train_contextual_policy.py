#!/usr/bin/env python3
"""Chronological LinUCB replay over a full-information counterfactual log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from echoguard.bandit.eval_policy import replay_policy
from echoguard.bandit.features import LEVEL1_FEATURES
from echoguard.bandit.policies import LinUCBPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", default=Path("bandit_results/linucb_trace.parquet"), type=Path)
    parser.add_argument("--alpha", default=1.0, type=float)
    args = parser.parse_args()
    frame = pd.read_parquet(args.log)
    policy = LinUCBPolicy(len(LEVEL1_FEATURES), alpha=args.alpha)
    trace, metrics = replay_policy(frame, policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace.to_parquet(args.output, index=False)
    (args.output.parent / "linucb_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
