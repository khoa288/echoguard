#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echoguard.bandit.arms import ARM_REGISTRY
from echoguard.bandit.eval_policy import replay_policy
from echoguard.bandit.policies import EpsilonGreedyPolicy, FixedPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("bandit_results"), type=Path)
    args = parser.parse_args()
    frame = pd.read_parquet(args.log)
    policies = {f"fixed_{arm}": FixedPolicy(arm) for arm in ARM_REGISTRY}
    policies["epsilon_greedy"] = EpsilonGreedyPolicy(seed=2026)
    summaries = []
    traces = []
    for name, policy in policies.items():
        trace, metrics = replay_policy(frame, policy)
        trace["policy"] = name
        traces.append(trace)
        summaries.append({"policy": name, **metrics})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.output_dir / "policy_eval_summary.csv", index=False)
    pd.concat(traces).to_parquet(args.output_dir / "policy_replay_traces.parquet", index=False)


if __name__ == "__main__":
    main()
