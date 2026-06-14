#!/usr/bin/env python3
"""Validate/materialize a primitive all-model window log.

The expensive audio/model pass remains provider-specific. Export its primitive
S1/S2/S3 columns to CSV or Parquet, then this command validates and adds A0-A14.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echoguard.bandit.features import add_causal_history
from echoguard.bandit.outcomes import materialize_arm_outcomes
from echoguard.bandit.schema import validate_counterfactual_log


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Primitive all-model window table")
    parser.add_argument("--output", default=Path("bandit_logs/window_counterfactuals.parquet"), type=Path)
    parser.add_argument("--small-csv", default=Path("bandit_logs/window_counterfactuals_small.csv"), type=Path)
    parser.add_argument("--sample-rows", default=200, type=int)
    args = parser.parse_args()
    frame = add_causal_history(read_table(args.input))
    validate_counterfactual_log(frame)
    frame = materialize_arm_outcomes(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    frame.head(args.sample_rows).to_csv(args.small_csv, index=False)
    print(f"Wrote {len(frame)} windows to {args.output}")


if __name__ == "__main__":
    main()
