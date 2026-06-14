"""Counterfactual row construction independent of model implementation details."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd

from echoguard.costs import CostModel
from echoguard.labels import EMERGENCY_CLASS_SET_VERSION

from .config import BanditConfig
from .features import add_causal_history
from .outcomes import materialize_arm_outcomes
from .schema import validate_counterfactual_log

WindowProvider = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def build_counterfactual_log(
    windows: Iterable[Mapping[str, Any]],
    provider: WindowProvider,
    config: BanditConfig | None = None,
    costs: CostModel | None = None,
) -> pd.DataFrame:
    """Evaluate a provider once per decision window, then simulate every arm."""
    config = config or BanditConfig()
    costs = costs or CostModel()
    rows = []
    for window in windows:
        row = {**window, **provider(window)}
        row.update(
            {
                "schema_version": config.schema_version,
                "config_hash": config.config_hash,
                "cost_model_version": costs.version,
                "emergency_class_set_version": EMERGENCY_CLASS_SET_VERSION,
            }
        )
        rows.append(row)
    frame = add_causal_history(pd.DataFrame(rows))
    validate_counterfactual_log(frame)
    return materialize_arm_outcomes(frame)
