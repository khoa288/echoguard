#!/usr/bin/env python3
from echoguard.bandit.plot_policy_results import plot_pareto

if __name__ == "__main__":
    plot_pareto(
        "bandit_results/policy_eval_summary.csv",
        "bandit_results/figures/compute_recall_pareto.png",
    )
