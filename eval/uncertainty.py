from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from train.utils import save_json


def analyze(frame, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if "gap_percent" not in frame:
        raise ValueError("gap uncertainty analysis requires evaluation with a matching reference")
    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, column in zip(axes, ["sigma_c", "sigma_latent"]):
        valid = frame[[column, "gap_percent"]].dropna()
        if len(valid) >= 3 and valid[column].nunique() > 1 and valid.gap_percent.nunique() > 1:
            r = spearmanr(valid[column], valid.gap_percent)
            stats[column] = dict(spearman=float(r.statistic), pvalue=float(r.pvalue), count=len(valid))
            ax.scatter(valid[column], valid.gap_percent, s=4, alpha=0.4)
        else:
            stats[column] = dict(spearman=None, pvalue=None, reason="absent or constant values")
        ax.set(xlabel=column, ylabel="realised gap (%)")
    fig.tight_layout()
    fig.savefig(output / "sigma_vs_gap.png", dpi=180)
    plt.close(fig)
    # Difficulty head is trained on MEAN rollout cost, not best-of-P cost or gaps.
    error = (frame.mean_rollout_cost-frame.mu_c).to_numpy()
    sigma = frame.sigma_c.to_numpy()
    levels = np.arange(0.1, 1., 0.1)
    coverage = [(np.abs(error) <= norm.ppf((1+level)/2)*sigma).mean() for level in levels]
    calibration = pd.DataFrame(dict(nominal_coverage=levels, observed_coverage=coverage))
    calibration.to_csv(output / "calibration.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(levels, coverage, "o-", label="mean cost coverage")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="nominal coverage", ylabel="observed coverage", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=180)
    plt.close(fig)
    stats["mean_cost_nll_without_constant"] = float(np.mean(np.log(sigma) + 0.5*(error/sigma)**2))
    stats["interpretation"] = "sigma_c models instance mean-cost uncertainty; it need not match within-instance rollout spread or gap"
    save_json(output / "uncertainty.json", stats)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-instance", required=True)
    p.add_argument("--output", default="reports/uncertainty")
    a = p.parse_args()
    print(analyze(pd.read_csv(a.per_instance), a.output))


if __name__ == "__main__":
    main()
