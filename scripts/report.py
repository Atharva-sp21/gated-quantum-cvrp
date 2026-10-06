from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.statistics import holm_correction, paired_comparison
from train.utils import save_json


CONDITION = ["train_customers", "customers", "mode", "augmentation", "samples", "dataset_fingerprint", "cost_convention", "noise"]
COUNTS = ["policy_classical_parameters", "policy_quantum_parameters", "policy_total_parameters"]


def read_results(root):
    records, frames = [], {}
    for path in sorted(Path(root).rglob("summary.json")):
        summary = json.loads(path.read_text())
        if "variant" not in summary or "training_seed" not in summary:
            continue
        summary.setdefault("cost_convention", "euclidean")
        summary["noise"] = json.dumps(summary.get("noise", {}), sort_keys=True)
        summary["path"] = str(path.parent)
        counts = summary.get("parameter_counts", {})
        for field, source in zip(COUNTS, ["classical", "quantum", "total"]):
            summary[field] = counts.get(source)
        records.append({key: summary.get(key) for key in CONDITION+COUNTS+["variant", "training_seed", "mean_cost", "mean_gap_percent", "inference_seconds_per_instance", "path"]})
        frames[str(path.parent)] = pd.read_csv(path.parent/"per_instance.csv")
    if not records:
        raise ValueError("no evaluation summaries with variant and training_seed found")
    return pd.DataFrame(records), frames


def build_reports(root, output, baseline="M0", allow_incomplete=False, run_root=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, frames = read_results(root)
    grouped = records.groupby(CONDITION+["variant"], dropna=False)
    rows = []
    for key, group in grouped:
        if group.training_seed.nunique() != len(group):
            raise ValueError("duplicate seed/variant/condition; separate evaluation runs")
        if len(group) < 5 and not allow_incomplete:
            raise ValueError(f"{key[-1]} has {len(group)} seeds; >=5 required (or --allow-incomplete)")
        row = dict(zip(CONDITION+["variant"], key))
        row.update(n_seeds=len(group), mean_cost=group.mean_cost.mean(), std_cost=group.mean_cost.std(ddof=1),
                   mean_gap_percent=group.mean_gap_percent.mean(), std_gap_percent=group.mean_gap_percent.std(ddof=1),
                   inference_seconds_per_instance=group.inference_seconds_per_instance.mean())
        for field in COUNTS:
            if group[field].nunique() > 1:
                raise ValueError("different parameter counts within one variant; use distinct variant names")
            row[field] = group[field].iloc[0]
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output/"results.csv", index=False)
    display = table[["variant", "train_customers", "customers", "mode", "augmentation", "samples", "n_seeds"]].copy()
    display["cost mean ± std"] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(table.mean_cost, table.std_cost)]
    display["gap (%) mean ± std"] = [f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else "reference unavailable"
                                    for m, s in zip(table.mean_gap_percent, table.std_gap_percent)]
    (output/"results.tex").write_text(display.to_latex(index=False, escape=True))
    efficiency = table[["variant", "train_customers", "customers", "mode", "augmentation", "samples"]+COUNTS+["inference_seconds_per_instance"]]
    efficiency.to_csv(output/"inference_efficiency.csv", index=False)
    (output/"inference_efficiency.tex").write_text(efficiency.to_latex(index=False, float_format="%.6g"))
    comparisons = []
    for condition, group in records.groupby(CONDITION, dropna=False):
        base = group[group.variant == baseline]
        if base.empty:
            continue
        for variant, candidate in group[group.variant != baseline].groupby("variant"):
            if set(base.training_seed) != set(candidate.training_seed):
                raise ValueError("paired comparisons require identical training seed sets")
            arrays_base, arrays_candidate = [], []
            for seed in sorted(base.training_seed):
                bf = frames[base[base.training_seed == seed].path.iloc[0]].sort_values("instance_id")
                cf = frames[candidate[candidate.training_seed == seed].path.iloc[0]].sort_values("instance_id")
                if not np.array_equal(bf.instance_id, cf.instance_id):
                    raise ValueError("per-instance IDs mismatch")
                if ("reference" in bf) != ("reference" in cf) or ("reference" in bf and not np.allclose(bf.reference, cf.reference, rtol=0, atol=1e-9)):
                    raise ValueError("paired references mismatch")
                arrays_base.append(bf.cost.to_numpy())
                arrays_candidate.append(cf.cost.to_numpy())
            stats = paired_comparison(np.mean(arrays_base, axis=0), np.mean(arrays_candidate, axis=0))
            comparisons.append(dict(zip(CONDITION, condition), baseline=baseline, variant=variant,
                                    n_seeds=len(arrays_base), **stats))
    if comparisons:
        adjusted = holm_correction([row["pvalue"] for row in comparisons])
        for row, p in zip(comparisons, adjusted):
            row["pvalue_holm"] = p
        stats = pd.DataFrame(comparisons)
        stats.to_csv(output/"statistics.csv", index=False)
        (output/"statistics.tex").write_text(stats[["variant", "customers", "mode", "augmentation", "samples", "pvalue", "pvalue_holm", "rank_biserial_improvement", "mean_cost_delta"]].to_latex(index=False, float_format="%.5g"))
    fig, ax = plt.subplots(figsize=(8, 4))
    for (variant, mode, aug, samples), group in table.groupby(["variant", "mode", "augmentation", "samples"]):
        group = group.sort_values("customers")
        ax.errorbar(group.customers, group.mean_gap_percent, yerr=group.std_gap_percent.fillna(0),
                    marker="o", label=f"{variant} {mode} x{aug} S={samples}")
    ax.set(xlabel="test customers", ylabel="gap vs cached reference (%)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output/"gap_vs_size.png", dpi=180)
    plt.close(fig)
    if run_root:
        plot_training(run_root, output)
    save_json(output/"report_notes.json", dict(exploratory=allow_incomplete,
               seed_summary="sample standard deviation of seed-level mean metrics",
               paired_test="Wilcoxon over per-instance costs averaged across matched seeds; no pooling seed-instance pairs",
               effect="rank-biserial: positive means candidate improves on baseline",
               multiplicity="Holm over all emitted comparisons", no_quantum_benefit_assumed=True))
    return table


def plot_training(root, output):
    files = sorted(Path(root).glob("*/metrics.csv"))
    if not files:
        raise ValueError("no training logs found")
    import yaml
    rows = []
    for path in files:
        cfg = yaml.safe_load((path.parent/"config.yaml").read_text())
        frame = pd.read_csv(path)
        metadata = json.loads((path.parent/"metadata.json").read_text())
        rows.append(dict(variant=cfg["name"], train_customers=cfg["customers"], seed=cfg["seed"],
                         mean_training_seconds_per_epoch=float(frame.epoch_seconds.mean()),
                         **metadata["parameter_counts"]))
    training_efficiency = pd.DataFrame(rows)
    training_efficiency.to_csv(output/"training_efficiency.csv", index=False)
    (output/"training_efficiency.tex").write_text(training_efficiency.to_latex(index=False, float_format="%.6g"))
    panels = [("training_curves.png", ["cost_mean", "validation_cost", "validation_gap_percent"]),
              ("latent_diagnostics.png", ["alpha", "contribution", "active_dimensions", "sigma_latent"]),
              ("training_losses.png", ["loss_policy", "loss_difficulty", "loss_kl", "loss_recon"]),
              ("efficiency.png", ["epoch_seconds", "grad_norm_quantum", "grad_norm_classical"])]
    for name, columns in panels:
        fig, axes = plt.subplots(1, len(columns), figsize=(4*len(columns), 3.5), squeeze=False)
        for path in files:
            frame = pd.read_csv(path)
            for ax, column in zip(axes[0], columns):
                if column in frame:
                    ax.plot(frame.epoch, frame[column], alpha=0.6, label=path.parent.name)
                ax.set(xlabel="epoch", ylabel=column)
        axes[0][0].legend(fontsize=5)
        fig.tight_layout()
        fig.savefig(output/name, dpi=180)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Actual measured tables, paired statistics, and figures")
    p.add_argument("--evaluations", default="reports/main")
    p.add_argument("--runs", default="runs")
    p.add_argument("--output", default="reports/paper")
    p.add_argument("--baseline", default="M0")
    p.add_argument("--allow-incomplete", action="store_true", help="exploratory outputs with fewer than five seeds")
    a = p.parse_args()
    print(build_reports(a.evaluations, a.output, a.baseline, a.allow_incomplete, a.runs))


if __name__ == "__main__":
    main()
