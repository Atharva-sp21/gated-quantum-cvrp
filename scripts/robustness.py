import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data.instances import load_instances
from eval.evaluate import evaluate_instances, load_checkpoint, load_reference, write_evaluation
from train.utils import seed_all


def main():
    p = argparse.ArgumentParser(description="Finite-shot and depolarizing inference sweep")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--reference")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", default="reports/robustness")
    a = p.parse_args()
    model, cfg = load_checkpoint(a.checkpoint, a.device)
    if cfg["model"]["backend"] not in {"quantum", "frozen_random_quantum"}:
        p.error("robustness sweep requires quantum backend")
    if cfg["model"]["n_qubits"] > 8:
        p.error("depolarizing density matrix sweep requires <=8 qubits")
    inst = load_instances(a.dataset)
    reference = load_reference(a.reference, inst) if a.reference else None
    output = Path(a.output)
    rows = []
    for shots in [None, 1024, 128]:
        for probability in [0., 0.001, 0.01]:
            seed_all(a.seed)
            noise = dict(shots=shots, depolarizing=probability)
            result = evaluate_instances(model, inst, batch_size=a.batch_size, reference=reference, noise=noise)
            write_evaluation(result, output / f"shots_{shots}_p_{probability}", variant=cfg["name"], training_seed=cfg["seed"])
            rows.append(dict(shots="inf" if shots is None else str(shots), depolarizing=probability,
                             cost=result["summary"]["mean_cost"], gap_percent=result["summary"].get("mean_gap_percent")))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "robustness.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    baseline = frame.query("shots == 'inf' and depolarizing == 0").cost.iloc[0]
    for shots, group in frame.groupby("shots"):
        ax.plot(group.depolarizing, 100*(group.cost/baseline-1), "o-", label=f"shots={shots}")
    ax.set(xlabel="depolarizing probability per qubit per layer", ylabel="cost degradation (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "noise_robustness.png", dpi=180)


if __name__ == "__main__":
    main()
