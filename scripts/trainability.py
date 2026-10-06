import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml

from data.instances import generate
from models.backbone import Policy
from train.config import load_config
from train.losses import reinforce_loss
from train.utils import gradient_norms, seed_all


def main():
    p = argparse.ArgumentParser(description="Quantum gradient sweep at initialization (zero optimizer updates)")
    p.add_argument("--config", default="configs/M3.yaml")
    p.add_argument("--qubits", nargs="+", type=int, default=[4, 6, 8, 10])
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--customers", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--runs", nargs="*", default=[], help="run directories with trainability.csv collected during training")
    p.add_argument("--output", default="reports/trainability")
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in a.qubits:
        for depth in a.depths:
            for seed in a.seeds:
                seed_all(seed)
                cfg = load_config(a.config, [f"model.n_qubits={n}", f"model.depth={depth}", f"customers={a.customers}"])
                if cfg["model"]["backend"] != "quantum":
                    raise ValueError("trainability requires the trainable quantum backend")
                model = Policy(cfg["model"]).to(a.device)
                inst = generate(a.batch_size, a.customers, torch.Generator().manual_seed(12345)).to(a.device)
                loss, _ = reinforce_loss(model(inst), cfg, epoch=6, progress=1.)
                loss.backward()
                gradients = gradient_norms(model)
                pgrad = model.latent.circuit.angles.grad
                rows.append(dict(phase="init", n_qubits=n, depth=depth, seed=seed, epoch=-1, **gradients,
                                 quantum_grad_rms=float(pgrad.square().mean().sqrt()),
                                 gradient_zero_fraction=float((pgrad.abs() < 1e-10).float().mean())))
                print(rows[-1], flush=True)
    for path in a.runs:
        cfg = yaml.safe_load((Path(path)/"config.yaml").read_text())
        records = pd.read_csv(Path(path)/"trainability.csv")
        for record in records.to_dict("records"):
            rows.append(dict(record, phase="training", seed=cfg["seed"], n_qubits=cfg["model"]["n_qubits"], depth=cfg["model"]["depth"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "gradients.csv", index=False)
    table = frame.query("phase == 'init'").groupby(["n_qubits", "depth"]).grad_norm_quantum.agg(["mean", "std"]).reset_index()
    table.to_csv(output / "gradient_table.csv", index=False)
    (output / "gradient_table.tex").write_text(table.to_latex(index=False, float_format="%.6g"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, group in table.groupby("n_qubits"):
        ax.errorbar(group.depth, group["mean"], yerr=group["std"].fillna(0), label=f"{n} qubits", marker="o")
    ax.set(xlabel="circuit depth", ylabel="quantum gradient norm", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "gradient_vs_depth.png", dpi=180)
    plt.close(fig)
    trained = frame.query("phase == 'training'")
    if len(trained):
        fig, ax = plt.subplots(figsize=(7, 4))
        for (n, depth, seed), group in trained.groupby(["n_qubits", "depth", "seed"]):
            ax.plot(group.epoch, group.grad_norm_quantum, label=f"n={n}, L={depth}, seed={seed}")
        ax.set(xlabel="epoch", ylabel="quantum gradient norm", yscale="log")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "gradient_during_training.png", dpi=180)


if __name__ == "__main__":
    main()
