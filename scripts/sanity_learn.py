import argparse
import csv
from pathlib import Path

import torch

from data.instances import generate
from data.reference import reference_cache
from eval.evaluate import evaluate_instances
from models.backbone import Policy
from train.config import load_config
from train.losses import reinforce_loss
from train.utils import seed_all


def main():
    p = argparse.ArgumentParser(description="OPT-IN M0 learning sanity test; actually trains a small model")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="reports/sanity")
    a = p.parse_args()
    seed_all(a.seed)
    cfg = load_config("configs/M0.yaml", ["customers=20", "model.d_model=64", "model.heads=4",
                                         "model.ffn=128", "model.layers=3", "training.lr=0.001"])
    model = Policy(cfg["model"]).to(a.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    fixed = generate(16, 20, torch.Generator().manual_seed(999)).to(a.device)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    reference = reference_cache(fixed.to("cpu"), output / "sanity_reference.npz", time_limit=0.2, seed=999)
    records = []
    for step in range(a.steps+1):
        if step % 25 == 0 or step == a.steps:
            result = evaluate_instances(model, fixed, batch_size=16, reference=reference)
            records.append(dict(step=step, cost=result["summary"]["mean_cost"], gap=result["summary"]["mean_gap_percent"]))
            print(records[-1], flush=True)
        if step == a.steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = reinforce_loss(model(fixed), cfg, epoch=6, progress=step/max(1, a.steps))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    with (output / "sanity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if records[-1]["cost"] >= records[0]["cost"] or records[-1]["gap"] >= records[0]["gap"]:
        raise RuntimeError("learning sanity failed: cost/gap did not decrease; inspect report, do not claim learning")
    print("Measured sanity pass: cost and cached-reference gap decreased. This is not a benchmark result.")


if __name__ == "__main__":
    main()
