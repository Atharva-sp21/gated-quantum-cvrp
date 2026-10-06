import argparse
from pathlib import Path

import numpy as np
import torch

from data.instances import generate, load_instances
from eval.evaluate import evaluate_instances, load_checkpoint, write_evaluation
from train.utils import seed_all


def main():
    p = argparse.ArgumentParser(description="Solve CVRP with a trained checkpoint; saves explicit depot-delimited routes")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", help="saved .npz; otherwise generates instances")
    p.add_argument("--customers", type=int, default=100)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--capacity", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=456)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--output", default="reports/inference")
    a = p.parse_args()
    seed_all(a.seed)
    model, cfg = load_checkpoint(a.checkpoint, a.device)
    inst = load_instances(a.dataset) if a.dataset else generate(a.count, a.customers, torch.Generator().manual_seed(a.seed), capacity=a.capacity)
    result = evaluate_instances(model, inst, augmentation=a.augment, return_routes=True)
    result["routes"] = [np.concatenate([[0], route, [0]]) for route in result["routes"]]
    write_evaluation(result, a.output, variant=cfg["name"], training_seed=cfg["seed"], train_customers=cfg["customers"])
    print(result["per_instance"][["instance_id", "cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
