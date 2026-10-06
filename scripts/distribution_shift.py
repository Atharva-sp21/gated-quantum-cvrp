import argparse
from pathlib import Path

import torch

from data.instances import generate, load_instances, save_instances
from data.reference import reference_cache
from eval.evaluate import evaluate_instances, load_checkpoint, load_reference, write_evaluation
from eval.uncertainty import analyze
from train.utils import seed_all


def main():
    p = argparse.ArgumentParser(description="Evaluate separately trained MVE/no-MVE models on identical shifted instances")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--customers", type=int, default=100)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--time-limit", type=float, default=1.)
    p.add_argument("--build-references", action="store_true")
    p.add_argument("--datasets", default="datasets/shifted")
    p.add_argument("--output", default="reports/shifted")
    a = p.parse_args()
    for name, distribution, lo, hi in [("clustered", "clustered", 1, 9), ("higher_demand", "uniform", 3, 12)]:
        path = Path(a.datasets) / f"{name}_{a.customers}.npz"
        if not path.exists():
            inst = generate(a.count, a.customers, torch.Generator().manual_seed(701),
                            distribution=distribution, demand_low=lo, demand_high=hi)
            save_instances(path, inst, seed=701, count=a.count, split="shifted_test")
        inst = load_instances(path)
        expected = dict(customers=a.customers, distribution=distribution, demand_low=lo, demand_high=hi)
        if len(inst.coords) != a.count or any(inst.metadata[k] != v for k, v in expected.items()):
            raise ValueError("existing shifted dataset has incompatible metadata")
        ref_path = path.with_name(path.stem+"_reference.npz")
        if a.build_references:
            reference_cache(inst, ref_path, a.time_limit, seed=701)
        reference = load_reference(ref_path, inst)
        mve_flags = set()
        for checkpoint in a.checkpoints:
            seed_all(123)
            model, cfg = load_checkpoint(checkpoint, a.device)
            mve_flags.add(cfg["model"]["mve_latent"])
            result = evaluate_instances(model, inst, reference=reference)
            directory = Path(a.output) / Path(checkpoint).parent.name / name
            write_evaluation(result, directory, variant=cfg["name"], training_seed=cfg["seed"],
                             train_customers=cfg["customers"], shift=name, mve_latent=cfg["model"]["mve_latent"])
            analyze(result["per_instance"], directory / "uncertainty")
        if len(mve_flags) < 2:
            print("Only one latent MVE setting supplied; no with/without-MVE comparison was computed.")


if __name__ == "__main__":
    main()
