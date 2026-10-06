import argparse
from pathlib import Path

import numpy as np

from data.cvrplib import load_cvrplib
from eval.evaluate import evaluate_instances, load_checkpoint, write_evaluation
from train.utils import seed_all
from envs.cvrp import recompute_cost
import torch


def main():
    p = argparse.ArgumentParser(description="Zero-shot CVRPLIB A/B/P/X generalisation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--instances", nargs="+", required=True, help="local .vrp paths; shell glob supported")
    p.add_argument("--solution-dir", help="matching <instance-stem>.sol best-known solutions")
    p.add_argument("--rounding", choices=["nearest", "floor", "ceil", "none", "explicit"], default="nearest")
    p.add_argument("--reference-rounding", choices=["nearest", "floor", "ceil", "none", "explicit"], default="nearest",
                   help="cost convention of supplied .sol files; must match comparison convention")
    p.add_argument("--device", default="cuda")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", default="reports/cvrplib")
    a = p.parse_args()
    if a.solution_dir and a.reference_rounding != a.rounding:
        p.error("solution reference convention must match --rounding")
    seed_all(a.seed)
    model, cfg = load_checkpoint(a.checkpoint, a.device)
    for path in a.instances:
        solution = Path(a.solution_dir) / (Path(path).stem + ".sol") if a.solution_dir else None
        if solution is not None and not solution.exists():
            raise FileNotFoundError(solution)
        inst, rounded, unrounded = load_cvrplib(path, solution, a.rounding)
        ref = np.array([inst.metadata["best_known"]]) if solution else None
        result = evaluate_instances(model, inst, batch_size=1, augmentation=a.augment, reference=ref,
                                    metric_distance=rounded, unrounded_distance=unrounded, return_routes=True)
        route = torch.tensor(result["routes"][0])[None, None]
        nearest = torch.floor(unrounded+0.5)
        result["per_instance"]["rounded_cost"] = float(recompute_cost(inst.coords, route, nearest))
        result["summary"]["mean_unrounded_cost"] = float(result["per_instance"].unrounded_cost.mean())
        result["summary"]["mean_rounded_cost"] = float(result["per_instance"].rounded_cost.mean())
        write_evaluation(result, Path(a.output) / Path(path).stem, variant=cfg["name"],
                         training_seed=cfg["seed"], train_customers=cfg["customers"], evaluation_seed=a.seed,
                         cost_convention=inst.metadata["objective"], coordinate_metadata=inst.metadata)
        print(f'{Path(path).stem}: compared={result["summary"]["mean_cost"]:.3f} '
              f'unrounded={result["per_instance"].unrounded_cost.iloc[0]:.3f}, rounding={a.rounding}')


if __name__ == "__main__":
    main()
