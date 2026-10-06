import argparse
from pathlib import Path

import torch

from data.instances import generate, save_instances


def main():
    p = argparse.ArgumentParser(description="Generate fixed validation, test, and shifted datasets")
    p.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--validation-count", type=int, default=10000)
    p.add_argument("--test-count", type=int, default=10000)
    p.add_argument("--test100-count", type=int, default=10000, help="set 1000 if compute-bound; count stored")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--distribution", choices=["uniform", "clustered"], default="uniform")
    p.add_argument("--demand-low", type=int, default=1)
    p.add_argument("--demand-high", type=int, default=9)
    p.add_argument("--output", default="datasets")
    a = p.parse_args()
    for m in a.sizes:
        for split, count in [("validation", a.validation_count), ("test", a.test100_count if m == 100 else a.test_count)]:
            if count < 1:
                raise ValueError("dataset count must be positive")
            seed = a.seed + m*100 + (0 if split == "validation" else 1)
            inst = generate(count, m, torch.Generator().manual_seed(seed), distribution=a.distribution,
                            demand_low=a.demand_low, demand_high=a.demand_high)
            path = Path(a.output) / f"{split}_{m}.npz"
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}; choose another output directory")
            save_instances(path, inst, seed=seed, count=count, split=split)
            print(path)


if __name__ == "__main__":
    main()
