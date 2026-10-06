import argparse
from pathlib import Path

from data.instances import load_instances
from data.reference import reference_cache


def main():
    p = argparse.ArgumentParser(description="Prepare all six reference caches; reference solving only, no ML training")
    p.add_argument("--datasets", default="datasets")
    p.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--time-limit", type=float, default=1.)
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()
    for split in ["validation", "test"]:
        for m in a.sizes:
            path = Path(a.datasets)/f"{split}_{m}.npz"
            reference_cache(load_instances(path), path.with_name(path.stem+"_reference.npz"), a.time_limit, a.seed)


if __name__ == "__main__":
    main()
