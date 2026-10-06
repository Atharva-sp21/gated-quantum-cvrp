import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.instances import fingerprint, load_instances
from data.reference import reference_cache, save_reference


def main():
    p = argparse.ArgumentParser(description="Build resumable reference cache or import supplied costs")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--time-limit", type=float, default=1.)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--published-csv", help="CSV with instance_id,cost; IDs must match zero-based dataset order")
    p.add_argument("--source", help="required provenance for published costs")
    a = p.parse_args()
    inst = load_instances(a.dataset)
    if a.published_csv:
        if not a.source:
            p.error("--source required for imported references")
        df = pd.read_csv(a.published_csv).sort_values("instance_id")
        if not np.array_equal(df.instance_id.to_numpy(), np.arange(len(inst.coords))):
            raise ValueError("published reference IDs must exactly match dataset order")
        if not np.isfinite(df.cost).all() or (df.cost <= 0).any():
            raise ValueError("reference costs must be finite and positive")
        if Path(a.output).exists():
            raise FileExistsError(a.output)
        save_reference(a.output, df.cost.to_numpy(), dict(dataset_fingerprint=fingerprint(inst),
                       objective="euclidean", reference_type="published_user_supplied", source=a.source,
                       cost_convention="continuous Euclidean; user responsible for matching instances"))
    else:
        reference_cache(inst, a.output, a.time_limit, a.seed)


if __name__ == "__main__":
    main()
