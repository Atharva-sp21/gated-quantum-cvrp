import json

import numpy as np
import pandas as pd
import pytest

from scripts.report import build_reports


def test_report_seed_requirement_and_paired_ids(tmp_path):
    # Explicit toy fixtures exercise report plumbing; never published as research results.
    root = tmp_path / "fixtures"
    for variant in ["M0", "M3"]:
        for seed in range(5):
            path = root / f"{variant}_{seed}"
            path.mkdir(parents=True)
            values = np.arange(1., 21.)*(1. if variant == "M0" else 0.99)
            pd.DataFrame(dict(instance_id=np.arange(20), cost=values, reference=np.arange(1., 21.)*0.9)).to_csv(path/"per_instance.csv", index=False)
            summary = dict(variant=variant, training_seed=seed, train_customers=20, customers=20,
                           mode="greedy", augmentation=1, samples=1, dataset_fingerprint="toy_fixture",
                           mean_cost=float(values.mean()), mean_gap_percent=10., inference_seconds_per_instance=0.01)
            (path/"summary.json").write_text(json.dumps(summary))
    table = build_reports(root, tmp_path/"report")
    assert len(table) == 2 and (table.n_seeds == 5).all()
    stats = pd.read_csv(tmp_path/"report/statistics.csv")
    assert stats.rank_biserial_improvement.iloc[0] == 1
    assert (tmp_path/"report/results.tex").exists()
    frame = pd.read_csv(root/"M3_0/per_instance.csv")
    frame.instance_id += 1
    frame.to_csv(root/"M3_0/per_instance.csv", index=False)
    with pytest.raises(ValueError, match="IDs mismatch"):
        build_reports(root, tmp_path/"bad")
