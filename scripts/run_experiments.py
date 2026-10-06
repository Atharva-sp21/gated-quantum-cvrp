from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from train.config import load_config
from train.utils import save_json


def main():
    p = argparse.ArgumentParser(description="Plan or explicitly execute sequential multi-seed GPU experiments")
    p.add_argument("--configs", nargs="+", default=["configs/M0.yaml", "configs/M1.yaml", "configs/M2.yaml", "configs/M3.yaml"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--customers", type=int, default=100)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--time-budget", type=float)
    p.add_argument("--device", default="cuda")
    p.add_argument("--datasets", default="datasets")
    p.add_argument("--run-output", default="runs")
    p.add_argument("--report-output", default="reports/main")
    p.add_argument("--manifest", default="reports/experiment_manifest.json")
    p.add_argument("--execute", action="store_true", help="ACTUALLY TRAIN, then evaluate; default only writes commands")
    p.add_argument("--resume-existing", action="store_true", help="resume each existing last.pt with identical config")
    p.add_argument("--sampling-samples", type=int, default=8)
    p.add_argument("--test-sizes", nargs="+", type=int, default=[20, 50, 100])
    a = p.parse_args()
    if len(set(a.seeds)) < 5:
        p.error("research runner requires >=5 distinct seeds")
    jobs = []
    for config_path in a.configs:
        cfg = load_config(config_path)
        for seed in a.seeds:
            run = Path(a.run_output) / f'{cfg["name"]}_m{a.customers}_seed{seed}'
            override = [f"customers={a.customers}", f"seed={seed}", f"device={a.device}",
                        f"training.epochs={a.epochs}", f"output={a.run_output}", "training.trainability_every=10",
                        f"validation.dataset={a.datasets}/validation_{a.customers}.npz",
                        f"validation.reference={a.datasets}/validation_{a.customers}_reference.npz"]
            if a.time_budget is not None:
                override.append(f"training.time_budget_seconds={a.time_budget}")
            command = [sys.executable, "-m", "train.main", "--config", config_path, "--set", *override]
            if a.resume_existing and (run/"last.pt").exists():
                command += ["--resume", str(run/"last.pt")]
            jobs.append(dict(kind="train", command=command))
            for mode, augment, samples, label in [("greedy", False, 1, "greedy"), ("greedy", True, 1, "greedy_x8"),
                                                   ("sampling", False, a.sampling_samples, f"sampling_{a.sampling_samples}")]:
                command = [sys.executable, "-m", "eval.evaluate", "--checkpoint", str(run/"best.pt"),
                           "--datasets", *[f"{a.datasets}/test_{m}.npz" for m in a.test_sizes],
                           "--references", *[f"{a.datasets}/test_{m}_reference.npz" for m in a.test_sizes],
                           "--device", a.device, "--mode", mode, "--samples", str(samples),
                           "--output", str(Path(a.report_output)/run.name/label)]
                if augment:
                    command.append("--augment")
                jobs.append(dict(kind="evaluate", command=command))
    save_json(a.manifest, dict(execute_requested=a.execute, seeds=a.seeds, jobs=jobs))
    print(f"{len(jobs)} commands saved to {a.manifest}; execute={a.execute}")
    if a.execute:
        # Fail before any training if reproducibility inputs are absent/incompatible.
        from data.instances import load_instances
        from eval.evaluate import load_reference
        for split, m in [("validation", a.customers)]+[("test", m) for m in a.test_sizes]:
            inst = load_instances(f"{a.datasets}/{split}_{m}.npz")
            load_reference(f"{a.datasets}/{split}_{m}_reference.npz", inst)
        for job in jobs:
            subprocess.run(job["command"], check=True)


if __name__ == "__main__":
    main()
