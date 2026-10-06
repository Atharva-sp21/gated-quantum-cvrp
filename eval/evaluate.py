from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.instances import augment8, fingerprint, load_instances
from envs.cvrp import recompute_cost, validate_routes
from models.backbone import Policy
from train.utils import parameter_counts, runtime_metadata, save_json, seed_all


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = Policy(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_reference(path, instances):
    with np.load(path, allow_pickle=False) as cache:
        costs = cache["costs"].copy()
        metadata = json.loads(str(cache["metadata"]))
    if metadata["dataset_fingerprint"] != fingerprint(instances):
        raise ValueError("reference does not match these exact instances")
    if metadata["objective"] != "euclidean":
        raise ValueError("reference objective incompatible with synthetic evaluation")
    if len(costs) != len(instances.coords) or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("reference cache incomplete or invalid")
    return costs


@torch.no_grad()
def evaluate_instances(model, instances, batch_size=64, mode="greedy", augmentation=False,
                       samples=1, temperature=1., reference=None, noise=None, metric_distance=None,
                       unrounded_distance=None, sample_latent=None, verify=True, return_routes=False):
    if samples < 1 or batch_size < 1 or mode not in {"greedy", "sampling"}:
        raise ValueError("invalid decoding parameters")
    if mode == "greedy" and samples != 1:
        raise ValueError("greedy requires samples=1; use sampling for repeated stochastic passes")
    model.eval()
    device = next(model.parameters()).device
    rows, all_routes = [], []
    total_seconds = 0.
    for start in range(0, len(instances.coords), batch_size):
        original = instances.slice(start, start+batch_size).to(device)
        b = len(original.coords)
        inst = augment8(original) if augmentation else original
        factor = 8 if augmentation else 1
        best_cost = torch.full((b,), torch.inf, device=device, dtype=torch.float64)
        selected_unrounded = torch.zeros(b, device=device, dtype=torch.float64)
        best_routes = [None]*b
        mu, sigma, latent_sigma, means = [], [], [], []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        for _ in range(samples):
            result = model(inst, mode=mode, temperature=temperature,
                           sample_latent=False if mode == "greedy" else sample_latent, noise=noise)
            if verify:
                validate_routes(inst, result["routes"])
            cost = result["cost"].double()
            unrounded = cost
            if metric_distance is not None:
                d = metric_distance[start:start+b].to(device).repeat(factor, 1, 1)
                cost = recompute_cost(inst.coords, result["routes"], d)
                ud = unrounded_distance[start:start+b].to(device).repeat(factor, 1, 1) if unrounded_distance is not None else d
                unrounded = recompute_cost(inst.coords, result["routes"], ud)
            candidate = cost.reshape(factor, b, -1).permute(1, 0, 2).reshape(b, -1)
            candidate_u = unrounded.reshape(factor, b, -1).permute(1, 0, 2).reshape(b, -1)
            candidate_r = result["routes"].reshape(factor, b, instances.size, -1).permute(1, 0, 2, 3).reshape(b, -1, result["routes"].shape[-1])
            value, idx = candidate.min(-1)
            improved = value < best_cost
            for j in torch.nonzero(improved).flatten().tolist():
                best_routes[j] = candidate_r[j, idx[j]].cpu().numpy()
            selected_unrounded = torch.where(improved, candidate_u.gather(-1, idx[:, None]).squeeze(-1), selected_unrounded)
            best_cost = torch.minimum(best_cost, value)
            mu.append(result["aux"]["mu_c"].reshape(factor, b).mean(0).float())
            sigma.append(torch.exp(0.5*result["aux"]["logvar_c"]).reshape(factor, b).mean(0).float())
            means.append(result["cost"].reshape(factor, b, -1).mean((0, 2)).float())
            if "logvar" in result["aux"]:
                latent_sigma.append(torch.exp(0.5*result["aux"]["logvar"]).reshape(factor, b, -1).mean((0, 2)).float())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter()-tick
        total_seconds += duration
        for j in range(b):
            row = dict(instance_id=start+j, cost=float(best_cost[j]), unrounded_cost=float(selected_unrounded[j]),
                       mean_rollout_cost=float(torch.stack(means).mean(0)[j]), mu_c=float(torch.stack(mu).mean(0)[j]),
                       sigma_c=float(torch.stack(sigma).mean(0)[j]),
                       sigma_latent=float(torch.stack(latent_sigma).mean(0)[j]) if latent_sigma else np.nan)
            if reference is not None:
                ref = float(reference[start+j])
                if not np.isfinite(ref) or ref <= 0:
                    raise ValueError("positive finite reference required")
                row.update(reference=ref, gap_percent=100*(row["cost"]/ref-1))
            rows.append(row)
        if return_routes:
            all_routes.extend(best_routes)
    frame = pd.DataFrame(rows)
    summary = dict(mean_cost=float(frame.cost.mean()), std_instance_cost=float(frame.cost.std(ddof=0)),
                   count=len(frame), customers=instances.size, mode=mode, samples=samples, augmentation=factor,
                   total_candidates_per_instance=samples*instances.size*factor, temperature=temperature,
                   seconds=total_seconds, inference_seconds_per_instance=total_seconds/len(frame),
                   dataset_fingerprint=fingerprint(instances), noise=noise or {},
                   parameter_counts=parameter_counts(model))
    summary["latent_sampling"] = bool(mode == "sampling" and model.cfg["mve_latent"] and
                                     (model.cfg["eval_latent_sampling"] if sample_latent is None else sample_latent))
    if reference is not None:
        summary["mean_gap_percent"] = float(frame.gap_percent.mean())
    return dict(summary=summary, per_instance=frame, routes=all_routes if return_routes else None)


def write_evaluation(result, output, **metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result["per_instance"].to_csv(output / "per_instance.csv", index=False)
    save_json(output / "summary.json", dict(result["summary"], **metadata))
    if result.get("routes") is not None:
        save_json(output / "routes.json", [r.tolist() for r in result["routes"]])


def main():
    p = argparse.ArgumentParser(description="Evaluate checkpoint on saved synthetic instances")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--references", nargs="+")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["greedy", "sampling"], default="greedy")
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--sample-latent", action="store_true", default=None)
    p.add_argument("--shots", type=int)
    p.add_argument("--depolarizing", type=float, default=0.)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", default="reports/evaluation")
    p.add_argument("--save-routes", action="store_true")
    a = p.parse_args()
    if a.references and len(a.references) != len(a.datasets):
        p.error("one reference cache required per dataset")
    seed_all(a.seed)
    model, cfg = load_checkpoint(a.checkpoint, a.device)
    for i, path in enumerate(a.datasets):
        inst = load_instances(path)
        ref = load_reference(a.references[i], inst) if a.references else None
        noise = dict(shots=a.shots, depolarizing=a.depolarizing)
        result = evaluate_instances(model, inst, a.batch_size, a.mode, a.augment, a.samples, a.temperature,
                                    ref, noise, sample_latent=a.sample_latent, return_routes=a.save_routes)
        write_evaluation(result, Path(a.output) / Path(path).stem, checkpoint=str(a.checkpoint),
                         variant=cfg["name"], train_customers=cfg["customers"], training_seed=cfg["seed"],
                         evaluation_seed=a.seed, runtime=runtime_metadata(), reference_path=a.references[i] if a.references else None)
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
