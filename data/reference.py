from __future__ import annotations

import json
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch

from data.instances import fingerprint
from envs.cvrp import recompute_cost, validate_routes


def solve_pyvrp(instances, index, time_limit=1., seed=0, scale=1_000_000):
    """PyVRP 0.12.2 HGS with quantized edges, re-scored in true Euclidean units."""
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    if time_limit <= 0:
        raise ValueError("positive solver runtime required")
    inst = instances.slice(index, index+1).to("cpu")
    capacity = int(inst.metadata["capacity"])
    demands = (inst.demands[0]*capacity).round().long().tolist()
    if not torch.allclose(inst.demands[0]*capacity, torch.tensor(demands).float(), atol=1e-4):
        raise ValueError("reference adapter requires integer raw demands")
    coords = inst.coords[0].numpy()
    distance = inst.distances[0].numpy()
    model = Model()
    locations = [model.add_depot(x=float(coords[0, 0]), y=float(coords[0, 1]))]
    locations += [model.add_client(x=float(coords[i, 0]), y=float(coords[i, 1]), delivery=demands[i])
                  for i in range(1, inst.size+1)]
    model.add_vehicle_type(num_available=inst.size, capacity=capacity)
    for i, frm in enumerate(locations):
        for j, to in enumerate(locations):
            model.add_edge(frm, to, distance=int(np.floor(distance[i, j]*scale+0.5)))
    result = model.solve(stop=MaxRuntime(time_limit), seed=seed, display=False)
    if not result.best.is_feasible():
        raise RuntimeError(f"reference solver found no feasible solution for instance {index}")
    flat = []
    for route in result.best.routes():
        if flat:
            flat.append(0)
        flat.extend(route.visits())
    routes = torch.tensor(flat)[None, None]
    validate_routes(inst, routes)
    cost = float(recompute_cost(inst.coords, routes))
    return cost, flat


def save_reference(path, costs, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, costs=np.asarray(costs, dtype=np.float64), metadata=np.array(json.dumps(metadata)))
    temp.replace(path)


def reference_cache(instances, path, time_limit=1., seed=0, scale=1_000_000):
    meta = dict(dataset_fingerprint=fingerprint(instances), objective="euclidean", solver="PyVRP HGS",
                solver_version=version("pyvrp"), time_limit_seconds=time_limit, seed=seed,
                integer_distance_scale=scale, reference_type="heuristic_feasible",
                cost_convention="continuous Euclidean re-score of HGS routes")
    costs = np.full(len(instances.coords), np.nan)
    if Path(path).exists():
        with np.load(path, allow_pickle=False) as cache:
            existing = json.loads(str(cache["metadata"]))
            if existing != meta or len(cache["costs"]) != len(costs):
                raise ValueError("reference cache metadata mismatch; choose a new output path")
            costs[:] = cache["costs"]
    for i in range(len(costs)):
        if np.isfinite(costs[i]):
            continue
        costs[i], route = solve_pyvrp(instances, i, time_limit, seed+i, scale)
        # Checkpoint each reference, allowing interruption/resumption without losing work.
        save_reference(path, costs, meta)
        print(f"reference {i+1}/{len(costs)} cost={costs[i]:.6f}", flush=True)
    return costs
