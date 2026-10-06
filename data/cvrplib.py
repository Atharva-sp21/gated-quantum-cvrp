from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.instances import Instances


def load_cvrplib(path, solution=None, rounding="nearest"):
    """Single-depot CVRP, isotropic min-max scale; never anisotropically warp routes."""
    import vrplib
    raw = vrplib.read_instance(str(path))
    if raw.get("type", "CVRP") != "CVRP":
        raise ValueError("only CVRP instances supported")
    depot = np.asarray(raw["depot"]).reshape(-1)
    if len(depot) != 1:
        raise ValueError("only single-depot instances supported")
    points = np.asarray(raw["node_coord"], dtype=np.float64)
    order = np.concatenate([depot, np.delete(np.arange(len(points)), depot)])
    points = points[order]
    offset = points.min(0)
    scale = max(float(np.ptp(points, axis=0).max()), 1.)
    demands = np.asarray(raw["demand"], dtype=np.float32)[order] / raw["capacity"]
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    if rounding == "nearest":
        objective = np.floor(distances+0.5)  # TSPLIB nearest integer, not bankers' round
    elif rounding == "floor":
        objective = np.floor(distances)
    elif rounding == "ceil":
        objective = np.ceil(distances)
    elif rounding == "none":
        objective = distances
    elif rounding == "explicit":
        objective = np.asarray(raw["edge_weight"])[np.ix_(order, order)]
    else:
        raise ValueError("choose nearest/floor/ceil/none/explicit rounding explicitly")
    best_known = float(vrplib.read_solution(str(solution))["cost"]) if solution else None
    metadata = dict(name=raw.get("name", Path(path).stem), source=str(path), customers=len(points)-1,
                    capacity=int(raw["capacity"]), distribution="cvrplib", scale=scale, offset=offset.tolist(),
                    original_node_order=order.tolist(), raw_coords=points.tolist(),
                    edge_weight_type=raw.get("edge_weight_type"), rounding=rounding,
                    best_known=best_known, best_known_source=str(solution) if solution else None,
                    objective="raw_euclidean" if rounding == "none" else f"raw_{rounding}", format_version=1)
    inst = Instances(torch.tensor((points-offset)/scale, dtype=torch.float32)[None],
                     torch.from_numpy(demands)[None], metadata)
    return inst, torch.tensor(objective, dtype=torch.float64)[None], torch.tensor(distances, dtype=torch.float64)[None]
