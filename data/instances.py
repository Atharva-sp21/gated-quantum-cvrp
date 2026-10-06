from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CAPACITIES = {20: 30, 50: 40, 100: 50}


@dataclass
class Instances:
    coords: torch.Tensor  # B,N,2; depot at index 0
    demands: torch.Tensor  # B,N; normalized, depot zero
    metadata: dict

    def __post_init__(self):
        if self.coords.ndim != 3 or self.coords.shape[-1] != 2:
            raise ValueError("coords must have shape (B,N,2)")
        if self.demands.shape != self.coords.shape[:2]:
            raise ValueError("demands must have shape (B,N)")
        if not torch.isfinite(self.coords).all() or not torch.isfinite(self.demands).all():
            raise ValueError("non-finite instance")
        if (self.demands[:, 0] != 0).any() or (self.demands[:, 1:] <= 0).any() or (self.demands > 1).any():
            raise ValueError("depot demand must be zero; customer demands must be in (0,1]")

    def to(self, device):
        return Instances(self.coords.to(device), self.demands.to(device), self.metadata)

    def slice(self, start, end):
        return Instances(self.coords[start:end], self.demands[start:end], self.metadata)

    @property
    def distances(self):
        return torch.cdist(self.coords.float(), self.coords.float(), compute_mode="donot_use_mm_for_euclid_dist")

    @property
    def size(self):
        return self.coords.shape[1] - 1


def generate(batch, customers, generator=None, capacity=None, demand_low=1, demand_high=9,
             distribution="uniform", device="cpu"):
    capacity = capacity or CAPACITIES.get(customers)
    if capacity is None:
        raise ValueError("supply capacity for nonstandard size")
    if not 1 <= demand_low <= demand_high <= capacity:
        raise ValueError("invalid demand range/capacity")
    coords = torch.rand(batch, customers + 1, 2, generator=generator)
    if distribution == "clustered":
        centers = torch.rand(batch, 3, 2, generator=generator)
        labels = torch.randint(3, (batch, customers), generator=generator)
        points = centers.gather(1, labels[..., None].expand(-1, -1, 2))
        coords[:, 1:] = (points + 0.08 * torch.randn(batch, customers, 2, generator=generator)).clamp(0, 1)
    elif distribution != "uniform":
        raise ValueError(distribution)
    demands = torch.randint(demand_low, demand_high + 1, (batch, customers + 1), generator=generator).float() / capacity
    demands[:, 0] = 0
    meta = dict(customers=customers, capacity=capacity, distribution=distribution,
                demand_low=demand_low, demand_high=demand_high, objective="euclidean", format_version=1)
    return Instances(coords.to(device), demands.to(device), meta)


def augment8(instances):
    x, y = instances.coords.unbind(-1)
    variants = [(x, y), (x, 1-y), (1-x, y), (1-x, 1-y),
                (y, x), (y, 1-x), (1-y, x), (1-y, 1-x)]
    coords = torch.cat([torch.stack(v, -1) for v in variants], 0)
    return Instances(coords, instances.demands.repeat(8, 1), dict(instances.metadata, augmentation=8))


def fingerprint(instances):
    h = hashlib.sha256()
    for t in (instances.coords, instances.demands):
        a = t.detach().cpu().contiguous().numpy()
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def save_instances(path, instances, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(instances.metadata, **metadata, fingerprint=fingerprint(instances))
    with path.open("wb") as f:
        np.savez_compressed(f, coords=instances.coords.cpu().numpy(), demands=instances.demands.cpu().numpy(),
                            metadata=np.array(json.dumps(meta)))


def load_instances(path):
    with np.load(path, allow_pickle=False) as d:
        inst = Instances(torch.from_numpy(d["coords"].copy()).float(),
                         torch.from_numpy(d["demands"].copy()).float(), json.loads(str(d["metadata"])))
    if inst.metadata.get("fingerprint") != fingerprint(inst):
        raise ValueError("dataset fingerprint mismatch")
    return inst


def knn_mask(distances, k=20):
    n = distances.shape[-1]
    k = min(k, n - 1)
    d = distances.clone()
    d.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
    indices = d.topk(k, largest=False).indices
    mask = torch.zeros_like(d, dtype=torch.bool).scatter_(-1, indices, True)
    mask.diagonal(dim1=-2, dim2=-1).fill_(True)
    mask[:, :, 0] = True  # all nodes can attend to the depot
    mask[:, 0, :] = True
    return mask
