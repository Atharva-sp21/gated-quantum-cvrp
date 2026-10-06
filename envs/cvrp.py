from __future__ import annotations

import torch


class CVRPEnv:
    """Vectorized (batch, starts) CVRP with automatic final return to depot.

    Finished rollouts select depot as padding, with zero incremental cost/logprob.
    """

    def __init__(self, instances, starts=None, check_actions=True):
        self.instances = instances
        self.coords, self.demands = instances.coords, instances.demands
        self.B, self.N = self.demands.shape
        self.P = starts or self.N - 1
        if not 1 <= self.P <= self.N - 1:
            raise ValueError("starts must be between 1 and number of customers")
        self.check_actions = check_actions
        dev = self.coords.device
        self.current = torch.zeros(self.B, self.P, dtype=torch.long, device=dev)
        self.load = torch.ones(self.B, self.P, device=dev)
        self.visited = torch.zeros(self.B, self.P, self.N, dtype=torch.bool, device=dev)
        self.done = torch.zeros(self.B, self.P, dtype=torch.bool, device=dev)
        self.cost = torch.zeros(self.B, self.P, device=dev)
        self.distance = instances.distances
        self.batch = torch.arange(self.B, device=dev)[:, None]
        self.routes = []

    def mask(self):
        mask = self.visited.clone()
        mask |= self.demands[:, None, :] > self.load[..., None] + 1e-6
        mask[..., 0] = (self.current == 0) & ~self.done
        mask = torch.where(self.done[..., None], torch.ones_like(mask), mask)
        mask[..., 0] = torch.where(self.done, torch.zeros_like(self.done), mask[..., 0])
        return mask

    def start(self):
        actions = torch.arange(1, self.P + 1, device=self.coords.device)[None].expand(self.B, -1)
        return self.step(actions)

    def step(self, actions):
        if actions.shape != self.current.shape or actions.dtype != torch.long:
            raise ValueError("actions must be int64 (B,P)")
        if self.check_actions and ((actions < 0).any() or (actions >= self.N).any()):
            raise ValueError("action out of range")
        if self.check_actions and self.mask().gather(-1, actions[..., None]).any():
            raise ValueError("infeasible action")
        active = ~self.done
        self.cost = self.cost + self.distance[self.batch, self.current, actions] * active
        demand = self.demands[self.batch, actions]
        self.load = torch.where(active, torch.where(actions == 0, torch.ones_like(self.load), self.load - demand), self.load)
        self.visited = self.visited.clone().scatter(-1, actions[..., None], True)
        self.visited[..., 0] = False
        self.current = torch.where(active, actions, self.current)
        finished = self.visited[..., 1:].all(-1) & active
        self.cost = self.cost + self.distance[self.batch, self.current, 0] * finished
        self.current = torch.where(finished, torch.zeros_like(self.current), self.current)
        self.done = self.done | finished
        self.routes.append(actions.clone())
        return self.done

    @property
    def reward(self):
        return -self.cost

    @property
    def route_tensor(self):
        return torch.stack(self.routes, -1)


def recompute_cost(coords, routes, distance=None):
    b, p, _ = routes.shape
    zero = torch.zeros(b, p, 1, dtype=torch.long, device=routes.device)
    path = torch.cat([zero, routes, zero], -1)
    distance = torch.cdist(coords.float(), coords.float(), compute_mode="donot_use_mm_for_euclid_dist") if distance is None else distance
    bi = torch.arange(b, device=routes.device)[:, None, None]
    return distance[bi, path[..., :-1], path[..., 1:]].sum(-1)


def validate_routes(instances, routes, atol=1e-5):
    b, p, _ = routes.shape
    if b != instances.coords.shape[0]:
        raise ValueError("batch mismatch")
    for bi in range(b):
        for pi in range(p):
            seq = routes[bi, pi].tolist()
            customers = [a for a in seq if a != 0]
            if sorted(customers) != list(range(1, instances.size + 1)):
                raise ValueError("each customer must occur exactly once")
            load = 1.0
            for a in seq:
                if a == 0:
                    load = 1.0
                else:
                    load -= float(instances.demands[bi, a])
                    if load < -atol:
                        raise ValueError("capacity violation")
    return True
