from __future__ import annotations

import math
import torch
from torch import nn

from data.instances import knn_mask
from envs.cvrp import CVRPEnv


class NodeNorm(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.kind = kind
        self.norm = {"instance": lambda: nn.InstanceNorm1d(d, affine=True, track_running_stats=False),
                     "batch": lambda: nn.BatchNorm1d(d), "layer": lambda: nn.LayerNorm(d)}[kind]()

    def forward(self, x):
        return self.norm(x) if self.kind == "layer" else self.norm(x.transpose(1, 2)).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h = cfg["d_model"], cfg["heads"]
        self.heads, self.dim = h, d // h
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d)
        self.norm1, self.norm2 = NodeNorm(d, cfg["norm"]), NodeNorm(d, cfg["norm"])
        self.ff = nn.Sequential(nn.Linear(d, cfg["ffn"]), nn.ReLU(), nn.Linear(cfg["ffn"], d))
        self.bias_kind = cfg["distance_bias"]
        if self.bias_kind == "mlp":
            self.distance_mlp = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, h))
        elif self.bias_kind == "scalar":
            self.distance_scale = nn.Parameter(torch.ones(h))

    def forward(self, x, distance, allowed=None, edge_bias=None):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        if self.bias_kind == "mlp":
            logits = logits + self.distance_mlp(distance[..., None]).permute(0, 3, 1, 2)
        elif self.bias_kind == "scalar":
            logits = logits - distance[:, None] * self.distance_scale[None, :, None, None]
        if edge_bias is not None:
            logits = logits + edge_bias[:, None]
        if allowed is not None:
            logits = logits.masked_fill(~allowed[:, None], -torch.inf)
        attention = logits.float().softmax(-1).to(v.dtype)
        mixed = (attention @ v).transpose(1, 2).reshape(b, n, d)
        x = self.norm1(x + self.out(mixed))
        return self.norm2(x + self.ff(x))


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.heads, self.dim = cfg["heads"], d // cfg["heads"]
        self.query = nn.Linear(3*d+1, d, bias=False)
        self.key, self.value = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.combine = nn.Linear(d, d, bias=False)
        self.logit_key = nn.Linear(d, d, bias=False)

    def cache(self, h):
        b, n, d = h.shape
        return (self.key(h).reshape(b, n, self.heads, self.dim).transpose(1, 2),
                self.value(h).reshape(b, n, self.heads, self.dim).transpose(1, 2), self.logit_key(h))

    def forward(self, h, global_h, env, cache, temperature=1.):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        b, p = env.current.shape
        last = h[env.batch, env.current]
        context = torch.cat([global_h[:, None].expand(-1, p, -1), last, env.load[..., None]], -1)
        q = self.query(context).reshape(b, p, self.heads, self.dim).transpose(1, 2)
        k, v, logit_k = cache
        mask = env.mask()
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dim)
        logits = logits.masked_fill(mask[:, None], -torch.inf)
        glimpse = (logits.float().softmax(-1).to(v.dtype) @ v).transpose(1, 2).reshape(b, p, -1)
        query = self.combine(glimpse)
        scores = 10 * torch.tanh((query @ logit_k.transpose(1, 2)) / math.sqrt(h.shape[-1]))
        return (scores.float() / temperature).masked_fill(mask, -torch.inf)


class Policy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d = cfg["d_model"]
        self.customer_embedding = nn.Linear(3, d)
        self.depot_embedding = nn.Linear(2, d)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg["layers"])])
        self.decoder = Decoder(cfg)
        self.difficulty = nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Linear(d, 2))
        self.latent = None
        if cfg["backend"] != "none":
            from models.latent import LatentBlock
            self.latent = LatentBlock(cfg)
        self.edge = None
        if cfg["quantum_edge_scoring"]:
            from models.edge import QuantumEdgeScoring
            self.edge = QuantumEdgeScoring(cfg)

    def encode(self, instances, latent_eps=None, sample_latent=None, noise=None):
        features = torch.cat([instances.coords[:, 1:], instances.demands[:, 1:, None]], -1)
        x = torch.cat([self.depot_embedding(instances.coords[:, :1]), self.customer_embedding(features)], 1)
        distance = instances.distances
        allowed = knn_mask(distance) if self.cfg["knn"] else None
        aux, edge_bias = {}, None
        for i, layer in enumerate(self.layers, 1):
            x = layer(x, distance, allowed, edge_bias)
            if self.latent is not None and i == self.cfg["q_layer"]:
                x, aux = self.latent(x, eps=latent_eps, sample=sample_latent, noise=noise)
                if self.edge is not None:
                    edge_bias = self.edge(aux["z"], distance)
        customers = x[:, 1:]
        global_h = torch.cat([customers.mean(1), customers.amax(1)], -1)
        diff = self.difficulty(global_h.detach() if self.cfg["difficulty_detach"] else global_h)
        aux["mu_c"], aux["logvar_c"] = diff[:, 0], diff[:, 1].clamp(
            self.cfg["difficulty_logvar_min"], self.cfg["difficulty_logvar_max"])
        return x, global_h, aux

    def forward(self, instances, mode="sampling", temperature=1., starts=None, latent_eps=None,
                sample_latent=None, noise=None, actions=None):
        if mode not in {"sampling", "greedy"}:
            raise ValueError("mode must be sampling or greedy")
        h, global_h, aux = self.encode(instances, latent_eps, sample_latent, noise)
        cache = self.decoder.cache(h)
        env = CVRPEnv(instances, starts=starts, check_actions=actions is not None)
        env.start()
        logprobs, entropies, active_steps, currents, loads = [], [], [], [], []
        step = 1
        while not env.done.all():
            active = ~env.done
            currents.append(env.current.clone())
            loads.append(env.load.clone())
            scores = self.decoder(h, global_h, env, cache, temperature)
            logp = scores.log_softmax(-1)
            probs = logp.exp()
            if actions is not None:
                if step >= actions.shape[-1]:
                    raise ValueError("replay actions truncated")
                selected = actions[..., step]
            elif mode == "greedy":
                selected = scores.argmax(-1)
            else:
                selected = torch.multinomial(probs.reshape(-1, env.N), 1).reshape(env.B, env.P)
            logprobs.append(logp.gather(-1, selected[..., None]).squeeze(-1) * active)
            entropies.append(-(probs * logp.masked_fill(~torch.isfinite(logp), 0)).sum(-1) * active)
            active_steps.append(active)
            env.step(selected)
            step += 1
            if step > 2*instances.size:
                raise RuntimeError("rollout exceeded feasible maximum steps")
        zero = h.sum() * 0
        shape = (env.B, env.P, 0)
        stack = lambda seq, dtype=None: torch.stack(seq, -1) if seq else torch.empty(shape, device=h.device, dtype=dtype or h.dtype)
        logp_steps = stack(logprobs)
        return dict(cost=env.cost, reward=env.reward, logprob=logp_steps.sum(-1) + zero,
                    logprob_steps=logp_steps, entropy_steps=stack(entropies), active=stack(active_steps, torch.bool),
                    currents=stack(currents, torch.long), loads=stack(loads), routes=env.route_tensor,
                    h=h, global_h=global_h, aux=aux)
