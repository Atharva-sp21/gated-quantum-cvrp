from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class ClassicalBottleneck(nn.Module):
    """Small tanh MLP; count is measured rather than assumed exactly equal."""
    def __init__(self, n, out, depth, width=None):
        super().__init__()
        target = 3*n*depth
        width = width or max(1, round((target-out)/(n+out+1)))
        self.net = nn.Sequential(nn.Linear(n, width), nn.Tanh(), nn.Linear(width, out), nn.Tanh())

    def forward(self, theta, **noise):
        if noise and (noise.get("shots") is not None or noise.get("depolarizing", 0)):
            raise ValueError("quantum noise requested for classical backend")
        return self.net(theta)


class LatentBlock(nn.Module):
    @property
    def component_name(self):
        if not self.cfg["recon_loss"]:
            return "latent bottleneck"
        return "variational latent autoencoder" if self.cfg["mve_latent"] else "latent autoencoder"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d, n = cfg["d_model"], cfg["n_qubits"]
        self.norm = nn.LayerNorm(d)
        self.compress = nn.Linear(d, n)
        self.posterior = nn.Linear(n, 2*n) if cfg["mve_latent"] else None
        out = n*(2 if cfg["zz"] else 1)
        if cfg["backend"] == "classical_bottleneck":
            self.circuit = ClassicalBottleneck(n, out, cfg["depth"], cfg["classical_width"])
        else:
            from quantum.simulator import StatevectorCircuit
            self.circuit = StatevectorCircuit(n, cfg["depth"], cfg["zz"],
                                             frozen=cfg["backend"] == "frozen_random_quantum",
                                             chunk_size=cfg["quantum_chunk_size"])
        self.decode = nn.Linear(out, d)
        if cfg["alpha_zero"]:
            self.register_buffer("alpha", torch.tensor(0.))
        else:
            self.alpha = nn.Parameter(torch.tensor(float(cfg["alpha_init"])))
        self.reconstruction = None
        if cfg["recon_loss"]:
            self.reconstruction = nn.Sequential(nn.Linear(n, d), nn.Tanh(), nn.Linear(d, d))

    def forward(self, x, eps=None, sample=None, noise=None):
        target = self.norm(x)
        latent = self.compress(target)
        aux = {}
        if self.posterior is not None:
            mu, logvar = self.posterior(latent).chunk(2, -1)
            logvar = logvar.clamp(-6, 2)
            sigma = torch.exp(0.5*logvar)
            sampling = self.training or self.cfg["eval_latent_sampling"] if sample is None else sample
            if sampling:
                eps = torch.randn_like(mu) if eps is None else eps
                z = mu + sigma*eps
            else:
                z = mu
            kl_dimensions = 0.5*(mu.square()+logvar.exp()-1-logvar)
            aux.update(mu=mu, logvar=logvar, eps=eps, kl_raw_dimensions=kl_dimensions,
                       kl_dimensions=kl_dimensions.mean((0, 1)), sigma_latent=sigma.mean(),
                       active_dimensions=(kl_dimensions.mean((0, 1)) > 0.01).float().sum())
        else:
            z = latent
        angles = math.pi * torch.tanh(z)
        q = self.circuit(angles, **(noise or {}))
        residual = self.alpha * self.decode(q.to(x.dtype))
        aux.update(z=z, alpha=self.alpha, contribution=(residual.float().norm(dim=-1) /
                   x.float().norm(dim=-1).clamp_min(1e-8)).mean())
        if self.reconstruction is not None:
            aux["recon"] = F.mse_loss(self.reconstruction(z), target.detach())
        return x + residual, aux
