from __future__ import annotations

import torch
from torch import nn


class Critic(nn.Module):
    """Shared-encoder 1D-conv critic with dynamic last-node/load context."""
    def __init__(self, d_model=128, mve=True):
        super().__init__()
        self.mve = mve
        self.conv = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.ReLU(), nn.Conv1d(d_model, d_model, 1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2*d_model+1, d_model), nn.ReLU(), nn.Linear(d_model, 2 if mve else 1))

    def forward(self, h, currents, loads):
        b, p, t = currents.shape
        static = self.conv(h.transpose(1, 2))[:, :, 1:].mean(-1)
        batch = torch.arange(b, device=h.device)[:, None, None]
        last = h[batch, currents]
        context = torch.cat([static[:, None, None].expand(-1, p, t, -1), last, loads[..., None]], -1)
        output = self.head(context)
        mu = output[..., 0]
        logvar = output[..., 1].clamp(-10, 10) if self.mve else torch.zeros_like(mu)
        return mu, logvar
