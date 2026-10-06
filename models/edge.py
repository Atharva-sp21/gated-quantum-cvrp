from __future__ import annotations

import math
import torch
from torch import nn

from quantum.simulator import StatevectorCircuit


class QuantumEdgeScoring(nn.Module):
    """Experimental directed k-neighbor pair circuit; never forms N*N circuits."""
    def __init__(self, cfg):
        super().__init__()
        n = cfg["n_qubits"]
        self.encode = nn.Linear(2*n+1, n)
        self.circuit = StatevectorCircuit(n, cfg["depth"], chunk_size=cfg["quantum_chunk_size"])
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z, distance):
        b, nodes, n = z.shape
        k = min(20, nodes-1)
        d = distance.clone()
        d.diagonal(dim1=-2, dim2=-1).fill_(torch.inf)
        neighbors = d.topk(k, largest=False).indices
        batch = torch.arange(b, device=z.device)[:, None, None]
        neighbor_z = z[batch, neighbors]
        source_z = z[:, :, None].expand(-1, -1, k, -1)
        pair = torch.cat([source_z, neighbor_z, distance.gather(-1, neighbors)[..., None]], -1)
        angles = math.pi*torch.tanh(self.encode(pair))
        score = self.scale*self.circuit(angles).mean(-1)
        return torch.zeros_like(distance).scatter(-1, neighbors, score.to(distance.dtype))
