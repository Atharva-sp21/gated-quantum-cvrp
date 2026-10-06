from __future__ import annotations

import math
import torch
from torch import nn


def ry(angle):
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    return torch.stack([c, -s, s, c], -1).reshape(*angle.shape, 2, 2).to(torch.complex64)


def rz(angle):
    a = torch.exp(-0.5j * angle.to(torch.complex64))
    zero = torch.zeros_like(a)
    return torch.stack([a, zero, zero, a.conj()], -1).reshape(*angle.shape, 2, 2)


def apply_gate(state, gate, wire, wires):
    """Apply a 2x2 gate on one tensor axis; wire 0 is most significant.

    Also handles flattened density matrices with 2*n tensor axes.
    """
    batch = state.shape[0]
    x = state.reshape(batch, 2**wire, 2, 2**(wires-wire-1))
    if gate.ndim == 2:
        out = torch.einsum("ij,bljr->blir", gate, x)
    else:
        out = torch.einsum("bij,bljr->blir", gate, x)
    return out.reshape_as(state)


class StatevectorCircuit(nn.Module):
    def __init__(self, n_qubits=6, depth=2, zz=False, frozen=False, chunk_size=2048):
        super().__init__()
        if not 2 <= n_qubits <= 12 or depth < 1 or chunk_size < 1:
            raise ValueError("require 2<=n_qubits<=12, depth>=1, chunk_size>=1")
        self.n_qubits, self.depth, self.zz, self.chunk_size = n_qubits, depth, zz, chunk_size
        self.angles = nn.Parameter(torch.randn(depth, n_qubits, 3) * 0.1, requires_grad=not frozen)
        basis = torch.arange(2**n_qubits)
        signs = torch.stack([1 - 2 * ((basis >> (n_qubits-1-i)) & 1) for i in range(n_qubits)], -1).float()
        if zz:
            signs = torch.cat([signs, signs * signs.roll(-1, -1)], -1)
        self.register_buffer("signs", signs)
        for c in range(n_qubits):
            t = (c+1) % n_qubits
            permutation = basis ^ (((basis >> (n_qubits-1-c)) & 1) << (n_qubits-1-t))
            self.register_buffer(f"cnot_{c}", permutation)

    @property
    def n_out(self):
        return self.n_qubits * (2 if self.zz else 1)

    def _run(self, theta, depolarizing):
        b, n = theta.shape
        dim = 2**n
        state = torch.zeros(b, dim, dtype=torch.complex64, device=theta.device)
        state[:, 0] = 1
        density = depolarizing > 0
        if density:
            state = (state[..., None] * state[:, None, :].conj()).reshape(b, -1)

        def gate_apply(s, g, q):
            if density:
                s = apply_gate(s, g, q, 2*n)
                return apply_gate(s, g.conj(), n+q, 2*n)
            return apply_gate(s, g, q, n)

        for layer in range(self.depth):
            for q in range(n):
                state = gate_apply(state, ry(theta[:, q]), q)
                phi, th, omega = self.angles[layer, q].float().unbind()
                # PennyLane Rot(phi,theta,omega) = RZ(omega) RY(theta) RZ(phi).
                state = gate_apply(state, rz(phi), q)
                state = gate_apply(state, ry(th), q)
                state = gate_apply(state, rz(omega), q)
            for q in range(n):
                index = getattr(self, f"cnot_{q}")
                if density:
                    state = state.reshape(b, dim, dim).index_select(1, index).index_select(2, index).reshape(b, -1)
                else:
                    state = state.index_select(1, index)
            if density:
                paulis = [torch.tensor(g, dtype=torch.complex64, device=theta.device) for g in
                          [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]]
                for q in range(n):
                    noisy = sum(gate_apply(state, g, q) for g in paulis)
                    state = (1-depolarizing) * state + depolarizing/3 * noisy
        if density:
            probs = state.reshape(b, dim, dim).diagonal(dim1=-2, dim2=-1).real
        else:
            probs = state.abs().square()
        return probs @ self.signs.float()

    def forward(self, theta, shots=None, depolarizing=0., shot_method="binomial"):
        if theta.shape[-1] != self.n_qubits:
            raise ValueError("angle dimension must equal n_qubits")
        if not 0 <= depolarizing <= 1:
            raise ValueError("depolarizing probability must be in [0,1]")
        if (shots is not None or depolarizing) and self.training:
            raise ValueError("noise is inference-only; call eval()")
        if depolarizing and self.n_qubits > 8:
            raise ValueError("density-matrix path limited to <=8 qubits")
        if shots is not None and (not isinstance(shots, int) or shots < 1):
            raise ValueError("shots must be a positive integer or None")
        shape = theta.shape[:-1]
        # AMP is NEVER enabled in quantum operations; simulator stays complex64.
        with torch.autocast(device_type=theta.device.type, enabled=False):
            flat = theta.float().reshape(-1, self.n_qubits)
            chunk = min(self.chunk_size, 16 if depolarizing else self.chunk_size)
            out = torch.cat([self._run(t, depolarizing) for t in flat.split(chunk)], 0)
            out = out.clamp(-1, 1)
            if shots is not None:
                if shot_method == "binomial":
                    out = 2 * torch.binomial(torch.full_like(out, shots), (out+1)/2) / shots - 1
                elif shot_method == "gaussian":
                    out = (out + torch.randn_like(out) * ((1-out.square()).clamp_min(0)/shots).sqrt()).clamp(-1, 1)
                else:
                    raise ValueError("shot_method must be binomial or gaussian")
        return out.reshape(*shape, self.n_out)

    def memory_bytes(self, circuits, density=False):
        """One state allocation, excluding autograd history and temporaries."""
        return circuits * (4**self.n_qubits if density else 2**self.n_qubits) * 8
