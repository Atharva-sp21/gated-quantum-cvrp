from __future__ import annotations

import os
import random
import json
import platform
from pathlib import Path

import numpy as np
import torch


def seed_all(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def parameter_counts(model):
    quantum = sum(p.numel() for name, p in model.named_parameters() if ".circuit.angles" in name)
    total = sum(p.numel() for p in model.parameters())
    return dict(classical=total-quantum, quantum=quantum, total=total,
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                quantum_trainable=sum(p.numel() for name, p in model.named_parameters()
                                      if ".circuit.angles" in name and p.requires_grad))


def gradient_norms(model):
    sums = dict(quantum=0., classical=0.)
    for name, p in model.named_parameters():
        if p.grad is not None:
            group = "quantum" if ".circuit.angles" in name else "classical"
            sums[group] += float(p.grad.detach().float().square().sum())
    return {f"grad_norm_{key}": value**0.5 for key, value in sums.items()}


def runtime_metadata():
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for package in ["torch", "numpy", "pyvrp", "vrplib", "pennylane", "scipy", "PyYAML"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def rng_state(generator=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                generator=generator.get_state() if generator is not None else None)


def restore_rng(state, generator=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if generator is not None and state["generator"] is not None:
        generator.set_state(state["generator"].cpu())
