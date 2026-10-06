from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = dict(
    name="M0", seed=0, device="cuda", customers=100, capacity=None,
    model=dict(d_model=128, heads=8, ffn=512, layers=6, norm="instance", distance_bias="mlp",
               knn=False, backend="none", q_layer=3, n_qubits=6, depth=2, zz=False,
               mve_latent=False, eval_latent_sampling=False, alpha_init=0.1, alpha_zero=False,
               recon_loss=False, lambda_rec=0.01, difficulty_detach=True, difficulty_logvar_min=-10.,
               difficulty_logvar_max=10., quantum_edge_scoring=False,
               quantum_chunk_size=2048, classical_width=None),
    training=dict(algorithm="pomo_reinforce", advantage="none", advantage_clip=10., eps=1e-6,
                  epochs=100, steps_per_epoch=100, batch_size=None, lr=1e-4, lr_decay=1.,
                  grad_clip=1., amp=True, augmentation=False, beta_max=1e-3, free_bits=0.01,
                  lambda_c=1., difficulty_warmup=5, ppo_epochs=3, ppo_clip=0.2,
                  entropy=0.01, mve_critic=True, critic_warmup=5, value_weight=0.5,
                  time_budget_seconds=None, deterministic=True, tensorboard=True, wandb=False,
                  trainability_every=0),
    validation=dict(dataset=None, reference=None, batch_size=64, augmentation=False),
    output="runs", description="Classical POMO backbone")


def merge(base, patch):
    result = copy.deepcopy(base)
    for key, val in patch.items():
        if key not in result:
            raise ValueError(f"unknown configuration key: {key}")
        result[key] = merge(result[key], val) if isinstance(val, dict) and isinstance(result[key], dict) else val
    return result


def load_config(path=None, overrides=()):
    cfg = copy.deepcopy(DEFAULT)
    if path:
        with open(path) as f:
            cfg = merge(cfg, yaml.safe_load(f) or {})
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError("overrides use dotted.key=value")
        obj = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            obj = obj[part]
        if parts[-1] not in obj:
            raise ValueError(f"unknown override {key}")
        obj[parts[-1]] = yaml.safe_load(value)
    validate_config(cfg)
    return cfg


def validate_config(c):
    m, t = c["model"], c["training"]
    if m["backend"] not in {"none", "quantum", "classical_bottleneck", "frozen_random_quantum"}:
        raise ValueError("invalid backend")
    if m["d_model"] % m["heads"] or m["layers"] < 1:
        raise ValueError("invalid encoder dimensions")
    if not 1 <= m["q_layer"] <= m["layers"]:
        raise ValueError("q_layer is 1-based and must be within the encoder")
    if m["norm"] not in {"instance", "batch", "layer"} or m["distance_bias"] not in {"none", "scalar", "mlp"}:
        raise ValueError("invalid normalization/distance bias")
    if not 2 <= m["n_qubits"] <= 12 or m["depth"] < 1:
        raise ValueError("invalid circuit dimensions")
    if t["advantage"] not in {"none", "batch_std", "instance_std", "mve"}:
        raise ValueError("invalid advantage normalization")
    if t["algorithm"] not in {"pomo_reinforce", "ppo"}:
        raise ValueError("invalid algorithm")
    if t["epochs"] < 1 or t["steps_per_epoch"] < 1:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if c["customers"] not in {20, 50, 100} and not c["capacity"]:
        raise ValueError("nonstandard sizes require capacity")
    if m["backend"] == "none" and (m["recon_loss"] or m["quantum_edge_scoring"] or m["mve_latent"]):
        raise ValueError("latent options require a latent backend")
    if m["quantum_edge_scoring"] and m["q_layer"] == m["layers"]:
        raise ValueError("edge scoring requires a later encoder layer")
    if t["time_budget_seconds"] is not None and t["time_budget_seconds"] <= 0:
        raise ValueError("time budget must be positive")


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(ROOT / "configs/M3.yaml"))
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    return p
