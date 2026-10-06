from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from train.config import ROOT, load_config
from models.backbone import Policy
from train.utils import parameter_counts


def variants():
    base = load_config(ROOT / "configs/M3.yaml")
    matrix = {}
    for name in ["M0", "M1", "M2", "M3"]:
        matrix[name] = load_config(ROOT / f"configs/{name}.yaml")

    def add(name, **overrides):
        cfg = copy.deepcopy(base)
        cfg["name"] = name
        for key, value in overrides.items():
            section, field = key.split("__")
            cfg[section][field] = value
        cfg["description"] = ("Variational latent autoencoder" if cfg["model"]["recon_loss"] else "Latent bottleneck") + f" ablation: {name}"
        matrix[name] = cfg

    add("frozen_random", model__backend="frozen_random_quantum")
    add("latent_mve_only", training__advantage="none")
    add("advantage_mve_only", model__mve_latent=False)
    for mode in ["none", "batch_std", "instance_std", "mve"]:
        add(f"advantage_{mode}", training__advantage=mode)
    add("gate_zero", model__alpha_zero=True)
    add("zz", model__zz=True)
    for n in [4, 6, 8]:
        add(f"qubits_{n}", model__n_qubits=n)
    for depth in [1, 2, 3]:
        add(f"depth_{depth}", model__depth=depth)
    for layer in [1, 3, 5]:
        add(f"q_layer_{layer}", model__q_layer=layer)
    add("ppo", training__algorithm="ppo")
    add("ppo_no_mve_critic", training__algorithm="ppo", training__mve_critic=False)
    add("knn", model__knn=True)
    add("distance_none", model__distance_bias="none")
    add("distance_scalar", model__distance_bias="scalar")
    add("difficulty_encoder_gradients", model__difficulty_detach=False)
    for backend in ["quantum", "classical_bottleneck"]:
        for weight in [0., 0.01, 0.1]:
            add(f"recon_{backend}_{weight}", model__backend=backend, model__recon_loss=weight > 0,
                model__lambda_rec=weight)
    add("quantum_edges", model__quantum_edge_scoring=True)
    # Nearest feasible FFN width, with the REAL count difference recorded.
    target = parameter_counts(Policy(base["model"]))["total"]
    match = copy.deepcopy(matrix["M0"])
    original_count = parameter_counts(Policy(match["model"]))["total"]
    increment = match["model"]["layers"]*(2*match["model"]["d_model"]+1)
    width = max(1, round(match["model"]["ffn"]+(target-original_count)/increment))
    match["model"]["ffn"] = width
    match["name"] = "parameter_matched_M0"
    actual = parameter_counts(Policy(match["model"]))["total"]
    match["description"] = f"Nearest FFN parameter match: target={target}, actual={actual}, difference={actual-target}"
    matrix[match["name"]] = match
    return matrix


def main():
    p = argparse.ArgumentParser(description="Write complete ablation configs; performs no training")
    p.add_argument("--output", default="configs/ablations")
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, config in variants().items():
        (output / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote {len(variants())} configurations to {output}")


if __name__ == "__main__":
    main()
