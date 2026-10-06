import json

import numpy as np
import pytest
import torch

from data.cvrplib import load_cvrplib
from data.instances import fingerprint, generate, save_instances
from data.reference import reference_cache, save_reference
from eval.evaluate import evaluate_instances, load_reference, write_evaluation
from models.backbone import Policy
from models.critic import Critic
from train.losses import gaussian_nll
from test_backbone import small_config


def test_evaluation_modes_and_reference_identity(tmp_path):
    torch.manual_seed(1)
    model = Policy(small_config()["model"])
    inst = generate(3, 20)
    a = evaluate_instances(model, inst, batch_size=2)
    b = evaluate_instances(model, inst, batch_size=2, augmentation=True, return_routes=True)
    assert np.all(b["per_instance"].cost <= a["per_instance"].cost + 1e-5)
    assert b["summary"]["total_candidates_per_instance"] == 160
    c = evaluate_instances(model, inst, mode="sampling", samples=2)
    assert c["summary"]["samples"] == 2
    costs = a["per_instance"].cost.to_numpy()
    save_reference(tmp_path / "reference.npz", costs, dict(dataset_fingerprint=fingerprint(inst), objective="euclidean"))
    ref = load_reference(tmp_path / "reference.npz", inst)
    d = evaluate_instances(model, inst, reference=ref)
    assert abs(d["summary"]["mean_gap_percent"]) < 1e-4
    with pytest.raises(ValueError, match="exact instances"):
        load_reference(tmp_path / "reference.npz", generate(3, 20))
    write_evaluation(b, tmp_path / "report")
    assert len(json.loads((tmp_path / "report/routes.json").read_text())) == 3


def test_cvrplib_scale_rounding_and_depot(tmp_path):
    pytest.importorskip("vrplib")
    path = tmp_path / "A-tiny.vrp"
    path.write_text("NAME : A-tiny\nTYPE : CVRP\nDIMENSION : 4\nCAPACITY : 10\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n1 10 10\n2 20 12\n3 12 25\n4 25 25\nDEMAND_SECTION\n1 0\n2 3\n3 4\n4 5\nDEPOT_SECTION\n1\n-1\nEOF\n")
    inst, rounded, raw = load_cvrplib(path)
    assert inst.metadata["scale"] == 15
    assert torch.allclose(inst.distances.double()*15, raw, atol=2e-6)
    assert torch.equal(rounded, torch.floor(raw+0.5))
    model = Policy(small_config()["model"])
    result = evaluate_instances(model, inst, batch_size=1, metric_distance=rounded, unrounded_distance=raw)
    assert result["per_instance"].cost.iloc[0] == round(result["per_instance"].cost.iloc[0])


def test_reference_hgs_and_resume(tmp_path):
    pytest.importorskip("pyvrp")
    inst = generate(2, 3, torch.Generator().manual_seed(5), capacity=10)
    path = tmp_path / "hgs.npz"
    costs = reference_cache(inst, path, time_limit=0.02, seed=3)
    assert np.isfinite(costs).all() and (costs > 0).all()
    assert np.array_equal(reference_cache(inst, path, time_limit=0.02, seed=3), costs)
    assert np.array_equal(load_reference(path, inst), costs)


def test_critic_gradient_and_latent_replay():
    cfg = small_config("quantum")
    cfg["model"]["mve_latent"] = True
    model = Policy(cfg["model"])
    inst = generate(2, 20)
    old = model(inst)
    new = model(inst, actions=old["routes"], latent_eps=old["aux"]["eps"])
    assert torch.allclose(old["logprob_steps"], new["logprob_steps"], atol=1e-5)
    critic = Critic(32, True)
    mu, logvar = critic(new["h"], new["currents"], new["loads"])
    target = new["reward"][..., None].expand_as(mu).detach()
    loss = gaussian_nll(mu, logvar, target)[new["active"]].mean()
    loss.backward()
    assert critic.conv[0].weight.grad.norm() > 0
    assert model.customer_embedding.weight.grad.norm() > 0


def test_ppo_objective_without_optimizer_updates():
    from train.ppo import ppo_update
    cfg = small_config("quantum")
    cfg["model"]["mve_latent"] = True
    cfg["training"]["advantage"] = "mve"
    model, critic = Policy(cfg["model"]), Critic(32)
    parameters = list(model.parameters()) + list(critic.parameters())
    before = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters)

    class NoUpdates:
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            # Explicitly suppress optimizer.step: this is a computation check only.
            pass
        def update(self):
            pass

    result, terms, gradients = ppo_update(model, critic, generate(2, 20), optimizer, NoUpdates(), cfg, 6, 1.)
    assert np.isfinite(terms["loss_total"]) and gradients["grad_norm_quantum"] > 0
    assert all(torch.equal(a, b) for a, b in zip(before, parameters))
