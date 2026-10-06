import pytest
import torch

from data.instances import generate
from models.backbone import Policy
from train.losses import gaussian_nll, reinforce_loss
from test_backbone import small_config


@pytest.mark.parametrize("mode", ["none", "batch_std", "instance_std", "mve"])
def test_advantage_modes(mode):
    cfg = small_config("quantum")
    cfg["model"]["mve_latent"] = True
    cfg["training"]["advantage"] = mode
    model = Policy(cfg["model"])
    result = model(generate(2, 20))
    loss, terms = reinforce_loss(result, cfg, epoch=6, progress=0.1)
    assert torch.isfinite(loss)
    assert terms["beta"] == pytest.approx(cfg["training"]["beta_max"]/2)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(), name


def test_difficulty_detached_and_sigma_detached():
    cfg = small_config()
    cfg["training"]["advantage"] = "mve"
    model = Policy(cfg["model"])
    result = model(generate(2, 20))
    _, terms = reinforce_loss(result, cfg, epoch=6)
    grad = torch.autograd.grad(terms["loss_policy"], result["aux"]["logvar_c"], allow_unused=True, retain_graph=True)[0]
    assert grad is None
    terms["loss_difficulty"].backward()
    assert model.customer_embedding.weight.grad is None
    assert model.difficulty[-1].weight.grad.norm() > 0


def test_nll_formula():
    mu, var, target = torch.tensor([1.]), torch.tensor([0.]), torch.tensor([3.])
    assert gaussian_nll(mu, var, target).item() == 2
