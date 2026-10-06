import pytest
import torch

from data.instances import generate
from models.backbone import Policy
from models.latent import LatentBlock
from train.losses import reinforce_loss
from train.utils import parameter_counts, seed_all
from test_backbone import small_config


def test_classical_block_and_gradients():
    cfg = small_config("classical_bottleneck")
    model = Policy(cfg["model"])
    result = model(generate(2, 20))
    loss, _ = reinforce_loss(result, cfg)
    loss.backward()
    for name, p in model.latent.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all(), name
    assert result["aux"]["contribution"] > 0
    assert parameter_counts(model)["quantum"] == 0


def test_zero_gate_exact_noop():
    cfg = small_config("classical_bottleneck")["model"]
    cfg["alpha_zero"] = True
    block = LatentBlock(cfg)
    x = torch.randn(2, 21, 32)
    y, _ = block(x)
    assert torch.equal(x, y)
    assert "alpha" not in dict(block.named_parameters())


@pytest.mark.parametrize("backend", ["quantum", "frozen_random_quantum"])
def test_quantum_integration(backend):
    seed_all(8)
    cfg = small_config(backend)
    model = Policy(cfg["model"])
    result = model(generate(2, 20))
    loss, _ = reinforce_loss(result, cfg)
    loss.backward()
    p = model.latent.circuit.angles
    assert (p.grad is not None and p.grad.norm() > 0) if backend == "quantum" else p.grad is None
    assert model.latent.compress.weight.grad.norm() > 0
    assert parameter_counts(model)["quantum"] == 3*6*2


def test_mve_latent_sampling_eval_and_kl():
    cfg = small_config("quantum")
    cfg["model"]["mve_latent"] = True
    model = Policy(cfg["model"])
    inst = generate(2, 20)
    _, _, a = model.encode(inst)
    _, _, b = model.encode(inst)
    assert not torch.equal(a["z"], b["z"])
    assert a["logvar"].min() >= -6 and a["logvar"].max() <= 2
    model.eval()
    _, _, a = model.encode(inst, sample_latent=False)
    _, _, b = model.encode(inst, sample_latent=False)
    assert torch.equal(a["z"], a["mu"]) and torch.equal(a["z"], b["z"])
    model.train()
    result = model(inst)
    loss, terms = reinforce_loss(result, cfg, epoch=6, progress=1.)
    assert terms["loss_kl"] >= 6*cfg["training"]["free_bits"]
    loss.backward()
    assert model.latent.posterior.weight.grad.norm() > 0
