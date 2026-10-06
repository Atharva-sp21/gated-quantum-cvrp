import torch

from data.instances import generate
from envs.cvrp import recompute_cost, validate_routes
from models.backbone import Policy
from train.config import load_config
from train.losses import reinforce_loss
from train.utils import parameter_counts, seed_all


def small_config(backend="none"):
    c = load_config(overrides=["customers=20", "device=cpu", "model.d_model=32", "model.heads=4",
                               "model.ffn=64", "model.layers=3", f"model.backend={backend}"])
    return c


def test_m0_end_to_end_backward():
    seed_all(7)
    cfg = small_config()
    model = Policy(cfg["model"])
    instances = generate(2, 20)
    out = model(instances)
    assert out["cost"].shape == (2, 20)
    assert out["h"].shape == (2, 21, 32)
    assert validate_routes(instances, out["routes"])
    assert torch.allclose(out["cost"], recompute_cost(instances.coords, out["routes"]), atol=1e-5)
    loss, terms = reinforce_loss(out, cfg)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert p.grad.isfinite().all(), name
    assert model.depot_embedding.weight.grad.norm() > 0
    counts = parameter_counts(model)
    assert counts["quantum"] == 0
    assert counts["total"] == counts["classical"]


def test_greedy_and_replay():
    seed_all(0)
    cfg = small_config()
    model = Policy(cfg["model"]).eval()
    inst = generate(2, 20)
    a = model(inst, mode="greedy")
    b = model(inst, actions=a["routes"])
    assert torch.equal(a["routes"], b["routes"])
    assert torch.allclose(a["logprob_steps"], b["logprob_steps"])


def test_forward_reproducibility_three_steps():
    def run():
        seed_all(32)
        cfg = small_config()
        model = Policy(cfg["model"])
        return [float(reinforce_loss(model(generate(2, 20)), cfg)[0].detach()) for _ in range(3)]
    assert run() == run()
