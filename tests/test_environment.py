import itertools

import pytest
import torch

from data.instances import augment8, fingerprint, generate, load_instances, save_instances
from envs.cvrp import CVRPEnv, recompute_cost, validate_routes


def tiny_optimum(inst):
    """Exhaust all customer orders and route cuts (small correctness oracle)."""
    best = float("inf")
    for order in itertools.permutations(range(1, inst.size + 1)):
        for cuts in itertools.product([False, True], repeat=inst.size-1):
            route, load, valid = [], 1., True
            for i, a in enumerate(order):
                if i and cuts[i-1]:
                    route.append(0)
                    load = 1.
                load -= float(inst.demands[0, a])
                valid &= load >= -1e-6
                route.append(a)
            if valid:
                c = recompute_cost(inst.coords, torch.tensor(route)[None, None]).item()
                best = min(best, c)
    return best


@pytest.mark.parametrize("m", [3, 20, 50, 100])
def test_feasibility_and_cost(m):
    inst = generate(3, m, torch.Generator().manual_seed(10), capacity=30 if m == 3 else None)
    env = CVRPEnv(inst)
    env.start()
    while not env.done.all():
        scores = torch.rand(env.B, env.P, env.N).masked_fill(env.mask(), -1)
        env.step(scores.argmax(-1))
    assert validate_routes(inst, env.route_tensor)
    assert torch.allclose(env.cost, recompute_cost(inst.coords, env.route_tensor), atol=1e-5)
    assert torch.all(env.load >= -1e-6)
    assert env.route_tensor.shape[-1] <= 2*m-1


def test_mask_and_padding():
    inst = generate(2, 3, capacity=9)
    env = CVRPEnv(inst)
    assert env.mask()[..., 0].all()
    with pytest.raises(ValueError, match="infeasible"):
        env.step(torch.zeros(2, 3, dtype=torch.long))
    env.start()
    while not env.done.all():
        env.step((~env.mask()).float().argmax(-1))
    cost = env.cost.clone()
    env.step(torch.zeros(2, 3, dtype=torch.long))
    assert torch.equal(cost, env.cost)


def test_augmentation_optimum():
    inst = generate(1, 3, torch.Generator().manual_seed(7), capacity=10)
    aug = augment8(inst)
    optimum = tiny_optimum(inst)
    for i in range(8):
        assert tiny_optimum(aug.slice(i, i+1)) == pytest.approx(optimum, abs=1e-6)
    assert torch.allclose(aug.distances, inst.distances.expand(8, -1, -1), atol=1e-6)


def test_dataset_roundtrip(tmp_path):
    a = generate(4, 20, torch.Generator().manual_seed(3))
    b = generate(4, 20, torch.Generator().manual_seed(3))
    assert fingerprint(a) == fingerprint(b)
    save_instances(tmp_path / "test.npz", a, seed=3)
    c = load_instances(tmp_path / "test.npz")
    assert torch.equal(a.coords, c.coords)
    assert c.metadata["seed"] == 3
