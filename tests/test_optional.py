import pytest
import torch

from data.instances import generate
from models.backbone import Policy
from models.latent import LatentBlock
from train.losses import reinforce_loss
from train.utils import seed_all
from test_backbone import small_config


def test_reconstruction_shape_gradient_and_naming():
    cfg = small_config("classical_bottleneck")
    cfg["model"]["recon_loss"] = True
    cfg["model"]["mve_latent"] = True
    model = Policy(cfg["model"])
    result = model(generate(2, 20))
    loss, terms = reinforce_loss(result, cfg, progress=1.)
    loss.backward()
    assert terms["loss_recon"] > 0
    assert model.latent.reconstruction[0].weight.grad.norm() > 0
    assert model.latent.posterior.weight.grad.norm() > 0
    assert model.latent.component_name == "variational latent autoencoder"


def test_quantum_edge_knn_shapes_and_gradients():
    cfg = small_config("quantum")
    cfg["model"]["q_layer"] = 1
    cfg["model"]["n_qubits"] = 4
    cfg["model"]["quantum_edge_scoring"] = True
    model = Policy(cfg["model"])
    result = model(generate(2, 20))
    loss, _ = reinforce_loss(result, cfg)
    loss.backward()
    assert model.edge.encode.weight.grad.norm() > 0
    assert model.edge.circuit.angles.grad.norm() > 0
    assert result["cost"].shape == (2, 20)


@pytest.mark.training
@pytest.mark.parametrize("backend", ["classical_bottleneck", "quantum"])
def test_reconstruction_decreases_over_50_steps(backend):
    seed_all(4)
    cfg = small_config(backend)["model"]
    cfg["recon_loss"] = True
    cfg["mve_latent"] = True
    block = LatentBlock(cfg).train()
    x = torch.randn(4, 21, 32)
    eps = torch.zeros(4, 21, 6)  # fixed noise isolates reconstruction learning
    optimizer = torch.optim.Adam(block.parameters(), lr=0.01)
    initial = float(block(x, eps=eps)[1]["recon"].detach())
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        recon = block(x, eps=eps)[1]["recon"]
        recon.backward()
        optimizer.step()
    final = float(block(x, eps=eps)[1]["recon"].detach())
    assert final < initial*0.9


@pytest.mark.training
def test_three_optimizer_steps_reproducible():
    def run():
        seed_all(6)
        cfg = small_config("quantum")
        cfg["model"]["mve_latent"] = True
        model = Policy(cfg["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        values = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = reinforce_loss(model(generate(2, 20)), cfg, epoch=6, progress=1.)
            values.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        return values
    assert run() == run()
