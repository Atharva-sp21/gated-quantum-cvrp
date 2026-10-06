import pytest
import torch

from quantum.simulator import StatevectorCircuit, apply_gate, ry


def test_known_rotation():
    state = torch.tensor([[1, 0, 0, 0]], dtype=torch.complex64)
    out = apply_gate(state, ry(torch.tensor(torch.pi)), 0, 2)
    assert torch.allclose(out.abs().square(), torch.tensor([[0., 0., 1., 0.]]), atol=1e-6)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_output_and_gradients_vs_pennylane(n):
    qml = pytest.importorskip("pennylane")
    torch.manual_seed(n)
    circuit = StatevectorCircuit(n, 2, zz=True)
    x = torch.randn(2, n, requires_grad=True)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def oracle(theta, angles):
        for l in range(2):
            for q in range(n):
                qml.RY(theta[q], wires=q)
                qml.Rot(*angles[l, q], wires=q)
            for q in range(n):
                qml.CNOT(wires=[q, (q+1) % n])
        return [qml.expval(qml.PauliZ(q)) for q in range(n)] + [
            qml.expval(qml.PauliZ(q) @ qml.PauliZ((q+1) % n)) for q in range(n)]

    expected = torch.stack([torch.stack(oracle(row.double(), circuit.angles.double())) for row in x])
    actual = circuit(x)
    assert torch.allclose(actual.double(), expected, atol=1e-5, rtol=1e-5)
    ga = torch.autograd.grad(actual.square().sum(), (x, circuit.angles), retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), (x, circuit.angles))
    for a, b in zip(ga, gb):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_chunking_noise_and_gradient():
    torch.manual_seed(1)
    q = StatevectorCircuit(4, 2, zz=True, chunk_size=2)
    x = torch.randn(5, 4, requires_grad=True)
    y = q(x)
    assert y.shape == (5, 8) and y.abs().max() <= 1
    y.square().sum().backward()
    assert x.grad.isfinite().all() and q.angles.grad.norm() > 0
    with pytest.raises(ValueError, match="inference-only"):
        q(x, shots=128)
    q.eval()
    assert torch.allclose(y, q(x, depolarizing=0), atol=1e-6)
    noisy = q(x.detach(), depolarizing=0.01)
    assert noisy.shape == y.shape and noisy.abs().max() <= 1
    assert not torch.allclose(y, noisy)
    samples = q(x.detach(), shots=1024)
    assert ((samples+1)*512 - ((samples+1)*512).round()).abs().max() < 1e-4
    assert q.memory_bytes(5) == 5*16*8


def test_density_against_unitary_limit():
    q = StatevectorCircuit(2, 2).eval()
    x = torch.randn(3, 2)
    assert torch.allclose(q(x), q(x, depolarizing=1e-8), atol=1e-5)


def test_frozen():
    q = StatevectorCircuit(4, frozen=True)
    assert not q.angles.requires_grad


def test_density_vs_pennylane_mixed():
    qml = pytest.importorskip("pennylane")
    torch.manual_seed(7)
    circuit = StatevectorCircuit(4, 2, zz=True).eval()
    theta = torch.randn(4)
    dev = qml.device("default.mixed", wires=4)

    @qml.qnode(dev)
    def oracle():
        for layer in circuit.angles.detach().numpy():
            for q in range(4):
                qml.RY(float(theta[q]), q)
                qml.Rot(*layer[q], wires=q)
            for q in range(4):
                qml.CNOT(wires=[q, (q+1) % 4])
            for q in range(4):
                qml.DepolarizingChannel(0.01, wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(4)] + [
            qml.expval(qml.PauliZ(q) @ qml.PauliZ((q+1) % 4)) for q in range(4)]

    actual = circuit(theta[None], depolarizing=0.01).detach().numpy()[0]
    import numpy as np
    assert np.allclose(actual, oracle(), atol=1e-5)
