"""Optional isolated Qiskit check. IBM submission only occurs with --backend ibm."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.instances import load_instances
from eval.evaluate import load_checkpoint
from train.utils import save_json, seed_all


def expectations_from_counts(counts, n, zz=False):
    shots = sum(counts.values())
    out = np.zeros(n*(2 if zz else 1))
    for bits, count in counts.items():
        bits = bits.replace(" ", "").zfill(n)
        signs = np.array([1-2*int(bits[-1-q]) for q in range(n)])
        values = np.concatenate([signs, signs*np.roll(signs, -1)]) if zz else signs
        out += values*count/shots
    return out


def qiskit_circuit(theta, angles, depolarizing=0.):
    from qiskit import QuantumCircuit
    n = len(theta)
    circuit = QuantumCircuit(n)
    for layer in angles:
        for q, (phi, th, omega) in enumerate(layer):
            circuit.ry(float(theta[q]), q)
            circuit.rz(float(phi), q)
            circuit.ry(float(th), q)
            circuit.rz(float(omega), q)
        for q in range(n):
            circuit.cx(q, (q+1) % n)
        if depolarizing:
            from qiskit_aer.noise import pauli_error
            error = pauli_error([("I", 1-depolarizing), ("X", depolarizing/3),
                                 ("Y", depolarizing/3), ("Z", depolarizing/3)]).to_instruction()
            for q in range(n):
                circuit.append(error, [q])
    circuit.measure_all()
    return circuit


def main():
    p = argparse.ArgumentParser(description="Compare a few trained node circuits with Aer/fake/IBM expectations")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--backend", choices=["aer", "fake", "ibm"], default="aer")
    p.add_argument("--ibm-backend", help="explicit IBM backend name; uses previously saved account")
    p.add_argument("--max-nodes", type=int, default=8)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--depolarizing", type=float, default=0.)
    p.add_argument("--output", default="reports/hardware")
    a = p.parse_args()
    if a.backend == "ibm" and not a.ibm_backend:
        p.error("IBM submission requires an explicit --ibm-backend")
    if a.backend != "aer" and a.depolarizing:
        p.error("explicit layer depolarizing channel only supported by --backend aer")
    if a.max_nodes < 1 or a.shots < 1:
        p.error("max-nodes and shots must be positive")
    seed_all(123)
    model, cfg = load_checkpoint(a.checkpoint, "cpu")
    if cfg["model"]["backend"] not in {"quantum", "frozen_random_quantum"}:
        p.error("quantum checkpoint required")
    inst = load_instances(a.dataset).slice(0, 1)
    with torch.no_grad():
        _, _, aux = model.encode(inst, sample_latent=False)
        theta = (math.pi*torch.tanh(aux["z"])).reshape(-1, cfg["model"]["n_qubits"])[:a.max_nodes]
        circuit = model.latent.circuit
        expected = circuit(theta, depolarizing=a.depolarizing).numpy()
        ideal = circuit(theta).numpy()
    circuits = [qiskit_circuit(row.numpy(), circuit.angles.detach().numpy(), a.depolarizing) for row in theta]
    if a.backend == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler import generate_preset_pass_manager
        backend = QiskitRuntimeService().backend(a.ibm_backend)
        compiled = generate_preset_pass_manager(backend=backend, optimization_level=1).run(circuits)
        job = SamplerV2(mode=backend).run(compiled, shots=a.shots)
        raw = job.result()
        counts = [entry.data.meas.get_counts() for entry in raw]
        job_id = job.job_id()
    else:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        if a.backend == "fake":
            from qiskit.providers.fake_provider import GenericBackendV2
            backend = AerSimulator.from_backend(GenericBackendV2(cfg["model"]["n_qubits"], seed=123))
        else:
            backend = AerSimulator(method="density_matrix" if a.depolarizing else "statevector")
        compiled = transpile(circuits, backend, seed_transpiler=123)
        result = backend.run(compiled, shots=a.shots, seed_simulator=123).result()
        counts = [result.get_counts(i) for i in range(len(circuits))]
        job_id = None
    measured = np.stack([expectations_from_counts(count, circuit.n_qubits, circuit.zz) for count in counts])
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [dict(node=i, observable=j, simulator=float(expected[i, j]), ideal_simulator=float(ideal[i, j]),
                 measured=float(measured[i, j]), absolute_error=float(abs(measured[i, j]-expected[i, j])))
            for i in range(len(measured)) for j in range(circuit.n_out)]
    pd.DataFrame(rows).to_csv(output / "hardware_comparison.csv", index=False)
    save_json(output / "metadata.json", dict(backend=a.backend, ibm_backend=a.ibm_backend, job_id=job_id,
              shots=a.shots, node_circuits=len(circuits), depolarizing=a.depolarizing,
              note="fake/IBM include device errors; compare ideal values without claiming noise equivalence"))
    print(f"Mean absolute expectation error: {np.abs(measured-expected).mean():.6g}")


if __name__ == "__main__":
    main()
