# Build and verification report

Built on 6 October 2026. **No optimizer updates were performed. No trained checkpoints or benchmark results are supplied.**

## Final verification

- Python 3.12.14, PyTorch 2.8.0, CPU; full environment versions are in `verification/environment.json` and `verification/requirements-freeze.txt`.
- **54 pytest checks passed; 3 optimizer-update checks were deliberately skipped**. JUnit output is in `verification/pytest.xml`.
- All 17 CLI entry points passed import/`--help` checks.
- Editable package build/install and compilation of all Python sources succeeded.
- Full-width M3 REINFORCE and PPO dry runs completed forward/backward passes with `optimizer_steps=0`.
- The full runner's logging, best/last checkpoint serialization, RNG state, and resume path were exercised in temporary tests with Adam.step disabled. Tests assert unchanged model parameters and empty optimizer state. Temporary checkpoints are not included.
- PennyLane `default.qubit` outputs and input/circuit gradients match at tolerance 1e-5 for n=2/4/6, depth 2, including ZZ outputs. The depolarizing density-matrix path also matches `default.mixed` at tolerance 1e-5 for four qubits.
- Generated and fingerprint-verified 60,000 fixed instances: 10,000 validation and 10,000 test instances for each size 20/50/100. Manifest: `datasets/manifest.json`.
- Included 4 main configs and 36 generated experiment/ablation configs. Multi-seed planning emitted 80 train/evaluation commands without executing them.

## Requested build sequence

| Stage | Implemented and checked before proceeding |
|---|---|
| 1. Environment/data | 7 checks: capacity, coverage, padding, recomputed cost, exact tiny symmetry oracle, fixed-seed roundtrip. |
| 2. Simulator | Gate ordering, circuit autograd, chunking, finite shots, density matrix, frozen weights. Independent PennyLane checks completed once verification dependencies were installed. |
| 3. Classical M0 | Feasible m=20 multi-start rollouts, decoder replay, encoder/decoder backward flow, three seeded forward losses. Learning sanity tool supplied and left unexecuted per user instruction. |
| 4. Classical latent block | Gated compression/expansion, classical backend gradients, exact gate-zero no-op, parameter counting. |
| 5. Quantum backend | Integrated trainable/frozen circuit gradients and independent PennyLane output/gradient oracle. |
| 6. Latent MVE | Stochastic training forward, deterministic evaluation means, clamped logvar, per-dimension KL/free-bits, posterior gradients. |
| 7. MVE advantage scaling | Four normalization modes, detached sigma scaling, detached difficulty representation, NLL and annealing checks. Full suite: 28 passing. |
| 8. Evaluation/analyses | Greedy/sampling/x8 selection, exact reference fingerprints, tiny actual HGS/cache resumption, CVRPLIB scale and rounding, critic/replayed latent checks. Full suite: 32 passing. |
| 9. Multi-seed/reporting | Five-seed requirements, matched IDs/references, Wilcoxon effect direction/ties, Holm, CSV/LaTeX output, PPO computation with updates disabled. Full suite: 36 passing. |
| 10. Optional variants | Reconstruction gradients/naming and quantum nearest-neighbor edge gradients. 50-step reconstruction/optimizer reproducibility checks supplied but skipped. Full suite then: 38 passing, 3 skipped. |

Final additional checks cover all attention normalization/distance switches, cross-size inference shapes, all checked-in configs, density-matrix oracle, CLI/package installation, and the runner's save/resume path without optimization.

## Explicitly unverified

Learning improvement and GPU performance; CUDA AMP/determinism; actual five-seed research experiments; complete reference caches; trained uncertainty/calibration; trained noise robustness; barren-plateau conclusions; 50-step reconstruction improvement; IBM/Qiskit hardware execution; exact reproduction of the unspecified optional GAT/Q-GAT paper.

`README.md` gives commands for these measurements. `ASSUMPTIONS.md` explains conventions and deviations. Reports never assert quantum benefit or substitute synthetic fixtures for research results. Toy report fixtures live only in temporary pytest directories.
