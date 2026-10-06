# Gated latent quantum transformer for CVRP

A complete PyTorch research pipeline for POMO multi-start CVRP, a gated latent bottleneck, and mean-variance estimation (MVE). **Built without training. No benchmark performance or quantum advantage is claimed.**

Includes M0–M3, 36 ablation presets, REINFORCE and PPO, a differentiable complex64 simulator, fixed validation/test data, reference caching, CVRPLIB evaluation, uncertainty/calibration, robustness, trainability analysis, paired statistics, and CSV/LaTeX/figure generation. See `ASSUMPTIONS.md` for methodological choices and limits, and `BUILD_REPORT.md` for checks actually run.

## GPU setup

Use Python 3.12 for the verified dependency set; the core supports Python 3.10+. Run commands from this repository directory. Choose the PyTorch wheel for your GPU driver using the [official installation instructions](https://pytorch.org/get-started/previous-versions/). For a compatible NVIDIA driver and CUDA 12.8 wheel:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-verified.txt
python -m pip install -e '.[test,reference]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
python -m pip freeze > environment-gpu.txt
```

For Python 3.10/3.11, install a compatible GPU PyTorch wheel and then `python -m pip install -e '.[test,reference]'`, allowing dependency resolution instead of using the Python 3.12 pins. Pennylane is an optional **test oracle only**, never part of the policy's forward pass. PyVRP is pinned to its HGS release 0.12.2; later releases may change the solver and API.

## Verify without training

```bash
python -m pytest -q
python -m train.main --config configs/M3.yaml --customers 20 --dry-run --set device=cuda training.batch_size=2
python -m train.main --config configs/ablations/ppo.yaml --customers 20 --dry-run --set device=cuda training.batch_size=2
```

`--dry-run` runs a forward/backward pass and **zero optimizer steps**, then exits without checkpoints. Default pytest skips every test that performs optimizer updates. The PPO computation test suppresses optimizer updates and verifies that parameters remain identical. Use `device=cpu` for CPU verification. CUDA/AMP behavior must be checked on your GPU; only CPU was available during this build.

The explicitly opt-in commands below **do train**. They were not run during construction:

```bash
python -m pytest -q --run-training
python -m scripts.sanity_learn --device cuda --steps 300 --seed 0
```

The learning sanity tool measures before/after CVRP20 costs and gaps against actual cached HGS solutions. It fails if they do not decrease. This small fixed-batch overfitting check is not a quality benchmark.

## Fixed datasets and reference solutions

The archive includes 10,000 validation and 10,000 test instances **for each of 20, 50, and 100 customers** (60,000 instances total). Each NPZ stores its independent seed, generator settings, count, split, capacity, and SHA-256 instance fingerprint. Training generates fresh batches on the fly. Recreate data without overwriting supplied sets:

```bash
python -m scripts.generate_data --output datasets-regenerated
```

To use the allowed compute-limited CVRP100 test setting, create a separate dataset directory and document the count:

```bash
python -m scripts.generate_data --test100-count 1000 --output datasets-small
```

References are **not precomputed** for the 60,000 supplied instances. Generate them on your machine before reporting any gaps:

```bash
python -m scripts.prepare_references --datasets datasets --time-limit 1 --seed 123
```

This is CPU HGS reference solving, not ML training. One second for each of 60,000 instances takes roughly 16.7 hours of solver budget plus overhead. The script checkpoints after each instance and resumes existing matching caches. Altering the time limit, seed, dataset, or solver version requires a new cache path/dataset directory. HGS provides feasible heuristic references, not proofs of optimality. Negative gaps are retained honestly.

For one set, or a supplied published reference CSV:

```bash
python -m scripts.reference_costs --dataset datasets/test_20.npz --output datasets/test_20_reference.npz --time-limit 1 --seed 123
python -m scripts.reference_costs --dataset datasets/test_20.npz --output datasets/test_20_published.npz --published-csv supplied_costs.csv --source 'Exact source and matching dataset provenance'
```

CSV columns are `instance_id,cost`, with consecutive zero-based IDs in the exact saved dataset order and continuous Euclidean costs. The user must establish that published instances match; the importer cannot infer this from costs alone. Evaluation rejects a missing, incomplete, or mismatched reference cache.

## Train one variant on your GPU

```bash
python -m train.main --config configs/M3.yaml --customers 100 --seed 0 --epochs 100 --set validation.dataset=datasets/validation_100.npz validation.reference=datasets/validation_100_reference.npz
```

Default batches are 256/128/64 for 20/50/100 customers. Defaults are 100 epochs and 100 batches per epoch; **POMO-quality results typically require far longer training**. Set `training.steps_per_epoch`, `--epochs`, or a time budget explicitly. Quantum simulation uses complex64 outside AMP; classical CUDA operations use AMP. If GPU memory is limited, reduce `training.batch_size` and `model.quantum_chunk_size`:

```bash
python -m train.main --config configs/M3.yaml --customers 100 --seed 0 --epochs 1000 --time-budget 86400 --set training.batch_size=16 model.quantum_chunk_size=512 validation.dataset=datasets/validation_100.npz validation.reference=datasets/validation_100_reference.npz output=runs-long
```

Time budgets include validation and are checked between training batches; a single batch/validation can exceed the limit. The best checkpoint minimizes deterministic validation cost. Training writes `best.pt`, `last.pt`, resolved config, runtime/parameter counts, epoch CSV, TensorBoard, completion status, and collapse diagnostics. Set `training.wandb=true` after installing `.[wandb]` to use your configured W&B account. Epoch logs include individual losses, pre-clipping quantum/classical gradient norms, KL per dimension, active dimensions, latent sigma, alpha, residual norm fraction, and reconstruction loss when enabled.

Resume with the **identical** resolved configuration and `--resume runs/M3_m100_seed0/last.pt`. Checkpoints restore optimizer, scheduler, AMP scaler, Python/NumPy/Torch/CUDA RNG, and the data-generator RNG. Validation and reference fingerprints must match. Resume occurs at the next saved epoch; an interrupted unsaved epoch is recomputed.

## Main paper tables and decoding figures

The runner defaults to **plan only**. It writes a human-readable JSON list of commands and never trains unless `--execute` is present:

```bash
python -m scripts.run_experiments --customers 100 --epochs 1000
python -m scripts.run_experiments --customers 100 --epochs 1000 --execute
python -m scripts.report --evaluations reports/main --runs runs --output reports/paper
```

This trains M0/M1/M2/M3 for seeds 0–4, and evaluates every checkpoint on sizes 20/50/100 with greedy POMO, greedy POMO ×8, and sampling S=8. Each sampling pass has P=m forced starts, so S passes produce S×P candidates (and another ×8 if augmentation is enabled). Candidate counts, S, decoding temperature, test fingerprints, and training/evaluation seeds are recorded explicitly.

Outputs: `results.csv`, `results.tex`, `statistics.csv`, `statistics.tex`, `inference_efficiency.csv/.tex`, `training_efficiency.csv/.tex`, `gap_vs_size.png`, `training_curves.png`, `training_losses.png`, `latent_diagnostics.png`, and `efficiency.png`. Efficiency tables distinguish policy parameter counts from complete training-module counts (including the PPO critic). Tables use seed-level mean ± sample standard deviation. Wilcoxon tests pair the same test instances after averaging costs across matching seeds; effect sizes are rank-biserial with positive values indicating improvement. Holm correction covers all emitted comparisons. Fewer than five seeds require explicit `--allow-incomplete` and are labeled exploratory. No table/plot template contains fabricated research numbers.

Direct evaluation and inference:

```bash
python -m eval.evaluate --checkpoint runs/M3_m100_seed0/best.pt --datasets datasets/test_20.npz datasets/test_50.npz datasets/test_100.npz --references datasets/test_20_reference.npz datasets/test_50_reference.npz datasets/test_100_reference.npz --augment --output reports/direct
python -m eval.evaluate --checkpoint runs/M3_m100_seed0/best.pt --datasets datasets/test_100.npz --references datasets/test_100_reference.npz --mode sampling --samples 128 --temperature 1 --output reports/sampling128
python -m scripts.infer --checkpoint runs/M3_m100_seed0/best.pt --customers 100 --count 4 --augment --output reports/inference
```

Inference saves depot-delimited routes and verifies customer coverage and capacity. `eval.evaluate --save-routes` saves customer/depot sequences with the initial and final depot implicit. Add `--sample-latent` to sample the MVE posterior during evaluation sampling; greedy evaluation always uses posterior means.

## Ablation table

```bash
python -m scripts.experiment_matrix
python -m scripts.run_experiments --configs configs/ablations/latent_mve_only.yaml configs/ablations/advantage_mve_only.yaml configs/ablations/advantage_batch_std.yaml configs/ablations/advantage_instance_std.yaml configs/ablations/frozen_random.yaml configs/ablations/gate_zero.yaml --customers 100 --epochs 1000 --run-output runs-ablations --report-output reports/ablations --manifest reports/ablation_manifest.json --execute
python -m scripts.report --evaluations reports/ablations --runs runs-ablations --baseline latent_mve_only --output reports/ablation-tables
```

Use any generated config list with the same runner and five seeds. All four advantage modes, MVE-only variants, frozen circuit, gate-zero, ZZ, qubits 4/6/8, depth 1/2/3, insertion layer 1/3/5, PPO/no-MVE-critic, distance none/scalar/MLP, kNN, difficulty-detach, reconstruction weights 0/0.01/0.1 for quantum and classical, parameter matching, and quantum edges are included. M0/M1/M2 have no latent MVE by default; M3 enables both latent MVE and MVE normalization. The difficulty head is present in all main variants for diagnostics, detached from the encoder by default.

The classical circuit replacement has a **comparable**, not automatically identical, parameter count. The classical backbone parameter match chooses the nearest FFN width and records its exact count difference. Include the matching counts in your paper rather than calling it exact.

## Uncertainty and shifted-distribution figures

```bash
python -m eval.uncertainty --per-instance reports/main/M3_m100_seed0/greedy/test_100/per_instance.csv --output reports/uncertainty
python -m scripts.distribution_shift --checkpoints runs/M2_m100_seed0/best.pt runs/M3_m100_seed0/best.pt --customers 100 --count 1000 --build-references --time-limit 1 --output reports/shifted
```

These emit `sigma_vs_gap.png`, `calibration.png`, `calibration.csv`, and uncertainty JSON, plus held-out shifted costs/gaps. Repeat with all five checkpoint seeds for research comparisons. The shift tool creates identical clustered or higher-demand instances for both variants, caches new references, and uses separately trained models. It does not retrospectively toggle a training-time ablation. Calibration checks coverage for **mean rollout cost**, the difficulty head's target; best-of-P gaps have a different meaning. The script retains absent/constant uncertainty values as unavailable.

## Gradient-depth/qubit and robustness figures

Collect during-training gradients with `training.trainability_every=10` (the multi-seed runner enables this). Initialization analysis uses backward passes and zero optimizer updates:

```bash
python -m scripts.trainability --config configs/M3.yaml --qubits 4 6 8 10 --depths 1 2 3 4 --seeds 0 1 2 3 4 --customers 20 --runs runs/M3_m100_seed0 --output reports/trainability
python -m scripts.robustness --checkpoint runs/M3_m100_seed0/best.pt --dataset datasets/test_100.npz --reference datasets/test_100_reference.npz --batch-size 8 --output reports/robustness
python -m scripts.efficiency --checkpoint runs/M3_m100_seed0/best.pt --dataset datasets/test_100.npz --count 100 --output reports/efficiency.json
```

Trainability emits `gradients.csv`, `gradient_table.csv/.tex`, `gradient_vs_depth.png`, and, when run logs are supplied, `gradient_during_training.png`. Train each desired depth/qubit configuration separately to measure its trained gradients; the script never pretends an architecture sweep checkpoint is trained. Norms, per-parameter RMS, and near-zero gradient fraction at initialization are reported; a small gradient alone does not prove a barren plateau.

Robustness evaluates all nine combinations of shots {∞,1024,128} and p {0,0.001,0.01}, saving per-instance outputs, `robustness.csv`, and `noise_robustness.png`. Depolarizing simulation is a full density-matrix path and requires ≤8 qubits. Finite-shot outputs are independent marginal binomial approximations (Gaussian option is available through the simulator API), not a joint hardware measurement model. Efficiency records measured inference time and CUDA peak memory, plus the explicitly limited one-state simulator allocation estimate. Training time per epoch is already in `metrics.csv`.

## CVRPLIB generalisation

Supply your local A/B/P/X `.vrp` files and matching best-known `.sol` files:

```bash
python -m scripts.evaluate_cvrplib --checkpoint runs/M3_m100_seed0/best.pt --instances cvrplib/A/*.vrp cvrplib/B/*.vrp cvrplib/P/*.vrp cvrplib/X/*.vrp --solution-dir cvrplib/solutions --rounding nearest --reference-rounding nearest --augment --output reports/cvrplib
```

Coordinates use an isotropic min-max scale; its offset/scale and original node order are retained. The saved reports contain unscaled continuous costs, nearest-integer rounded costs, the selected comparison convention, and gaps against supplied best-known costs. Rounding is `floor(d+0.5)`, avoiding bankers' rounding. Explicit distance matrices can be selected with `--rounding explicit --reference-rounding explicit`. Do not compare continuous costs against a rounded reference. CVRPLIB routes use converted node indices; map through `original_node_order` to recover original zero-based node indices. The neural policy always encodes Euclidean geometry; selection among its candidate routes uses the requested evaluation objective.

## Optional circuit hardware check

```bash
python -m pip install -e '.[hardware]'
python -m scripts.ibm_hardware_check --checkpoint runs/M3_m100_seed0/best.pt --dataset datasets/test_20.npz --backend aer --max-nodes 8 --shots 1024
python -m scripts.ibm_hardware_check --checkpoint runs/M3_m100_seed0/best.pt --dataset datasets/test_20.npz --backend fake --max-nodes 8 --shots 1024
```

`--backend ibm --ibm-backend YOUR_BACKEND_NAME` explicitly submits a job using a previously configured IBM account. This isolated tool compares a few node expectations, not complete hardware-trained CVRP tours. Qiskit/IBM execution was not verified during this build. It uses the documented [IBM Sampler V2 interface](https://quantum.cloud.ibm.com/docs/en/guides/sampler-options) and [Aer simulation](https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html).

## Repository layout and provenance

`data/`, `envs/`, `models/`, `quantum/`, `train/`, `eval/`, `configs/`, `tests/`, and `scripts/` match the requested layout. `datasets/` contains the saved fixed sets. No PyTorch Geometric is used. Research background: [POMO](https://arxiv.org/abs/2010.16011); gate convention: [PennyLane Rot](https://docs.pennylane.ai/en/stable/code/api/pennylane.Rot.html); data-rounding caveat: [VRPLIB documentation](https://github.com/PyVRP/VRPLIB).
