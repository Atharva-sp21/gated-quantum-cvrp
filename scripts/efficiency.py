import argparse
from pathlib import Path

import torch

from data.instances import load_instances
from eval.evaluate import evaluate_instances, load_checkpoint
from train.utils import parameter_counts, runtime_metadata, save_json


def main():
    p = argparse.ArgumentParser(description="Measure actual inference and simulator allocation size")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="reports/efficiency.json")
    a = p.parse_args()
    model, cfg = load_checkpoint(a.checkpoint, a.device)
    inst = load_instances(a.dataset).slice(0, a.count)
    evaluate_instances(model, inst.slice(0, min(a.batch_size, len(inst.coords))), verify=False)
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    result = evaluate_instances(model, inst, batch_size=a.batch_size, verify=False)
    circuit = model.latent.circuit if model.latent is not None else None
    allocated = circuit.memory_bytes(min(cfg["model"]["quantum_chunk_size"], a.batch_size*(inst.size+1))) if hasattr(circuit, "memory_bytes") else 0
    report = dict(parameter_counts=parameter_counts(model), runtime=runtime_metadata(),
                  inference_seconds_per_instance=result["summary"]["inference_seconds_per_instance"],
                  quantum_one_state_allocation_bytes=allocated,
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith("cuda") else None,
                  note="one-state size excludes gate temporaries and autograd; CUDA peak is measured after warm-up",
                  checkpoint=a.checkpoint, customers=inst.size, count=len(inst.coords))
    save_json(a.output, report)
    print(report)


if __name__ == "__main__":
    main()
