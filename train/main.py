from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
import yaml

from data.instances import augment8, generate, load_instances
from models.backbone import Policy
from train.config import load_config, parser
from train.losses import reinforce_loss
from train.utils import gradient_norms, parameter_counts, restore_rng, rng_state, runtime_metadata, save_json, seed_all


def scalar_metrics(result, terms):
    metrics = {key: float(value.detach()) if torch.is_tensor(value) else float(value) for key, value in terms.items()}
    metrics.update(cost_mean=float(result["cost"].detach().mean()), cost_best=float(result["cost"].detach().amin(-1).mean()))
    for key in ["alpha", "contribution", "sigma_latent", "active_dimensions"]:
        if key in result["aux"]:
            metrics[key] = float(result["aux"][key].detach())
    if "kl_dimensions" in result["aux"]:
        for i, value in enumerate(result["aux"]["kl_dimensions"].detach()):
            metrics[f"kl_dim_{i}"] = float(value)
    return metrics


def collapse_report(metrics, model_cfg, path):
    flags = []
    if model_cfg["backend"] != "none":
        if metrics.get("alpha", 1) < 0.02:
            flags.append("alpha < 0.02: possible gate bypass (also inspect negative alpha magnitude)")
        if model_cfg["mve_latent"]:
            if metrics.get("active_dimensions", 99) <= 2:
                flags.append("active dimensions <= 2: possible posterior collapse")
            if metrics.get("sigma_latent", 0) > 0.9:
                flags.append("mean latent sigma > 0.9: possible weak information in posterior")
    text = "Collapse diagnostics\n\n" + ("\n".join(flags) if flags else "No configured heuristic collapse flags triggered.")
    text += "\nThese are diagnostics, not proof of collapse or benefit. Compare held-out ablations.\n"
    Path(path).write_text(text)


def train(cfg, resume=None, dry_run=False):
    seed_all(cfg["seed"], cfg["training"]["deterministic"])
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --set device=cpu for verification")
    t = cfg["training"]
    generator = torch.Generator().manual_seed(cfg["seed"] + 9173)
    batch_size = t["batch_size"] or {20: 256, 50: 128, 100: 64}.get(cfg["customers"], 64)
    model = Policy(cfg["model"]).to(device)
    modules = torch.nn.ModuleDict({"policy": model})
    critic = None
    if t["algorithm"] == "ppo":
        from models.critic import Critic
        critic = Critic(cfg["model"]["d_model"], t["mve_critic"]).to(device)
        modules["critic"] = critic
    counts = parameter_counts(modules)
    print(json.dumps(dict(parameter_counts=counts, runtime=runtime_metadata()), indent=2))
    amp = bool(t["amp"] and device.type == "cuda")
    optimizer = torch.optim.Adam((p for p in modules.parameters() if p.requires_grad), lr=t["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=t["lr_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if dry_run:
        inst = generate(min(batch_size, 2), cfg["customers"], generator, capacity=cfg["capacity"]).to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            result = model(inst)
            loss, terms = reinforce_loss(result, cfg, epoch=t["difficulty_warmup"], progress=1.)
            if critic is not None:
                value, logvar = critic(result["h"], result["currents"], result["loads"])
                target = result["reward"][..., None].expand_as(value).detach()
                active = result["active"]
                loss = loss + ((value-target).square() + logvar*0)[active].mean()
        loss.backward()
        metrics = scalar_metrics(result, terms)
        metrics.update(gradient_norms(modules))
        print(json.dumps(dict(dry_run=True, optimizer_steps=0, metrics=metrics), indent=2))
        return metrics
    if cfg["validation"]["dataset"] is None:
        raise ValueError("training requires a saved fixed validation set; set validation.dataset")
    val = load_instances(cfg["validation"]["dataset"])
    if val.size != cfg["customers"]:
        raise ValueError("validation size must match training size")
    from eval.evaluate import load_reference
    reference = load_reference(cfg["validation"]["reference"], val) if cfg["validation"]["reference"] else None
    import hashlib
    reference_hash = hashlib.sha256(reference.tobytes()).hexdigest() if reference is not None else None
    output = Path(cfg["output"]) / f'{cfg["name"]}_m{cfg["customers"]}_seed{cfg["seed"]}'
    output.mkdir(parents=True, exist_ok=True)
    start_epoch, best, cumulative_seconds = 0, float("inf"), 0.
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if ckpt["config"] != cfg:
            raise ValueError("resume requires the identical config; epochs/time budget are stored in config")
        if ckpt.get("validation_fingerprint") != val.metadata["fingerprint"] or ckpt.get("reference_hash") != reference_hash:
            raise ValueError("resume validation/reference contents changed")
        model.load_state_dict(ckpt["model"])
        if critic is not None:
            critic.load_state_dict(ckpt["critic"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        restore_rng(ckpt["rng"], generator)
        start_epoch, best = ckpt["epoch"] + 1, ckpt["best_validation"]
        cumulative_seconds = ckpt.get("training_seconds", 0.)
    elif (output / "last.pt").exists():
        raise FileExistsError(f"existing run {output}; use --resume or a new output directory")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    save_json(output / "metadata.json", dict(parameter_counts=counts, runtime=runtime_metadata(),
                                             validation_fingerprint=val.metadata["fingerprint"], reference_hash=reference_hash,
                                             component_name=model.latent.component_name if model.latent is not None else "classical backbone"))
    writer = None
    if t["tensorboard"]:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "tensorboard"))
    tracker = None
    if t["wandb"]:
        import wandb
        tracker = wandb.init(project="gated-quantum-cvrp", config=cfg, name=output.name)
    training_start, last_metrics = time.perf_counter(), {}
    stop = False
    try:
        for epoch in range(start_epoch, t["epochs"]):
            epoch_start = time.perf_counter()
            modules.train()
            accumulated, completed = {}, 0
            for step in range(t["steps_per_epoch"]):
                elapsed = cumulative_seconds + time.perf_counter()-training_start
                if t["time_budget_seconds"] is not None and elapsed >= t["time_budget_seconds"]:
                    stop = True
                    break
                inst = generate(batch_size, cfg["customers"], generator, capacity=cfg["capacity"]).to(device)
                if t["augmentation"]:
                    # One random symmetry per batch; batch size and compute remain unchanged.
                    idx = torch.randint(8, (), generator=generator).item()
                    inst = augment8(inst).slice(idx*batch_size, (idx+1)*batch_size)
                progress = (epoch + step/t["steps_per_epoch"]) / t["epochs"]
                if t["algorithm"] == "ppo":
                    from train.ppo import ppo_update
                    result, terms, gradients = ppo_update(model, critic, inst, optimizer, scaler, cfg, epoch, progress, amp)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        result = model(inst)
                        loss, terms = reinforce_loss(result, cfg, epoch, progress)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    gradients = gradient_norms(modules)
                    torch.nn.utils.clip_grad_norm_(modules.parameters(), t["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                metrics = scalar_metrics(result, terms)
                metrics.update(gradients)
                if t["trainability_every"] and step % t["trainability_every"] == 0:
                    record = dict(epoch=epoch, step=step, **gradients)
                    path = output / "trainability.csv"
                    present = path.exists()
                    with path.open("a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(record))
                        if not present:
                            w.writeheader()
                        w.writerow(record)
                for key, value in metrics.items():
                    accumulated[key] = accumulated.get(key, 0.) + value
                completed += 1
            if not completed:
                break
            last_metrics = {key: value/completed for key, value in accumulated.items()}
            last_metrics.update(epoch=epoch, steps_completed=completed, epoch_seconds=time.perf_counter()-epoch_start,
                                lr=optimizer.param_groups[0]["lr"])
            from eval.evaluate import evaluate_instances
            validation = evaluate_instances(model, val, batch_size=cfg["validation"]["batch_size"],
                                            augmentation=cfg["validation"]["augmentation"], reference=reference)
            last_metrics["validation_cost"] = validation["summary"]["mean_cost"]
            if reference is not None:
                last_metrics["validation_gap_percent"] = validation["summary"]["mean_gap_percent"]
            improved = last_metrics["validation_cost"] < best
            best = min(best, last_metrics["validation_cost"])
            scheduler.step()
            checkpoint = dict(model=model.state_dict(), critic=critic.state_dict() if critic is not None else None,
                              optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                              config=cfg, epoch=epoch, best_validation=best, rng=rng_state(generator),
                              validation_fingerprint=val.metadata["fingerprint"], reference_hash=reference_hash,
                              training_seconds=cumulative_seconds+time.perf_counter()-training_start)
            torch.save(checkpoint, output / "last.pt")
            if improved:
                torch.save(checkpoint, output / "best.pt")
            csv_path = output / "metrics.csv"
            exists = csv_path.exists()
            with csv_path.open("a", newline="") as f:
                csv_writer = csv.DictWriter(f, fieldnames=list(last_metrics))
                if not exists:
                    csv_writer.writeheader()
                csv_writer.writerow(last_metrics)
            print(json.dumps(last_metrics), flush=True)
            if writer:
                for key, value in last_metrics.items():
                    writer.add_scalar(key, value, epoch)
            if tracker:
                tracker.log(last_metrics, step=epoch)
            if stop:
                break
        if last_metrics:
            collapse_report(last_metrics, cfg["model"], output / "collapse_report.txt")
        save_json(output / "completion.json", dict(completed_epochs=int(last_metrics.get("epoch", start_epoch-1))+1,
                                                  stopped_by_time_budget=stop, best_validation=None if best == float("inf") else best))
    finally:
        if writer:
            writer.close()
        if tracker:
            tracker.finish()
    return output


def main():
    p = parser("Train CVRP policy (only --dry-run avoids optimizer updates)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--customers", type=int)
    p.add_argument("--time-budget", type=float, help="total training seconds, checked between batches")
    p.add_argument("--resume")
    p.add_argument("--dry-run", action="store_true", help="forward/backward validation; zero optimizer steps")
    args = p.parse_args()
    extra = list(args.set)
    for key, value in [("training.epochs", args.epochs), ("seed", args.seed), ("customers", args.customers),
                       ("training.time_budget_seconds", args.time_budget)]:
        if value is not None:
            extra.append(f"{key}={value}")
    cfg = load_config(args.config, extra)
    train(cfg, args.resume, args.dry_run)


if __name__ == "__main__":
    main()
