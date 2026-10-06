from __future__ import annotations

import torch

from train.losses import gaussian_nll, reinforce_loss
from train.utils import gradient_norms


def ppo_update(model, critic, inst, optimizer, scaler, cfg, epoch, progress, amp=False):
    """On-policy clipped PPO with undiscounted Monte Carlo terminal returns.

    The old latent noise is replayed in every update so likelihood ratios compare
    policies conditioned on the SAME sampled latent, not different random draws.
    """
    t = cfg["training"]
    device = inst.coords.device
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp):
        old = model(inst)
        old_values, _ = critic(old["h"], old["currents"], old["loads"])
    active = old["active"]
    if not active.any():
        raise ValueError("PPO needs at least one non-forced decision")
    target = old["reward"][..., None].expand_as(old_values).detach()
    advantage = (target-old_values).detach()
    mode = t["advantage"]
    if mode == "batch_std":
        advantage /= advantage[active].std(unbiased=False).clamp_min(t["eps"])
    elif mode == "instance_std":
        weight = active.float()
        count = weight.sum(1, keepdim=True).clamp_min(1)
        mean = (advantage*weight).sum(1, keepdim=True)/count
        variance = ((advantage-mean).square()*weight).sum(1, keepdim=True)/count
        advantage /= variance.sqrt().clamp_min(t["eps"])
    elif mode == "mve":
        sigma = torch.exp(0.5*old["aux"]["logvar_c"]).detach()[:, None, None]
        advantage = (advantage/(sigma+t["eps"])).clamp(-t["advantage_clip"], t["advantage_clip"])
    eps = old["aux"].get("eps")
    eps = eps.detach() if eps is not None else None
    metrics, gradients = {}, {}
    modules = torch.nn.ModuleDict({"policy": model, "critic": critic})
    for _ in range(t["ppo_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            new = model(inst, actions=old["routes"], latent_eps=eps)
            ratio = (new["logprob_steps"] - old["logprob_steps"]).clamp(-20, 20).exp()
            surrogate = torch.minimum(ratio*advantage, ratio.clamp(1-t["ppo_clip"], 1+t["ppo_clip"])*advantage)
            actor = -surrogate[active].mean()
            value, logvar = critic(new["h"], new["currents"], new["loads"])
            value_loss = ((value-target).square() if epoch < t["critic_warmup"] or not t["mve_critic"]
                          else gaussian_nll(value, logvar, target))[active].mean()
            entropy = new["entropy_steps"][active].mean()
            _, auxiliary = reinforce_loss(new, cfg, epoch, progress)
            total = actor + t["value_weight"]*value_loss - t["entropy"]*entropy
            total = total + t["lambda_c"]*auxiliary["loss_difficulty"] + auxiliary["beta"]*auxiliary["loss_kl"]
            total = total + cfg["model"]["lambda_rec"]*auxiliary["loss_recon"]
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        for key, val in gradient_norms(modules).items():
            gradients[key] = gradients.get(key, 0.) + val/t["ppo_epochs"]
        torch.nn.utils.clip_grad_norm_(modules.parameters(), t["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        auxiliary.update(loss_policy=actor, loss_critic=value_loss, entropy=entropy, loss_total=total,
                         ppo_clip_fraction=((ratio-1).abs() > t["ppo_clip"])[active].float().mean())
        for key, value in auxiliary.items():
            metrics[key] = metrics.get(key, 0.) + float(value.detach() if torch.is_tensor(value) else value)/t["ppo_epochs"]
    # Return original on-policy costs (not costs associated with a resampled policy).
    return old, metrics, gradients
