from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_nll(mu, logvar, target):
    return 0.5 * (logvar + (target-mu).square() * torch.exp(-logvar))


def reinforce_loss(result, cfg, epoch=0, progress=0.):
    t = cfg["training"]
    reward = result["reward"].detach()
    advantage = reward - reward.mean(-1, keepdim=True)
    mode = t["advantage"]
    if mode == "batch_std":
        advantage = advantage / advantage.std(unbiased=False).clamp_min(t["eps"])
    elif mode == "instance_std":
        advantage = advantage / advantage.std(-1, keepdim=True, unbiased=False).clamp_min(t["eps"])
    elif mode == "mve":
        sigma = torch.exp(0.5 * result["aux"]["logvar_c"]).detach()
        advantage = (advantage / (sigma[:, None] + t["eps"])).clamp(-t["advantage_clip"], t["advantage_clip"])
    elif mode != "none":
        raise ValueError(mode)
    policy = -(advantage * result["logprob"]).mean()
    aux = result["aux"]
    target = result["cost"].detach().mean(-1)
    difficulty = (F.mse_loss(aux["mu_c"], target) if epoch < t["difficulty_warmup"] else
                  gaussian_nll(aux["mu_c"], aux["logvar_c"], target).mean())
    zero = policy * 0
    # Free bits are applied separately to each latent dimension, then summed.
    kl = (aux["kl_raw_dimensions"].mean((0, 1)).clamp_min(t["free_bits"]).sum()
          if "kl_raw_dimensions" in aux else zero)
    recon = aux.get("recon", zero)
    beta = t["beta_max"] * min(1., progress/0.2)
    total = policy + t["lambda_c"] * difficulty + beta * kl + cfg["model"]["lambda_rec"] * recon
    return total, dict(loss_policy=policy, loss_difficulty=difficulty, loss_kl=kl, loss_recon=recon,
                       loss_total=total, beta=beta)
