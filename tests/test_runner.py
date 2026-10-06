import copy

import torch

from data.instances import generate, save_instances
from models.backbone import Policy
from train.main import train
from train.utils import seed_all
from test_backbone import small_config


def test_checkpoint_logging_and_resume_without_updates(tmp_path, monkeypatch):
    """Exercise the full runner with Adam.step suppressed: no model training occurs."""
    cfg = small_config("quantum")
    cfg["model"]["mve_latent"] = True
    cfg["training"].update(epochs=1, steps_per_epoch=1, batch_size=2, tensorboard=False, trainability_every=1)
    cfg["validation"]["dataset"] = str(tmp_path / "validation.npz")
    cfg["output"] = str(tmp_path / "runs")
    save_instances(cfg["validation"]["dataset"], generate(2, 20, torch.Generator().manual_seed(123)))
    seed_all(cfg["seed"])
    initial = copy.deepcopy(Policy(cfg["model"]).state_dict())
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, *args, **kwargs: None)
    output = train(cfg)
    checkpoint = torch.load(output/"best.pt", weights_only=False)
    assert all(torch.equal(initial[key], value) for key, value in checkpoint["model"].items())
    assert checkpoint["optimizer"]["state"] == {}
    assert (output/"metrics.csv").exists() and (output/"collapse_report.txt").exists()
    assert (output/"trainability.csv").exists()
    assert train(cfg, resume=output/"last.pt") == output
