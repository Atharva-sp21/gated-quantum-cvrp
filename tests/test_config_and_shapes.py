import pytest
import torch

from data.instances import generate, knn_mask
from models.backbone import Policy
from train.config import load_config
from train.losses import reinforce_loss
from scripts.ibm_hardware_check import expectations_from_counts
from test_backbone import small_config


@pytest.mark.parametrize("norm", ["instance", "batch", "layer"])
@pytest.mark.parametrize("bias", ["none", "scalar", "mlp"])
def test_attention_norm_switches(norm, bias):
    cfg = small_config()
    cfg["model"].update(norm=norm, distance_bias=bias, knn=True)
    model = Policy(cfg["model"])
    result = model(generate(2, 20))
    loss, _ = reinforce_loss(result, cfg)
    loss.backward()
    assert torch.isfinite(loss) and model.layers[0].qkv.weight.grad.norm() > 0


@pytest.mark.parametrize("size", [20, 50, 100])
def test_cross_size_shape(size):
    model = Policy(small_config()["model"]).eval()
    with torch.no_grad():
        result = model(generate(1, size), mode="greedy")
    assert result["cost"].shape == (1, size)


def test_unknown_config_and_knn_and_count_convention():
    with pytest.raises(ValueError, match="unknown"):
        load_config(overrides=["model.typo=true"])
    mask = knn_mask(generate(2, 50).distances)
    assert mask[..., 0].all() and mask[:, 0].all()
    assert mask.diagonal(dim1=-2, dim2=-1).all()
    assert (mask[:, 1:].sum(-1) <= 22).all()
    assert expectations_from_counts({"01": 1}, 2, zz=True).tolist() == [-1., 1., -1., -1.]


def test_all_checked_in_presets_are_valid():
    from pathlib import Path
    paths = list((Path(__file__).resolve().parents[1]/"configs").rglob("*.yaml"))
    assert len(paths) >= 36
    for path in paths:
        cfg = load_config(path)
        assert cfg["name"] and cfg["model"]["d_model"] > 0
