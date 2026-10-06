import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)


def pytest_addoption(parser):
    parser.addoption("--run-training", action="store_true", help="opt in to optimization/learning checks")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-training"):
        skip = pytest.mark.skip(reason="optimizer updates disabled; use --run-training")
        for item in items:
            if "training" in item.keywords:
                item.add_marker(skip)
