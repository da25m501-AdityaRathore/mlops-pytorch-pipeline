"""Unit tests for model construction and forward-pass shapes."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import SimpleCNN, get_model


@pytest.mark.parametrize("architecture", ["resnet18", "simple_cnn"])
def test_get_model_output_shape(architecture):
    model = get_model(architecture=architecture, num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 10)


def test_get_model_unknown_architecture_raises():
    with pytest.raises(ValueError):
        get_model(architecture="not_a_real_model", num_classes=10)


def test_simple_cnn_param_count_is_reasonable():
    model = SimpleCNN(num_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    # Sanity bound: a compact CNN, not a huge network.
    assert 0 < n_params < 5_000_000


def test_resnet18_num_classes_respected():
    model = get_model(architecture="resnet18", num_classes=7)
    assert model.fc.out_features == 7


def test_forward_backward_pass_runs():
    """Smoke test: a single optimization step should not error."""
    model = get_model(architecture="simple_cnn", num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
