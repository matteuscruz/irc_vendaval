"""Testes do condicionamento da cGAN (src/gan/conditioning.py)."""

import torch

from src.gan.conditioning import (
    combine_vectors,
    get_input_dimensions,
    get_one_hot_labels,
)


def test_one_hot_shape_and_values():
    labels = torch.tensor([0, 2, 1])
    oh = get_one_hot_labels(labels, n_classes=3)
    assert oh.shape == (3, 3)
    assert torch.equal(oh.argmax(dim=1), labels)
    assert oh.dtype == torch.float32


def test_combine_vectors_concatenates_on_feature_dim():
    x = torch.randn(4, 16)
    y = torch.randn(4, 6)
    out = combine_vectors(x, y)
    assert out.shape == (4, 22)


def test_get_input_dimensions():
    gen_in, crit_in = get_input_dimensions(z_dim=16, n_classes=6, target_dim=1)
    assert gen_in == 22
    assert crit_in == 7
