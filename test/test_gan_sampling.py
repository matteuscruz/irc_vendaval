"""Testes da amostragem (src/gan/sampling.py)."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.gan.models import Generator
from src.gan.sampling import get_truncated_noise, sample, sample_extremes

N_CLASSES = 3
Z_DIM = 8


def _gen_and_scaler():
    torch.manual_seed(0)
    gen = Generator(Z_DIM, N_CLASSES, hidden_dim=8).eval()
    scaler = StandardScaler().fit(np.random.RandomState(0).normal(10, 2, (500, 1)))
    return gen, scaler


def test_truncated_noise_within_bounds():
    noise = get_truncated_noise(100, Z_DIM, truncation=1.0)
    assert noise.shape == (100, Z_DIM)
    assert noise.abs().max().item() <= 1.0


def test_sample_shape_and_inverse_transform():
    gen, scaler = _gen_and_scaler()
    out = sample(gen, 50, class_idx=1, n_classes=N_CLASSES, z_dim=Z_DIM,
                 scaler=scaler)
    assert out.shape == (50,)
    assert np.isfinite(out).all()
    # Saída em escala física (~média do scaler), não em z-score (~0)
    assert abs(out.mean() - 10.0) < 5.0


def test_sample_extremes_only_above_threshold():
    gen, scaler = _gen_and_scaler()
    # Limiar baixo: deve coletar amostras e todas ≥ threshold
    thr = 0.0
    ext = sample_extremes(gen, 0, N_CLASSES, Z_DIM, scaler,
                          threshold=thr, n_target=100, batch=512)
    assert ext.size > 0
    assert (ext >= thr).all()
    assert ext.size <= 100


def test_sample_extremes_empty_when_threshold_unreachable():
    gen, scaler = _gen_and_scaler()
    ext = sample_extremes(gen, 0, N_CLASSES, Z_DIM, scaler,
                          threshold=1e9, n_target=100, batch=256, max_iter=2)
    assert ext.size == 0
