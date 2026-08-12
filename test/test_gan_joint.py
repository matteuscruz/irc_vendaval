"""Testes da geração conjunta [features + alvo] (data augmentation)."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.gan.losses import moment_matching_penalty
from src.gan.models import Generator
from src.gan.sampling import sample_joint, sample_joint_extremes
from src.gan.train import TrainConfig, train_cgan

N_CLASSES = 2
Z_DIM = 8
D = 4  # 3 features + 1 alvo
TARGET_COL = 3


def _gen_and_scaler():
    torch.manual_seed(0)
    gen = Generator(Z_DIM, N_CLASSES, hidden_dim=8, out_dim=D).eval()
    rng = np.random.RandomState(0)
    scaler = StandardScaler().fit(rng.normal(5, 2, (500, D)))
    return gen, scaler


def test_moment_matching_multidim():
    a = torch.randn(64, D)
    assert moment_matching_penalty(a, a).item() < 1e-6
    b = torch.randn(64, D) * 5 + 10
    assert moment_matching_penalty(b, a).item() > 1.0


def test_sample_joint_shape():
    gen, scaler = _gen_and_scaler()
    rows = sample_joint(gen, 50, 0, N_CLASSES, Z_DIM, scaler)
    assert rows.shape == (50, D)
    assert np.isfinite(rows).all()


def test_sample_joint_extremes_filters_target_column():
    gen, scaler = _gen_and_scaler()
    thr = 0.0
    rows = sample_joint_extremes(
        gen, 0, N_CLASSES, Z_DIM, scaler, target_col=TARGET_COL,
        threshold=thr, n_target=80, batch=512,
    )
    assert rows.shape[1] == D
    assert (rows[:, TARGET_COL] >= thr).all()


def test_train_cgan_joint_runs():
    rng = np.random.RandomState(0)
    x = rng.normal(0, 1, (300, D)).astype(np.float32)
    labels = rng.randint(0, N_CLASSES, 300)
    cfg = TrainConfig(n_classes=N_CLASSES, data_dim=D, z_dim=Z_DIM,
                      hidden_dim=8, epochs=2, batch_size=64, device="cpu")
    gen, hist = train_cgan(x, labels, cfg, verbose=False)
    out = gen(torch.cat(
        (torch.randn(5, Z_DIM), torch.eye(N_CLASSES)[[0, 1, 0, 1, 0]]), dim=1
    ))
    assert out.shape == (5, D)
