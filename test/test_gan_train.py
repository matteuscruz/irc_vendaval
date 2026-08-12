"""Teste de integração do loop de treino (src/gan/train.py) — CPU, curto."""

import numpy as np

from src.gan.models import Generator
from src.gan.train import TrainConfig, train_cgan


def test_train_cgan_runs_and_returns_generator():
    rng = np.random.RandomState(0)
    # 2 clusters com médias distintas (em z-score já normalizado o teste só
    # verifica que o loop roda e produz um gerador funcional).
    x0 = rng.normal(-1, 1, (200, 1))
    x1 = rng.normal(1, 1, (200, 1))
    x = np.vstack([x0, x1]).astype(np.float32)
    labels = np.array([0] * 200 + [1] * 200)

    cfg = TrainConfig(
        n_classes=2, z_dim=4, hidden_dim=8, epochs=2,
        batch_size=64, crit_repeats=2, device="cpu",
    )
    gen, history = train_cgan(x, labels, cfg, verbose=False)

    assert isinstance(gen, Generator)
    assert len(history["gen"]) == 2
    assert len(history["crit"]) == 2
    assert all(np.isfinite(history["gen"]))
    assert all(np.isfinite(history["crit"]))
