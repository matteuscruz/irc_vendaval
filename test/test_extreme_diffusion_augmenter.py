"""Cobertura mínima do ExtremeDiffusionAugmenter (DDPM+CFG, gera X+y juntos).

Mesmo foco de test_extreme_gan_augmenter.py: garantir que a localização das
colunas one-hot de cluster no tensor de features é calculada dinamicamente
(len(feature_names) - len(cluster_ids)), não assumida fixa — bug real
corrigido nesta sessão (_N_BASE hardcoded em 7, esquema atual tem dezenas de
features).

Também cobre checkpoint/resume (ver test_extreme_gan_augmenter.py pro
racional completo).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.augmentation.extreme_diffusion_augmenter import ExtremeDiffusionAugmenter
from src.pipeline.data.cluster_preprocessor import SEASONS, ClusterDataBatch

_N_BASE = 10
_N_CLUSTERS = 3
_LOOKBACK = 4
_CLUSTER_IDS = [1, 2, 3]


def _build_batch(n: int = 60, season: str = "DJF") -> ClusterDataBatch:
    rng = np.random.default_rng(1)
    feature_names = [f"feat{i}" for i in range(_N_BASE)] + [
        f"cluster_{c}" for c in _CLUSTER_IDS
    ]

    x = rng.normal(size=(n, _LOOKBACK, _N_BASE + _N_CLUSTERS)).astype("float32")
    x[:, :, _N_BASE:] = 0.0
    cluster_assign = rng.integers(0, _N_CLUSTERS, size=n)
    for i, c_idx in enumerate(cluster_assign):
        x[i, :, _N_BASE + c_idx] = 1.0

    y = rng.normal(loc=1.0, scale=0.3, size=(n, 1)).astype("float32")

    data = ClusterDataBatch()
    data.x_train = {s: np.empty((0, _LOOKBACK, _N_BASE + _N_CLUSTERS), dtype="float32") for s in SEASONS}
    data.y_train = {s: np.empty((0, 1), dtype="float32") for s in SEASONS}
    data.x_train[season] = x
    data.y_train[season] = y
    data.feature_names = feature_names
    data.cluster_ids = _CLUSTER_IDS
    return data


def test_diffusion_synthetic_cluster_columns_are_exact_onehot():
    data = _build_batch()
    augmenter = ExtremeDiffusionAugmenter(
        extreme_percentile=90.0,
        multiplier=1.0,
        time_steps=10,
        sampling_steps=3,
        hidden_units=8,
        time_emb_dim=4,
        epochs=1,
        batch_size=8,
    )
    n_before = sum(len(v) for v in data.x_train.values())
    augmented = augmenter.fit_augment(data)
    n_after = sum(len(v) for v in augmented.x_train.values())
    assert n_after > n_before

    n_base = len(data.feature_names) - len(data.cluster_ids)
    for season, x in augmented.x_train.items():
        if len(x) == 0:
            continue
        cluster_cols = x[:, -1, n_base:]
        assert np.all(np.isfinite(cluster_cols))
        assert np.allclose(cluster_cols.sum(axis=1), 1.0)
        assert np.all((cluster_cols == 0.0) | (cluster_cols == 1.0))


def _make_augmenter(checkpoint_dir: str, epochs: int) -> ExtremeDiffusionAugmenter:
    return ExtremeDiffusionAugmenter(
        extreme_percentile=90.0, multiplier=1.0, time_steps=10, sampling_steps=3,
        hidden_units=8, time_emb_dim=4, epochs=epochs, batch_size=8,
        steps_per_epoch=3, checkpoint_dir=checkpoint_dir, checkpoint_every=1,
    )


def test_diffusion_checkpoint_resume_continues_from_saved_epoch(tmp_path, capsys):
    data = _build_batch()
    checkpoint_dir = str(tmp_path / "ckpt")

    _make_augmenter(checkpoint_dir, epochs=2).fit_augment(data)
    state_path = Path(checkpoint_dir) / "state.json"
    state = json.loads(state_path.read_text())
    assert state["done"] is True
    assert state["epoch"] == 2
    state["done"] = False  # simula timeout logo após o checkpoint da época 2
    state_path.write_text(json.dumps(state))

    capsys.readouterr()
    _make_augmenter(checkpoint_dir, epochs=4).fit_augment(data)
    out = capsys.readouterr().out
    assert "retomando da época 3/4" in out

    state2 = json.loads(state_path.read_text())
    assert state2["done"] is True
    assert state2["epoch"] == 4


def test_diffusion_checkpoint_skips_retraining_when_already_done(tmp_path, capsys):
    data = _build_batch()
    checkpoint_dir = str(tmp_path / "ckpt")

    _make_augmenter(checkpoint_dir, epochs=2).fit_augment(data)

    capsys.readouterr()
    _make_augmenter(checkpoint_dir, epochs=2).fit_augment(data)
    out = capsys.readouterr().out
    assert "Checkpoint indica treino já concluído" in out
