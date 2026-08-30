"""Cobertura mínima do ExGANAugmenter (WGAN-GP condicional, gera X+y juntos).

Foco: o assert que teria pego o bug real do _N_BASE hardcoded (a localização
das colunas one-hot de cluster no tensor de features era assumida fixa em 7,
quebrando silenciosamente com o esquema atual de dezenas de features) — as
colunas de cluster nas amostras sintéticas precisam ser binárias exatas e
corresponder ao cluster pedido, na posição correta de feature_names.

Também cobre checkpoint/resume (src/pipeline/augmentation/checkpoint.py) —
mecanismo adicionado depois de um run real no Modal travar 3h sem gerar
nenhum artefato: sem teste, um bug no resume só apareceria gastando GPU de
novo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pipeline.augmentation.extreme_gan_augmenter import ExGANAugmenter
from src.pipeline.data.cluster_preprocessor import SEASONS, ClusterDataBatch

_N_BASE = 10
_N_CLUSTERS = 3
_LOOKBACK = 4
_CLUSTER_IDS = [1, 2, 3]


def _build_batch(n: int = 60, season: str = "DJF") -> ClusterDataBatch:
    rng = np.random.default_rng(0)
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


def test_exgan_synthetic_cluster_columns_are_exact_onehot():
    data = _build_batch()
    augmenter = ExGANAugmenter(
        extreme_percentile=90.0,
        multiplier=1.0,
        latent_dim=4,
        hidden_units=8,
        epochs=1,
        batch_size=8,
        n_critic=1,
        k_shift=1,
        c_shift=0.3,
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
        # Cada linha é one-hot exato: soma 1, valores em {0,1}.
        assert np.allclose(cluster_cols.sum(axis=1), 1.0)
        assert np.all((cluster_cols == 0.0) | (cluster_cols == 1.0))


def _make_augmenter(checkpoint_dir: str, epochs: int, **overrides) -> ExGANAugmenter:
    kwargs = dict(
        extreme_percentile=90.0, multiplier=1.0, latent_dim=4, hidden_units=8,
        epochs=epochs, batch_size=8, steps_per_epoch=3, n_critic=1, k_shift=0,
        checkpoint_dir=checkpoint_dir, checkpoint_every=1,
    )
    kwargs.update(overrides)
    return ExGANAugmenter(**kwargs)


def test_exgan_checkpoint_resume_continues_from_saved_epoch(tmp_path, capsys):
    """Simula o cenário real que motivou o checkpoint: um run anterior
    salvou progresso até a época 2/4 (ex.: timeout no meio do treino) — uma
    nova instância com o MESMO checkpoint_dir precisa retomar da época 3,
    não recomeçar do zero."""
    data = _build_batch()
    checkpoint_dir = str(tmp_path / "ckpt")

    # Roda só até a época 2, marcando manualmente "done": False depois —
    # equivalente a um crash logo após o checkpoint da época 2 de um treino
    # de 4 épocas.
    _make_augmenter(checkpoint_dir, epochs=2).fit_augment(data)

    state_path = Path(checkpoint_dir) / "state.json"
    state = json.loads(state_path.read_text())
    assert state["done"] is True
    assert state["final_epoch"] == 2
    state["done"] = False  # simula timeout logo depois do checkpoint da época 2
    state_path.write_text(json.dumps(state))

    capsys.readouterr()  # limpa saída da primeira rodada
    _make_augmenter(checkpoint_dir, epochs=4).fit_augment(data)
    out = capsys.readouterr().out
    assert "retomando treino final da época 3/4" in out

    state2 = json.loads(state_path.read_text())
    assert state2["done"] is True
    assert state2["final_epoch"] == 4


def test_exgan_checkpoint_skips_retraining_when_already_done(tmp_path, capsys):
    data = _build_batch()
    checkpoint_dir = str(tmp_path / "ckpt")

    _make_augmenter(checkpoint_dir, epochs=2).fit_augment(data)

    capsys.readouterr()
    _make_augmenter(checkpoint_dir, epochs=2).fit_augment(data)
    out = capsys.readouterr().out
    assert "Checkpoint indica treino já concluído" in out


def test_exgan_checkpoint_resume_shift_round(tmp_path, capsys):
    """Mesma lógica, mas pro checkpoint por rodada de distribution shifting
    (granularidade mais grossa que época)."""
    data = _build_batch()
    checkpoint_dir = str(tmp_path / "ckpt")

    _make_augmenter(checkpoint_dir, epochs=2, k_shift=2).fit_augment(data)
    state_path = Path(checkpoint_dir) / "state.json"
    state = json.loads(state_path.read_text())
    assert state["round_idx"] == 2  # as 2 rodadas de shift completaram

    # Simula um crash logo após a 1a rodada de shift (round_idx=1), antes da
    # 2a rodada e do treino final.
    state["round_idx"] = 1
    state["done"] = False
    state_path.write_text(json.dumps(state))

    capsys.readouterr()
    _make_augmenter(checkpoint_dir, epochs=2, k_shift=2).fit_augment(data)
    out = capsys.readouterr().out
    assert "retomando shifting da rodada 2/2" in out
