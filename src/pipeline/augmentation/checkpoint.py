"""Checkpoint/resume compartilhado por ExGANAugmenter e
ExtremeDiffusionAugmenter.

Treino real (centenas de milhares de amostras) leva dezenas de minutos por
fase — sem isso, qualquer timeout/crash no meio do treino no Modal perde o
trabalho inteiro (foi o que aconteceu num run real: travou 3h sem gerar
nenhum artefato). Todas as funções são no-op quando `checkpoint_dir` é None,
então checkpointing é opt-in e não muda o comportamento default.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_state(checkpoint_dir: str | None) -> dict:
    if not checkpoint_dir:
        return {}
    path = Path(checkpoint_dir) / "state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_state(checkpoint_dir: str | None, state: dict) -> None:
    if not checkpoint_dir:
        return
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "state.json").write_text(json.dumps(state))


def save_weights(checkpoint_dir: str | None, name: str, model) -> None:
    if not checkpoint_dir:
        return
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path / f"{name}.weights.h5"))


def load_weights_if_present(checkpoint_dir: str | None, name: str, model) -> bool:
    if not checkpoint_dir:
        return False
    path = Path(checkpoint_dir) / f"{name}.weights.h5"
    if not path.exists():
        return False
    model.load_weights(str(path))
    return True


def save_arrays(checkpoint_dir: str | None, name: str, **arrays: np.ndarray) -> None:
    if not checkpoint_dir:
        return
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.savez(str(path / f"{name}.npz"), **arrays)


def load_arrays_if_present(checkpoint_dir: str | None, name: str) -> dict | None:
    if not checkpoint_dir:
        return None
    path = Path(checkpoint_dir) / f"{name}.npz"
    if not path.exists():
        return None
    npz = np.load(str(path), allow_pickle=True)
    return {k: npz[k] for k in npz.files}
