from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod

import numpy as np

from src.pipeline.data.cluster_preprocessor import (
    SEASONS,
    ClusterDataBatch,
)


class BaseAugmenter(ABC):
    """
    Interface para augmentadores generativos de eventos extremos.

    Fluxo padrao de qualquer subclasse:
      1. Concatenar todas as seasons de x_train/y_train em pool global.
      2. Identificar extremos: y > percentile(y_all, extreme_percentile).
      3. Treinar modelo generativo nos extremos (e/ou todos os dados).
      4. Gerar n_gen = multiplier x n_extreme novas amostras sinteticas.
      5. Corrigir colunas cluster (indices 7:) para valores binarios exatos.
      6. Distribuir sinteticos por season proporcionalmente.
      7. Retornar ClusterDataBatch com x_train/y_train aumentados.
    """

    def __init__(
        self,
        extreme_percentile: float = 90.0,
        multiplier: float = 3.0,
        n_cluster_cols: int = 6,
    ) -> None:
        self.extreme_percentile = extreme_percentile
        self.multiplier = multiplier
        self.n_cluster_cols = n_cluster_cols

    @abstractmethod
    def fit_augment(self, data: ClusterDataBatch) -> ClusterDataBatch:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers compartilhados

    def _pool(
        self, data: ClusterDataBatch
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Concatena todas as seasons de x_train/y_train em arrays globais.

        Retorna (X_all, y_all, season_labels) onde season_labels tem shape (N,)
        com o nome da season de cada amostra — util para redistribuir depois.
        """
        xs, ys, labels = [], [], []
        for season in SEASONS:
            x = data.x_train.get(season)
            y = data.y_train.get(season)
            if x is None or len(x) == 0:
                continue
            n = min(len(x), len(y))
            xs.append(x[:n])
            ys.append(y[:n])
            labels.extend([season] * n)
        return (
            np.concatenate(xs, axis=0),
            np.concatenate(ys, axis=0),
            labels,
        )

    def _build_condition(
        self,
        cluster_ids: list[int],
        all_cluster_ids: list[int],
        season: str,
        quantile_q: float | np.ndarray | None = None,
        y_val: float | np.ndarray | None = None,
        n: int = 1,
    ) -> np.ndarray:
        """Monta vetor de condicao [cluster_onehot, season_onehot, extra].

        extra = quantile_q (ExGAN) OU y_val (Diffusion).
        """
        n_clusters = len(all_cluster_ids)
        cluster_oh = np.zeros((n, n_clusters), dtype="float32")
        for i, cid in enumerate(cluster_ids):
            if cid in all_cluster_ids:
                cluster_oh[:, all_cluster_ids.index(cid)] = 1.0

        season_oh = np.zeros((n, len(SEASONS)), dtype="float32")
        if season in SEASONS:
            season_oh[:, list(SEASONS).index(season)] = 1.0

        parts = [cluster_oh, season_oh]
        if quantile_q is not None:
            q = np.full((n, 1), quantile_q, dtype="float32")
            if isinstance(quantile_q, np.ndarray):
                q = quantile_q.reshape(n, 1).astype("float32")
            parts.append(q)
        elif y_val is not None:
            y = np.full((n, 1), y_val, dtype="float32")
            if isinstance(y_val, np.ndarray):
                y = y_val.reshape(n, 1).astype("float32")
            parts.append(y)
        else:
            parts.append(np.zeros((n, 1), dtype="float32"))

        return np.concatenate(parts, axis=1)  # (n, n_clusters + 4 + 1)

    def _fix_cluster_cols(
        self,
        x_syn: np.ndarray,
        cluster_onehot: np.ndarray,
        n_base_features: int,
    ) -> np.ndarray:
        """Substitui colunas cluster por binarios exatos da condicao."""
        x_out = x_syn.copy()
        n_clusters = cluster_onehot.shape[-1]
        x_out[:, :, n_base_features: n_base_features + n_clusters] = (
            cluster_onehot[:, np.newaxis, :]
        )
        return x_out

    def _distribute_by_season(
        self,
        x_syn: np.ndarray,
        y_syn: np.ndarray,
        season_labels_orig: list[str],
        data: ClusterDataBatch,
    ) -> ClusterDataBatch:
        """Distribui sinteticos por season proporcionalmente e concatena."""
        counts = {
            s: season_labels_orig.count(s) for s in SEASONS
        }
        total = sum(counts.values())
        n_syn = len(x_syn)

        season_indices: dict[str, list[int]] = {s: [] for s in SEASONS}
        for i, s in enumerate(
            np.random.choice(
                [s for s in SEASONS if counts[s] > 0],
                size=n_syn,
                p=[counts[s] / total for s in SEASONS if counts[s] > 0],
            )
        ):
            season_indices[s].append(i)

        new_x = {s: data.x_train[s].copy() for s in data.x_train}
        new_y = {s: data.y_train[s].copy() for s in data.y_train}

        for s, idxs in season_indices.items():
            if not idxs:
                continue
            new_x[s] = np.concatenate(
                [new_x.get(s, np.empty((0,) + x_syn.shape[1:])),
                 x_syn[idxs]], axis=0
            )
            new_y[s] = np.concatenate(
                [new_y.get(s, np.empty((0, 1))),
                 y_syn[idxs]], axis=0
            )

        return dataclasses.replace(data, x_train=new_x, y_train=new_y)
