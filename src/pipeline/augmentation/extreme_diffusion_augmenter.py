from __future__ import annotations

import math

import numpy as np

from src.pipeline.augmentation.base import BaseAugmenter
from src.pipeline.data.cluster_preprocessor import SEASONS, ClusterDataBatch

_N_BASE = 7   # features ERA5 (indices 0:7)


class ExtremeDiffusionAugmenter(BaseAugmenter):
    """
    Augmentador baseado em DDPM com Classifier-Free Guidance (CFG).

    Estrategia:
      - Treinar o denoiser em TODOS os dados (nao so extremos) para evitar
        overfitting em conjuntos escassos.
      - Condicionar em [cluster_onehot, season_onehot, y_normalizado].
      - Durante o treino, com probabilidade p_uncond, substitui condicao por
        vetor nulo -> modelo aprende versao incondicional
        (necessario para CFG).
      - Na inferencia (DDIM), usar CFG com y_target extremo e guidance_scale
        alto para empurrar a geracao para a cauda da distribuicao.

    Referencias: DDPM (Ho et al. 2020), DDIM (Song et al. 2020),
                 CFG (Ho & Salimans 2022), DiffESM (Bassetti et al. 2024).
    """

    def __init__(
        self,
        extreme_percentile: float = 90.0,
        multiplier: float = 3.0,
        time_steps: int = 200,
        sampling_steps: int = 50,
        hidden_units: int = 512,
        time_emb_dim: int = 64,
        epochs: int = 100,
        batch_size: int = 64,
        p_uncond: float = 0.1,
        guidance_scale: float = 5.0,
        y_target: float = 2.0,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(
            extreme_percentile=extreme_percentile,
            multiplier=multiplier,
        )
        self.time_steps = time_steps
        self.sampling_steps = sampling_steps
        self.hidden_units = hidden_units
        self.time_emb_dim = time_emb_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        self.y_target = y_target
        self.learning_rate = learning_rate

        self._denoiser = None
        self._alpha_bars: np.ndarray | None = None
        self._lookback: int = 7
        self._n_features: int = 13
        self._cond_dim: int = 11
        self._flat_dim: int = 91

    # ------------------------------------------------------------------

    def fit_augment(self, data: ClusterDataBatch) -> ClusterDataBatch:
        import tensorflow as tf

        X_all, y_all, season_labels = self._pool(data)
        self._lookback = X_all.shape[1]
        self._n_features = X_all.shape[2]
        self._flat_dim = self._lookback * self._n_features
        n_clusters = len(data.cluster_ids)
        self._cond_dim = n_clusters + len(SEASONS) + 1  # +1 y_normalized

        # Noise schedule
        betas = np.linspace(1e-4, 0.02, self.time_steps, dtype="float32")
        alphas = 1.0 - betas
        self._alpha_bars = np.cumprod(alphas).astype("float32")

        # Extremos para contar quantos sinteticos gerar
        threshold = float(np.percentile(y_all, self.extreme_percentile))
        n_ext = int(np.sum(y_all.flatten() > threshold))
        print(
            f"[Diffusion] Treino em {len(X_all)} amostras "
            f"(incluindo nao-extremos). "
            f"{n_ext} extremos (P{self.extreme_percentile:.0f})."
        )

        # Condicao para todas as amostras
        cond_all = self._condition_from_pool(
            X_all, y_all, season_labels, data.cluster_ids
        )

        # Treino DDPM
        self._train(X_all, cond_all, tf)

        # Geracao via DDIM + CFG
        n_gen = max(1, int(n_ext * self.multiplier))
        x_syn, y_syn = self._generate(n_gen, data.cluster_ids, tf)

        print(f"[Diffusion] {n_gen} amostras sinteticas geradas.")
        return self._distribute_by_season(
            x_syn, y_syn, season_labels, data
        )

    # ------------------------------------------------------------------

    def _sinusoidal_embed(self, t, tf):
        """Embedding sinusoidal de timestep. t: (B,) int."""
        half = self.time_emb_dim // 2
        freqs = tf.exp(
            -tf.range(half, dtype=tf.float32)
            * (math.log(10000.0) / max(half - 1, 1))
        )
        t_f = tf.cast(t, tf.float32)[:, tf.newaxis]
        args = t_f * freqs[tf.newaxis, :]
        return tf.concat([tf.sin(args), tf.cos(args)], axis=-1)  # (B, dim)

    def _build_denoiser(self, tf):
        from tensorflow.keras.layers import (
            Concatenate, Dense, Input
        )
        from tensorflow.keras.models import Model

        inp_xt = Input(shape=(self._flat_dim,), name="x_noisy")
        inp_t = Input(shape=(self.time_emb_dim,), name="t_emb")
        inp_c = Input(shape=(self._cond_dim,), name="cond")
        h = Concatenate()([inp_xt, inp_t, inp_c])
        h = Dense(self.hidden_units, activation="swish")(h)
        h = Dense(self.hidden_units, activation="swish")(h)
        eps_out = Dense(self._flat_dim, name="eps")(h)
        return Model(
            inputs=[inp_xt, inp_t, inp_c], outputs=eps_out,
            name="denoiser"
        )

    def _train(self, X_all: np.ndarray, cond_all: np.ndarray, tf) -> None:
        denoiser = self._build_denoiser(tf)
        opt = tf.keras.optimizers.Adam(self.learning_rate)
        alpha_bars_tf = tf.constant(self._alpha_bars)

        n = len(X_all)
        X_flat = X_all.reshape(n, self._flat_dim).astype("float32")
        steps = max(1, n // self.batch_size)

        for epoch in range(self.epochs):
            loss_ep = 0.0
            for _ in range(steps):
                idx = np.random.randint(0, n, self.batch_size)
                x0 = tf.constant(X_flat[idx])
                c = tf.constant(cond_all[idx])

                t = tf.random.uniform(
                    [self.batch_size], 1, self.time_steps, dtype=tf.int32
                )
                t_emb = self._sinusoidal_embed(t, tf)

                # CFG: dropout de condicao com p_uncond
                drop_mask = tf.cast(
                    tf.random.uniform([self.batch_size, 1]) > self.p_uncond,
                    tf.float32,
                )
                c_input = c * drop_mask

                ab = tf.gather(alpha_bars_tf, t)[:, tf.newaxis]
                eps = tf.random.normal(tf.shape(x0))
                x_t = tf.sqrt(ab) * x0 + tf.sqrt(1.0 - ab) * eps

                with tf.GradientTape() as tape:
                    eps_pred = denoiser(
                        [x_t, t_emb, c_input], training=True
                    )
                    loss = tf.reduce_mean((eps_pred - eps) ** 2)

                grads = tape.gradient(loss, denoiser.trainable_variables)
                opt.apply_gradients(
                    zip(grads, denoiser.trainable_variables)
                )
                loss_ep += float(loss)

            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                print(
                    f"  [Diffusion] epoch {epoch+1}/{self.epochs} "
                    f"loss={loss_ep/steps:.4f}"
                )

        self._denoiser = denoiser

    def _generate(
        self,
        n_gen: int,
        all_cluster_ids: list[int],
        tf,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha_bars = self._alpha_bars

        # Subconjunto de passos DDIM uniformemente espacados
        ddim_steps = np.linspace(
            self.time_steps - 1, 0, self.sampling_steps, dtype=int
        )

        # Condicao extrema: cluster/season aleatorios, y = y_target
        cluster_choices = np.random.choice(all_cluster_ids, n_gen)
        season_choices = np.random.choice(list(SEASONS), n_gen)
        n_cl = len(all_cluster_ids)

        c_extreme = np.zeros((n_gen, self._cond_dim), dtype="float32")
        for i in range(n_gen):
            cid = cluster_choices[i]
            if cid in all_cluster_ids:
                c_extreme[i, all_cluster_ids.index(cid)] = 1.0
            s = season_choices[i]
            if s in SEASONS:
                c_extreme[i, n_cl + list(SEASONS).index(s)] = 1.0
            c_extreme[i, -1] = float(self.y_target)

        c_null = np.zeros_like(c_extreme)

        x = np.random.normal(0.0, 1.0, (n_gen, self._flat_dim)).astype(
            "float32"
        )

        for step_idx, t_val in enumerate(ddim_steps):
            t_arr = np.full((n_gen,), int(t_val), dtype=np.int32)
            t_emb = self._sinusoidal_embed(
                tf.constant(t_arr), tf
            ).numpy()

            eps_cond = self._denoiser(
                [tf.constant(x),
                 tf.constant(t_emb),
                 tf.constant(c_extreme)],
                training=False,
            ).numpy()

            eps_uncond = self._denoiser(
                [tf.constant(x),
                 tf.constant(t_emb),
                 tf.constant(c_null)],
                training=False,
            ).numpy()

            # CFG: amplifica direcao condicionada
            eps_guided = (
                eps_uncond
                + self.guidance_scale * (eps_cond - eps_uncond)
            )

            ab_t = alpha_bars[t_val]
            ab_prev = (
                alpha_bars[ddim_steps[step_idx + 1]]
                if step_idx + 1 < len(ddim_steps)
                else 1.0
            )

            # DDIM deterministic update
            x0_pred = (x - np.sqrt(1.0 - ab_t) * eps_guided) / np.sqrt(
                ab_t
            )
            x0_pred = np.clip(x0_pred, -5.0, 5.0)
            x = np.sqrt(ab_prev) * x0_pred + np.sqrt(
                1.0 - ab_prev
            ) * eps_guided

        x_syn = x.reshape(n_gen, self._lookback, self._n_features).astype(
            "float32"
        )

        # y sintetico: estimar da condicao (y_target escalonado)
        y_syn = np.full(
            (n_gen, 1), self.y_target, dtype="float32"
        ) + np.random.normal(0, 0.1, (n_gen, 1)).astype("float32")

        # Forcar colunas cluster para binario exato
        cluster_oh = c_extreme[:, :n_cl]
        x_syn[:, :, _N_BASE: _N_BASE + n_cl] = cluster_oh[:, np.newaxis, :]

        return x_syn, y_syn

    def _condition_from_pool(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        season_labels: list[str],
        all_cluster_ids: list[int],
    ) -> np.ndarray:
        """Monta condicao [cluster_oh, season_oh, y] para todo o pool."""
        n = len(X_all)
        n_cl = len(all_cluster_ids)
        cond = np.zeros((n, self._cond_dim), dtype="float32")
        y_flat = y_all.flatten()

        for i in range(n):
            # cluster from X one-hot cols
            cluster_row = X_all[i, -1, _N_BASE: _N_BASE + n_cl]
            cond[i, :n_cl] = cluster_row

            # season
            if i < len(season_labels):
                s = season_labels[i]
                if s in SEASONS:
                    cond[i, n_cl + list(SEASONS).index(s)] = 1.0

            # y normalizado (ja esta em espaco RobustScaler)
            cond[i, -1] = float(y_flat[i])

        return cond
