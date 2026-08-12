"""GAN condicional tabular (alvo-only) para geração de rajadas sintéticas.

Versão simplificada, para dados tabulares (sem dimensão de lookback/sequência),
do ExGANAugmenter (src/pipeline/augmentation/extreme_gan_augmenter.py) — reusa
a mesma matemática de EVT/GPD + WGAN-GP condicional, mas o gerador produz
apenas o alvo escalar (rajada), não um vetor de features completo. As features
associadas a cada alvo sintético são atribuídas depois, por nearest-neighbor,
em src/pipelines/cluster_gan.py — não é responsabilidade desta classe.

Não herda de BaseAugmenter: essa interface (src/pipeline/augmentation/base.py)
é acoplada a ClusterDataBatch e tensores (lookback, features) específicos da
LSTM, o que não se aplica aqui.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_SEASON_LIST = ["DJF", "MAM", "JJA", "SON"]


class TabularExtremeGANAugmenter:
    """WGAN-GP condicional alvo-only + EVT/GPD para dados tabulares.

    Condition vector: [cluster_onehot(n_clusters), season_onehot(4)?, e_normalized(1)]
    e_normalized = (y - threshold) / gpd_scale, clipado em 0 para não-extremos.
    """

    def __init__(
        self,
        extreme_percentile: float = 90.0,
        latent_dim: int = 32,
        hidden_units: int = 128,
        epochs: int = 300,
        batch_size: int = 64,
        steps_per_epoch: int = 150,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        lambda_y: float = 5.0,
        lr_gen: float = 1e-4,
        lr_crit: float = 1e-4,
        include_season: bool = False,
        k_shift: int = 3,
        c_shift: float = 0.3,
    ) -> None:
        self.extreme_percentile = extreme_percentile
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lambda_y = lambda_y
        self.lr_gen = lr_gen
        self.lr_crit = lr_crit
        self.include_season = include_season
        self.k_shift = k_shift
        self.c_shift = c_shift

        self._generator = None
        self._cluster_ids: list = []
        self._cond_dim: int = 0
        self._gpd_shape: float = 0.0
        self._gpd_scale: float = 1.0
        self._threshold: float = 0.0

    # ------------------------------------------------------------------
    # GPD (Generalized Pareto Distribution) — mesma lógica de ExGANAugmenter

    def _fit_gpd(self, exceedances: np.ndarray) -> None:
        from scipy.stats import genpareto

        if len(exceedances) < 5:
            self._gpd_shape = 0.0
            self._gpd_scale = float(exceedances.mean()) if len(exceedances) else 1.0
            return
        shape, _loc, scale = genpareto.fit(exceedances, floc=0)
        self._gpd_shape = float(shape)
        self._gpd_scale = float(scale)

    def _sample_y_gpd(self, n: int) -> np.ndarray:
        u = np.random.uniform(0.0, 1.0, n)
        xi, sigma = self._gpd_shape, self._gpd_scale
        if abs(xi) < 1e-6:
            excess = -sigma * np.log(np.clip(u, 1e-9, 1.0))
        else:
            excess = sigma / xi * ((1.0 - u) ** (-xi) - 1.0)
        return (self._threshold + excess).astype("float32")

    def _e_to_normalized(self, e: np.ndarray) -> np.ndarray:
        return ((e - self._threshold) / max(self._gpd_scale, 1e-6)).astype("float32")

    # ------------------------------------------------------------------
    # Condição

    def _build_condition(
        self, cluster_ids: np.ndarray, season_labels: np.ndarray | None, e_norm: np.ndarray,
    ) -> np.ndarray:
        n = len(cluster_ids)
        n_cl = len(self._cluster_ids)
        cond = np.zeros((n, self._cond_dim), dtype="float32")

        cl_idx = np.array([self._cluster_ids.index(c) for c in cluster_ids], dtype=int)
        cond[np.arange(n), cl_idx] = 1.0

        if self.include_season:
            if season_labels is None:
                raise ValueError("include_season=True requer season_labels em fit().")
            s_idx = np.array([_SEASON_LIST.index(s) for s in season_labels], dtype=int)
            cond[np.arange(n), n_cl + s_idx] = 1.0

        cond[:, -1] = e_norm
        return cond

    # ------------------------------------------------------------------
    # Arquitetura

    def _build_generator(self, tf):
        from tensorflow.keras.layers import Concatenate, Dense, Input
        from tensorflow.keras.models import Model

        inp_z = Input(shape=(self.latent_dim,), name="z")
        inp_c = Input(shape=(self._cond_dim,), name="cond")
        h = Concatenate()([inp_z, inp_c])
        h = Dense(self.hidden_units, activation="elu")(h)
        h = Dense(self.hidden_units, activation="elu")(h)
        y_out = Dense(1)(h)
        return Model(inputs=[inp_z, inp_c], outputs=y_out, name="tabular_generator")

    def _build_critic(self, tf):
        from tensorflow.keras.layers import Concatenate, Dense, Input
        from tensorflow.keras.models import Model

        inp_y = Input(shape=(1,), name="y")
        inp_c = Input(shape=(self._cond_dim,), name="cond")
        h = Concatenate()([inp_y, inp_c])
        h = Dense(self.hidden_units, activation="elu")(h)
        h = Dense(self.hidden_units // 2, activation="elu")(h)
        score = Dense(1)(h)
        return Model(inputs=[inp_y, inp_c], outputs=score, name="tabular_critic")

    def _gradient_penalty(self, critic, y_real, y_fake, cond, tf):
        B = tf.shape(y_real)[0]
        alpha = tf.random.uniform([B, 1])
        y_hat = alpha * y_real + (1.0 - alpha) * y_fake[:B]
        with tf.GradientTape() as tape:
            tape.watch(y_hat)
            score = critic([y_hat, cond[:B]], training=True)
        grad = tape.gradient(score, y_hat)
        norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis=1) + 1e-8)
        return tf.reduce_mean((norm - 1.0) ** 2)

    # ------------------------------------------------------------------
    # Distribution Shifting — k rounds: manter os mais extremos + GAN
    # auxiliar (não condicionado em e_normalized) pra completar o pool com
    # mais extremos. Sem isso, ~extreme_percentile% dos pares reais têm
    # e_normalized=0 (tudo abaixo do threshold cai em cond=0, association
    # fraca com qualquer valor), e o gerador final aprende a ignorar a
    # condição de extremo — colapsa pra um valor baixo quase constante.
    # Mesma lógica de ExGANAugmenter._distribution_shifting, adaptada para
    # alvo escalar (sem dimensão de sequência).

    def _train_and_generate_aux(
        self, y_kept: np.ndarray, cid_kept: np.ndarray, n_gen_aux: int, aux_epochs: int, tf,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(y_kept)
        cond = np.zeros((n, self._cond_dim), dtype="float32")
        n_cl = len(self._cluster_ids)
        cl_idx = np.array([self._cluster_ids.index(c) for c in cid_kept], dtype=int)
        cond[np.arange(n), cl_idx] = 1.0
        # season e e_normalized ficam 0 — sem condicionamento nesta fase auxiliar

        gen_aux = self._build_generator(tf)
        crit_aux = self._build_critic(tf)
        opt_g = tf.keras.optimizers.Adam(self.lr_gen, beta_1=0.0)
        opt_d = tf.keras.optimizers.Adam(self.lr_crit, beta_1=0.0)

        y_col = y_kept.reshape(-1, 1).astype("float32")
        steps = self.steps_per_epoch

        @tf.function
        def critic_step_aux(y_r, c_r, z):
            with tf.GradientTape() as tape:
                y_f = gen_aux([z, c_r], training=True)
                s_real = crit_aux([y_r, c_r], training=True)
                s_fake = crit_aux([y_f, c_r], training=True)
                gp = self._gradient_penalty(crit_aux, y_r, y_f, c_r, tf)
                loss_c = tf.reduce_mean(s_fake) - tf.reduce_mean(s_real) + self.lambda_gp * gp
            grads_c = tape.gradient(loss_c, crit_aux.trainable_variables)
            opt_d.apply_gradients(zip(grads_c, crit_aux.trainable_variables))
            return loss_c

        @tf.function
        def generator_step_aux(c_r, z):
            with tf.GradientTape() as tape:
                y_f = gen_aux([z, c_r], training=True)
                s_fake = crit_aux([y_f, c_r], training=True)
                loss_g = -tf.reduce_mean(s_fake)
            grads_g = tape.gradient(loss_g, gen_aux.trainable_variables)
            opt_g.apply_gradients(zip(grads_g, gen_aux.trainable_variables))
            return loss_g

        for _ in range(aux_epochs):
            for _ in range(steps):
                for _ in range(self.n_critic):
                    idx = np.random.randint(0, n, self.batch_size)
                    critic_step_aux(
                        tf.constant(y_col[idx]), tf.constant(cond[idx]),
                        tf.random.normal([self.batch_size, self.latent_dim]),
                    )
                idx = np.random.randint(0, n, self.batch_size)
                generator_step_aux(
                    tf.constant(cond[idx]), tf.random.normal([self.batch_size, self.latent_dim]),
                )

        # Gerar com e_normalized=1 pra encorajar valores extremos — cluster
        # amostrado proporcionalmente aos mantidos.
        cluster_choices = np.random.choice(cid_kept, n_gen_aux)
        cond_gen = np.zeros((n_gen_aux, self._cond_dim), dtype="float32")
        cl_idx_gen = np.array([self._cluster_ids.index(c) for c in cluster_choices], dtype=int)
        cond_gen[np.arange(n_gen_aux), cl_idx_gen] = 1.0
        cond_gen[:, -1] = 1.0
        z = tf.random.normal([n_gen_aux, self.latent_dim])
        y_syn = gen_aux([z, tf.constant(cond_gen)], training=False).numpy().reshape(-1)
        return y_syn.astype("float32"), cluster_choices

    def _distribution_shifting(
        self, y_train: np.ndarray, cluster_ids: np.ndarray, season_labels: np.ndarray | None, tf,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        y_cur = y_train.copy()
        cid_cur = cluster_ids.copy()
        season_cur = season_labels.copy() if season_labels is not None else None
        aux_epochs = max(1, self.epochs // max(1, self.k_shift + 1))

        for it in range(self.k_shift):
            n = len(y_cur)
            order = np.argsort(y_cur)[::-1]
            n_keep = max(self.batch_size * 2, int(n * (1.0 - self.c_shift)))
            n_gen_aux = n - n_keep
            if n_gen_aux < 1:
                break

            keep_idx = order[:n_keep]
            y_kept, cid_kept = y_cur[keep_idx], cid_cur[keep_idx]
            season_kept = season_cur[keep_idx] if season_cur is not None else None

            y_syn, cid_syn = self._train_and_generate_aux(y_kept, cid_kept, n_gen_aux, aux_epochs, tf)

            top_order = np.argsort(y_syn)[::-1][:n_gen_aux]
            y_cur = np.concatenate([y_kept, y_syn[top_order]])
            cid_cur = np.concatenate([cid_kept, cid_syn[top_order]])
            if season_cur is not None:
                # Sintéticos do shifting não têm season original — sorteia.
                season_syn = np.random.choice(_SEASON_LIST, n_gen_aux)
                season_cur = np.concatenate([season_kept, season_syn])

            print(
                f"  [TabularGAN shift {it+1}/{self.k_shift}] "
                f"N={len(y_cur)}  y_mean={y_cur.mean():.3f}  y_max={y_cur.max():.3f}"
            )

        return y_cur, cid_cur, season_cur

    # ------------------------------------------------------------------
    # Fit

    def fit(
        self,
        y_train: np.ndarray,
        cluster_ids: np.ndarray,
        season_labels: np.ndarray | None = None,
    ) -> "TabularExtremeGANAugmenter":
        import tensorflow as tf

        y_train = np.asarray(y_train, dtype="float32").reshape(-1)
        cluster_ids = np.asarray(cluster_ids)
        if season_labels is not None:
            season_labels = np.asarray(season_labels)

        self._cluster_ids = sorted(set(cluster_ids.tolist()), key=str)
        n_cl = len(self._cluster_ids)
        self._cond_dim = n_cl + (len(_SEASON_LIST) if self.include_season else 0) + 1

        n_total = len(y_train)
        self._threshold = float(np.percentile(y_train, self.extreme_percentile))
        exceedances = y_train - self._threshold
        self._fit_gpd(exceedances[exceedances > 0])
        print(
            f"[TabularGAN] {n_total} amostras. GPD: xi={self._gpd_shape:.3f}, "
            f"sigma={self._gpd_scale:.3f}, threshold={self._threshold:.3f}"
        )

        print(f"[TabularGAN] Distribution shifting (k={self.k_shift}, c={self.c_shift})...")
        y_shifted, cid_shifted, season_shifted = self._distribution_shifting(
            y_train, cluster_ids, season_labels, tf,
        )

        e_norm_real = np.maximum(0.0, self._e_to_normalized(y_shifted))
        cond_real = self._build_condition(cid_shifted, season_shifted, e_norm_real)
        y_real_col = y_shifted.reshape(-1, 1)

        generator = self._build_generator(tf)
        critic = self._build_critic(tf)
        opt_g = tf.keras.optimizers.Adam(self.lr_gen, beta_1=0.0)
        opt_d = tf.keras.optimizers.Adam(self.lr_crit, beta_1=0.0)

        n = len(y_shifted)
        steps_per_epoch = self.steps_per_epoch

        # Passos compilados com @tf.function — evita overhead de Python/eager
        # por chamada (o loop puro em eager mode chegou a levar >50min para
        # 150 épocas em ~10k amostras; compilado, cai para poucos minutos).
        @tf.function
        def critic_step(y_r, c_r, c_fake, z):
            with tf.GradientTape() as tape:
                y_f = generator([z, c_fake], training=True)
                s_real = critic([y_r, c_r], training=True)
                s_fake = critic([y_f, c_r], training=True)
                gp = self._gradient_penalty(critic, y_r, y_f, c_r, tf)
                loss_c = tf.reduce_mean(s_fake) - tf.reduce_mean(s_real) + self.lambda_gp * gp
            grads_c = tape.gradient(loss_c, critic.trainable_variables)
            opt_d.apply_gradients(zip(grads_c, critic.trainable_variables))
            return loss_c

        @tf.function
        def generator_step(c_g, z, e_target_abs, c_r_sample):
            with tf.GradientTape() as tape:
                y_f = generator([z, c_g], training=True)
                s_fake = critic([y_f, c_r_sample], training=True)
                loss_ext = tf.reduce_mean(
                    tf.abs(y_f - e_target_abs) / (tf.abs(e_target_abs) + 1e-8)
                )
                loss_g = -tf.reduce_mean(s_fake) + self.lambda_y * loss_ext
            grads_g = tape.gradient(loss_g, generator.trainable_variables)
            opt_g.apply_gradients(zip(grads_g, generator.trainable_variables))
            return loss_g

        for epoch in range(self.epochs):
            c_loss_ep, g_loss_ep = 0.0, 0.0
            for _ in range(steps_per_epoch):
                for _ in range(self.n_critic):
                    idx = np.random.randint(0, n, self.batch_size)
                    y_r = tf.constant(y_real_col[idx])
                    c_r = tf.constant(cond_real[idx])

                    c_fake = cond_real[idx].copy()
                    e_gpd = self._sample_y_gpd(self.batch_size)
                    c_fake[:, -1] = self._e_to_normalized(e_gpd)
                    z = tf.random.normal([self.batch_size, self.latent_dim])

                    loss_c = critic_step(y_r, c_r, tf.constant(c_fake), z)
                    c_loss_ep += float(loss_c)

                idx = np.random.randint(0, n, self.batch_size)
                c_g = cond_real[idx].copy()
                e_target = self._sample_y_gpd(self.batch_size)
                c_g[:, -1] = self._e_to_normalized(e_target)
                z = tf.random.normal([self.batch_size, self.latent_dim])
                e_target_abs = tf.constant(e_target.reshape(-1, 1), dtype=tf.float32)

                loss_g = generator_step(
                    tf.constant(c_g), z, e_target_abs, tf.constant(cond_real[idx]),
                )
                g_loss_ep += float(loss_g)

            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                print(
                    f"  [TabularGAN] epoch {epoch+1}/{self.epochs} "
                    f"C={c_loss_ep/steps_per_epoch:.3f} G={g_loss_ep/steps_per_epoch:.3f}"
                )

        self._generator = generator
        return self

    # ------------------------------------------------------------------
    # Generate

    def generate(
        self,
        n_per_cluster: dict,
        season_probs: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        import tensorflow as tf

        if self._generator is None:
            raise RuntimeError("Chame fit() antes de generate().")

        rows_cluster: list = []
        rows_season: list = []
        for cid, n in n_per_cluster.items():
            if n <= 0:
                continue
            rows_cluster.extend([cid] * n)
            if self.include_season:
                if season_probs:
                    seasons = list(season_probs.keys())
                    probs = np.array([season_probs[s] for s in seasons], dtype=float)
                    probs = probs / probs.sum()
                else:
                    seasons = _SEASON_LIST
                    probs = None
                rows_season.extend(np.random.choice(seasons, size=n, p=probs).tolist())

        n_gen = len(rows_cluster)
        if n_gen == 0:
            return pd.DataFrame(columns=["cluster_id", "y_synth", "is_extreme"])

        cluster_arr = np.array(rows_cluster)
        season_arr = np.array(rows_season) if self.include_season else None

        e_targets = self._sample_y_gpd(n_gen)
        e_norm = np.maximum(0.0, self._e_to_normalized(e_targets))
        cond = self._build_condition(cluster_arr, season_arr, e_norm)

        z = tf.random.normal([n_gen, self.latent_dim])
        y_syn = self._generator([z, tf.constant(cond)], training=False).numpy().reshape(-1)
        y_syn = np.clip(y_syn, 0, 80)  # faixa física

        out = {"cluster_id": cluster_arr, "y_synth": y_syn.astype("float32"), "is_extreme": True}
        if self.include_season:
            out["season"] = season_arr
        return pd.DataFrame(out)
