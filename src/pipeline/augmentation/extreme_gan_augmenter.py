from __future__ import annotations

import numpy as np

from src.pipeline.augmentation.base import BaseAugmenter
from src.pipeline.data.cluster_preprocessor import SEASONS, ClusterDataBatch


class ExGANAugmenter(BaseAugmenter):
    """
    ExGAN: Conditional WGAN-GP com Distribution Shifting iterativo.

    Segue os 5 passos do paper ExGAN:
      1. E(x) = y (velocidade maxima do vento)
      2. Distribution Shifting: k iteracoes de corte + GAN auxiliar
      3. EVT: ajuste GPD nas excedancias do dataset deslocado
      4. Conditional GAN condicionado em e ~ GPD; L_ext = |e_pedida - e_gerada|
      5. Inferencia: e' = GPD_inverse(tau) -> geracao com probabilidade alvo

    Condition vector (cond_dim = n_clusters + 4 + 1):
      [cluster_onehot(n_cl), season_onehot(4), e_normalized(1)]
      e_normalized = (y - threshold) / gpd_scale  (0 para nao-extremos)
    """

    def __init__(
        self,
        extreme_percentile: float = 90.0,
        multiplier: float = 3.0,
        latent_dim: int = 32,
        hidden_units: int = 256,
        epochs: int = 200,
        batch_size: int = 64,
        steps_per_epoch: int = 150,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        lambda_y: float = 0.5,
        learning_rate_g: float = 1e-4,
        learning_rate_d: float = 1e-4,
        k_shift: int = 3,
        c_shift: float = 0.3,
        tau: float | None = None,
        use_bias_correction: bool = False,
        n_quantiles: int = 100,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 25,
    ) -> None:
        super().__init__(
            extreme_percentile=extreme_percentile,
            multiplier=multiplier,
        )
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lambda_y = lambda_y
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.k_shift = k_shift
        self.c_shift = c_shift
        self.tau = tau
        self.use_bias_correction = use_bias_correction
        self.n_quantiles = n_quantiles
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every

        self._generator = None
        self._bias_corrector: object = None
        self._feature_names_cache: list[str] = []
        self._gpd_shape: float = 0.0
        self._gpd_scale: float = 1.0
        self._threshold: float = 0.0
        self._lookback: int = 7
        self._n_features: int = 13
        self._n_base: int = 7
        self._n_clusters: int = 6
        self._cond_dim: int = 11

    # ------------------------------------------------------------------

    def fit_augment(self, data: ClusterDataBatch) -> ClusterDataBatch:
        import tensorflow as tf
        from src.pipeline.augmentation import checkpoint as ckpt
        from src.pipeline.data.bias_corrector import ERA5BiasCorrector

        # Correção de viés ERA5→INMET antes do pool (fit no treino)
        if self.use_bias_correction:
            self._bias_corrector = ERA5BiasCorrector()
            self._bias_corrector.fit(data, n_quantiles=self.n_quantiles)
            print(
                f"[ExGAN] BiasCorrector QDM ajustado "
                f"(n_quantiles={self.n_quantiles})."
            )

        self._feature_names_cache = list(data.feature_names)
        # Colunas one-hot de cluster ocupam feature_names[n_base:] — ver
        # ClusterPreprocessor (feature_names = avail_features + cluster_cols).
        # NÃO pode ser uma constante fixa: avail_features varia com
        # feature_groups (hoje dezenas de colunas, não as 7 originais).
        self._n_base = len(data.feature_names) - len(data.cluster_ids)

        X_all, y_all, season_labels = self._pool(data)

        # Aplica correção ao conjunto poolado de treino
        if self._bias_corrector is not None:
            X_all = self._bias_corrector.transform(
                X_all, data.feature_names
            )
        self._lookback = X_all.shape[1]
        self._n_features = X_all.shape[2]
        self._n_clusters = len(data.cluster_ids)
        self._cond_dim = self._n_clusters + len(SEASONS) + 1

        n_total = len(X_all)

        # Passo 3 (ANTES do shifting): GPD nos dados ORIGINAIS
        # Ref: ExGAN.py usa genpareto_params pre-fit no dataset real
        self._threshold = float(
            np.percentile(y_all, self.extreme_percentile)
        )
        exceedances_orig = y_all.flatten() - self._threshold
        self._fit_gpd(exceedances_orig[exceedances_orig > 0])
        print(
            f"[ExGAN] {n_total} amostras totais. "
            f"GPD (original): xi={self._gpd_shape:.3f}, "
            f"sigma={self._gpd_scale:.3f}, threshold={self._threshold:.3f}"
        )

        # Checkpoint/resume: se um run anterior já terminou o treino final
        # (marcado "done" no state.json), pula shifting + treino inteiros e
        # só recarrega o gerador salvo — evita retreinar do zero depois de
        # um timeout/crash que aconteceu DEPOIS do treino já ter terminado
        # (ex.: numa etapa posterior do pipeline).
        state = ckpt.load_state(self.checkpoint_dir)
        if state.get("done"):
            print("[ExGAN] Checkpoint indica treino já concluído — pulando shifting e treino final.")
            generator = self._build_generator(tf)
            ckpt.load_weights_if_present(self.checkpoint_dir, "final_generator", generator)
            self._generator = generator
        else:
            print(
                f"[ExGAN] Iniciando distribution shifting "
                f"(k={self.k_shift}, c={self.c_shift})..."
            )

            # Passo 2: Distribution Shifting
            X_shifted, y_shifted, orig_indices = self._distribution_shifting(
                X_all, y_all, tf, self.checkpoint_dir
            )

            # Passo 4: Conditional GAN no dataset deslocado
            cond_shifted = self._build_condition_shifted(
                X_shifted, y_shifted, orig_indices, season_labels, data.cluster_ids
            )
            self._train(X_shifted, y_shifted, cond_shifted, tf, self.checkpoint_dir)

        # Passo 5: Gerar sintéticos via GPD (ou GPD inverse se tau definido)
        n_ext_orig = int(np.sum(y_all.flatten() > self._threshold))
        n_gen = max(1, int(n_ext_orig * self.multiplier))
        x_syn, y_syn = self._generate(n_gen, data.cluster_ids)
        print(f"[ExGAN] {n_gen} amostras sinteticas geradas.")

        return self._distribute_by_season(x_syn, y_syn, season_labels, data)

    # ------------------------------------------------------------------
    # GPD

    def _fit_gpd(self, exceedances: np.ndarray) -> None:
        from scipy.stats import genpareto

        if len(exceedances) < 5:
            self._gpd_shape = 0.0
            self._gpd_scale = (
                float(exceedances.mean()) if len(exceedances) else 1.0
            )
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
        return (self._threshold + excess).astype("float32").reshape(n, 1)

    def _sample_e_for_tau(self, tau: float, n: int) -> np.ndarray:
        """GPD inverse com ajuste de tau pelo shifting (ExGANSampling.py).

        tau' = tau / c^k  — compensa o deslocamento de distribuicao.
        Sem esse ajuste, tau corresponderia ao quantil na distribuicao
        deslocada (muito mais extrema), nao na original.
        """
        # Ajuste pelo shifting: tau' = tau / c^k
        tau_prime = tau / (self.c_shift ** max(1, self.k_shift))
        t = min(tau_prime, 1.0 - 1e-9)
        xi, sigma = self._gpd_shape, self._gpd_scale
        if abs(xi) < 1e-6:
            excess = -sigma * np.log(1.0 - t)
        else:
            excess = sigma / xi * ((1.0 - t) ** (-xi) - 1.0)
        e_val = float(self._threshold + excess)
        return np.full((n, 1), e_val, dtype="float32")

    def _e_to_normalized(self, e: np.ndarray) -> np.ndarray:
        return (
            (e - self._threshold) / max(self._gpd_scale, 1e-6)
        ).astype("float32")

    # ------------------------------------------------------------------
    # Distribution Shifting (Passo 2)

    def _distribution_shifting(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        tf,
        checkpoint_dir: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        k iteracoes: cortar c% menos extremos, treinar GAN auxiliar, repetir.

        Checkpoint por rodada: cada rodada de shifting leva minutos (não
        horas), então o granularidade "rodada completa" é suficiente — não
        checkpointa dentro de uma rodada. Ao retomar, rodadas já completas
        (round_idx no state.json) são puladas, carregando o array já
        deslocado salvo em disco em vez de recomeçar do zero.
        """
        from src.pipeline.augmentation import checkpoint as ckpt

        state = ckpt.load_state(checkpoint_dir)
        start_round = int(state.get("round_idx", 0))
        saved = (
            ckpt.load_arrays_if_present(checkpoint_dir, "shift_progress")
            if start_round > 0 else None
        )
        if saved is not None:
            X_cur, y_cur, orig_indices = saved["X_cur"], saved["y_cur"], saved["orig_indices"]
            print(f"[ExGAN] Checkpoint: retomando shifting da rodada {start_round + 1}/{self.k_shift}.")
        else:
            X_cur = X_all.copy().astype("float32")
            y_cur = y_all.copy().astype("float32")
            orig_indices = np.arange(len(X_all))
            start_round = 0

        aux_epochs = max(1, self.epochs // max(1, self.k_shift + 1))

        for iteration in range(start_round, self.k_shift):
            n = len(X_cur)
            order = np.argsort(y_cur.flatten())[::-1]  # mais extremos primeiro
            n_keep = max(self.batch_size * 2, int(n * (1.0 - self.c_shift)))
            n_gen_aux = n - n_keep
            if n_gen_aux < 1:
                break

            keep_idx = order[:n_keep]
            X_kept = X_cur[keep_idx]
            y_kept = y_cur[keep_idx]
            idx_kept = orig_indices[keep_idx]

            X_syn, y_syn = self._train_and_generate_aux(
                X_kept, y_kept, n_gen_aux, aux_epochs, tf
            )

            # Manter apenas os mais extremos dos sintéticos
            top_order = np.argsort(y_syn.flatten())[::-1][:n_gen_aux]
            X_cur = np.concatenate([X_kept, X_syn[top_order]], axis=0)
            y_cur = np.concatenate([y_kept, y_syn[top_order]], axis=0)
            # Sintéticos recebem índice -1 (sem season_label original)
            orig_indices = np.concatenate(
                [idx_kept, np.full(n_gen_aux, -1, dtype=int)]
            )

            print(
                f"  [ExGAN shift {iteration+1}/{self.k_shift}] "
                f"N={len(X_cur)}, "
                f"y_mean={y_cur.mean():.3f}, "
                f"y_max={y_cur.max():.3f}"
            )

            ckpt.save_arrays(
                checkpoint_dir, "shift_progress",
                X_cur=X_cur, y_cur=y_cur, orig_indices=orig_indices,
            )
            ckpt.save_state(
                checkpoint_dir,
                {"round_idx": iteration + 1, "final_epoch": 0, "done": False},
            )
            # Print DEPOIS de salvar (não só o de progresso acima): o
            # wrapper Modal reage a linhas novas de stdout pra decidir
            # quando commitar o volume — precisa de uma linha depois do
            # arquivo já estar gravado em disco.
            if checkpoint_dir:
                print(f"  [ExGAN] Checkpoint salvo (rodada {iteration + 1}/{self.k_shift}).")

        return X_cur, y_cur, orig_indices

    def _train_and_generate_aux(
        self,
        X_kept: np.ndarray,
        y_kept: np.ndarray,
        n_gen_aux: int,
        aux_epochs: int,
        tf,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAN auxiliar para shifting: sem L_ext (GPD ainda nao fitada)."""
        n = len(X_kept)
        cond = np.zeros((n, self._cond_dim), dtype="float32")
        for j in range(n):
            cluster_row = X_kept[j, -1, self._n_base: self._n_base + self._n_clusters]
            cond[j, : self._n_clusters] = cluster_row
        # season e e_normalized = 0 (sem condicao na fase auxiliar)

        gen_aux = self._build_generator(tf)
        crit_aux = self._build_critic(tf)
        opt_g = tf.keras.optimizers.Adam(self.learning_rate_g, beta_1=0.0)
        opt_d = tf.keras.optimizers.Adam(self.learning_rate_d, beta_1=0.0)

        # Passos compilados com @tf.function — mesmo motivo de
        # tabular_gan_augmenter.py: em modo eager, o loop puro é inviável em
        # datasets reais (centenas de milhares de amostras — travou 3h sem
        # completar 1 época num run real no Modal antes desta correção).
        @tf.function
        def critic_step_aux(x_r, y_r, c_r, z):
            with tf.GradientTape() as tape:
                x_f, y_f = gen_aux([z, c_r], training=True)
                s_real = crit_aux([x_r, y_r, c_r], training=True)
                s_fake = crit_aux([x_f, y_f, c_r], training=True)
                gp = self._gradient_penalty(crit_aux, x_r, y_r, x_f, y_f, c_r, tf)
                loss_c = (
                    tf.reduce_mean(s_fake) - tf.reduce_mean(s_real)
                    + self.lambda_gp * gp
                )
            grads_c = tape.gradient(loss_c, crit_aux.trainable_variables)
            opt_d.apply_gradients(zip(grads_c, crit_aux.trainable_variables))
            return loss_c

        @tf.function
        def generator_step_aux(c_r, z):
            with tf.GradientTape() as tape:
                x_f, y_f = gen_aux([z, c_r], training=True)
                s_fake = crit_aux([x_f, y_f, c_r], training=True)
                loss_g = -tf.reduce_mean(s_fake)
            grads_g = tape.gradient(loss_g, gen_aux.trainable_variables)
            opt_g.apply_gradients(zip(grads_g, gen_aux.trainable_variables))
            return loss_g

        # steps_per_epoch é um hiperparâmetro fixo, não n // batch_size —
        # treino adversarial por minibatch não tem noção de "época = 1
        # passada pelos dados" (idx já é amostrado aleatoriamente a cada
        # step); escalar steps com n faz o custo explodir em datasets reais
        # (~10900 steps/época com 700 mil amostras, inviável mesmo
        # compilado — travou 3h sem terminar 1 época num run real no Modal
        # antes desta correção). Mesma convenção de tabular_gan_augmenter.py.
        steps = self.steps_per_epoch
        for _ in range(aux_epochs):
            for _ in range(steps):
                for _ in range(self.n_critic):
                    idx = np.random.randint(0, n, self.batch_size)
                    critic_step_aux(
                        tf.constant(X_kept[idx]), tf.constant(y_kept[idx]),
                        tf.constant(cond[idx]),
                        tf.random.normal([self.batch_size, self.latent_dim]),
                    )
                idx = np.random.randint(0, n, self.batch_size)
                generator_step_aux(
                    tf.constant(cond[idx]),
                    tf.random.normal([self.batch_size, self.latent_dim]),
                )

        # Gerar com e_normalized=1 para encorajar valores extremos
        cond_gen = np.zeros((n_gen_aux, self._cond_dim), dtype="float32")
        cond_gen[:, -1] = 1.0
        z = tf.random.normal([n_gen_aux, self.latent_dim])
        x_syn, y_syn = gen_aux([z, tf.constant(cond_gen)], training=False)
        return (
            x_syn.numpy().astype("float32"),
            y_syn.numpy().astype("float32").reshape(-1, 1),
        )

    # ------------------------------------------------------------------
    # Condition

    def _build_condition_shifted(
        self,
        X_shifted: np.ndarray,
        y_shifted: np.ndarray,
        orig_indices: np.ndarray,
        season_labels: list[str],
        cluster_ids: list[int],
    ) -> np.ndarray:
        """Condition para o GAN final: cluster + season + e_normalized."""
        n = len(X_shifted)
        n_cl = self._n_clusters
        cond = np.zeros((n, self._cond_dim), dtype="float32")
        season_list = list(SEASONS)

        for j in range(n):
            cluster_row = X_shifted[j, -1, self._n_base: self._n_base + n_cl]
            cond[j, :n_cl] = cluster_row

            orig_i = int(orig_indices[j])
            if 0 <= orig_i < len(season_labels):
                s = season_labels[orig_i]
                if s in SEASONS:
                    cond[j, n_cl + season_list.index(s)] = 1.0
            else:
                # amostra sintética do shifting: season aleatória
                cond[j, n_cl + np.random.randint(0, len(season_list))] = 1.0

            y_j = float(np.asarray(y_shifted[j]).flat[0])
            e_norm = (y_j - self._threshold) / max(self._gpd_scale, 1e-6)
            cond[j, -1] = max(0.0, e_norm)

        return cond

    # ------------------------------------------------------------------
    # Arquitetura

    def _build_generator(self, tf):
        from tensorflow.keras.layers import Concatenate, Dense, Input, Reshape
        from tensorflow.keras.models import Model

        inp_z = Input(shape=(self.latent_dim,), name="z")
        inp_c = Input(shape=(self._cond_dim,), name="cond")
        h = Concatenate()([inp_z, inp_c])
        h = Dense(self.hidden_units, activation="elu")(h)
        h = Dense(self.hidden_units, activation="elu")(h)
        flat_x = self._lookback * self._n_features
        out = Dense(flat_x + 1)(h)
        x_out = Reshape((self._lookback, self._n_features))(out[:, :flat_x])
        y_out = Reshape((1,))(out[:, flat_x:])
        return Model(
            inputs=[inp_z, inp_c], outputs=[x_out, y_out], name="generator"
        )

    def _build_critic(self, tf):
        from tensorflow.keras.layers import Concatenate, Dense, Flatten, Input
        from tensorflow.keras.models import Model

        inp_x = Input(shape=(self._lookback, self._n_features), name="x")
        inp_y = Input(shape=(1,), name="y")
        inp_c = Input(shape=(self._cond_dim,), name="cond")
        h = Concatenate()([Flatten()(inp_x), inp_y, inp_c])
        h = Dense(self.hidden_units, activation="elu")(h)
        h = Dense(self.hidden_units // 2, activation="elu")(h)
        score = Dense(1)(h)
        return Model(
            inputs=[inp_x, inp_y, inp_c], outputs=score, name="critic"
        )

    def _gradient_penalty(
        self, critic, x_real, y_real, x_fake, y_fake, cond, tf
    ):
        B = tf.shape(x_real)[0]
        alpha = tf.random.uniform([B, 1, 1])
        x_hat = alpha * x_real + (1.0 - alpha) * x_fake[:B]
        alpha_y = tf.random.uniform([B, 1])
        y_hat = alpha_y * y_real + (1.0 - alpha_y) * y_fake[:B]
        with tf.GradientTape() as tape:
            tape.watch([x_hat, y_hat])
            score = critic([x_hat, y_hat, cond[:B]], training=True)
        grads = tape.gradient(score, [x_hat, y_hat])
        grads_flat = tf.concat(
            [tf.reshape(g, [B, -1]) for g in grads], axis=1
        )
        norm = tf.sqrt(tf.reduce_sum(grads_flat ** 2, axis=1) + 1e-8)
        return tf.reduce_mean((norm - 1.0) ** 2)

    # ------------------------------------------------------------------
    # Treino final (Passo 4)

    def _train(self, X_data, y_data, cond_data, tf, checkpoint_dir: str | None = None):
        from src.pipeline.augmentation import checkpoint as ckpt

        generator = self._build_generator(tf)
        critic = self._build_critic(tf)
        opt_g = tf.keras.optimizers.Adam(self.learning_rate_g, beta_1=0.0)
        opt_d = tf.keras.optimizers.Adam(self.learning_rate_d, beta_1=0.0)

        # Resume: se já há pesos salvos do treino final (rodada de shifting
        # já concluída numa run anterior, ou o próprio treino final
        # interrompido no meio), carrega e continua da época salva. Perde o
        # estado do otimizador Adam (não checkpointado, custo aceitável —
        # ele reconverge rápido; o generator/critic é o que importa).
        state = ckpt.load_state(checkpoint_dir)
        start_epoch = 0
        if ckpt.load_weights_if_present(checkpoint_dir, "final_generator", generator):
            ckpt.load_weights_if_present(checkpoint_dir, "final_critic", critic)
            start_epoch = int(state.get("final_epoch", 0))
            if start_epoch > 0:
                print(f"[ExGAN] Checkpoint: retomando treino final da época {start_epoch + 1}/{self.epochs}.")

        n = len(X_data)
        steps_per_epoch = self.steps_per_epoch  # ver nota em _train_and_generate_aux

        # Passos compilados com @tf.function (ver mesma correção em
        # _train_and_generate_aux) — a amostragem GPD (self._sample_y_gpd,
        # numpy puro) continua no loop Python de fora, só o forward/backward
        # pass entra no grafo compilado.
        @tf.function
        def critic_step(x_r, y_r, c_r, c_fake, z):
            with tf.GradientTape() as tape:
                x_f, y_f = generator([z, c_fake], training=True)
                s_real = critic([x_r, y_r, c_r], training=True)
                s_fake = critic([x_f, y_f, c_r], training=True)
                gp = self._gradient_penalty(critic, x_r, y_r, x_f, y_f, c_r, tf)
                loss_c = (
                    tf.reduce_mean(s_fake) - tf.reduce_mean(s_real)
                    + self.lambda_gp * gp
                )
            grads_c = tape.gradient(loss_c, critic.trainable_variables)
            opt_d.apply_gradients(zip(grads_c, critic.trainable_variables))
            return loss_c

        @tf.function
        def generator_step(c_g, z, e_target_abs, c_r_sample):
            with tf.GradientTape() as tape:
                x_f, y_f = generator([z, c_g], training=True)
                s_fake = critic([x_f, y_f, c_r_sample], training=True)
                # L_ext relativa: mean(|y_fake - e_alvo| / |e_alvo|)
                # Equivalente ao rpd do paper (ExGAN.py linha ~202)
                loss_ext = tf.reduce_mean(
                    tf.abs(y_f - e_target_abs) / (tf.abs(e_target_abs) + 1e-8)
                )
                loss_g = -tf.reduce_mean(s_fake) + self.lambda_y * loss_ext
            grads_g = tape.gradient(loss_g, generator.trainable_variables)
            opt_g.apply_gradients(zip(grads_g, generator.trainable_variables))
            return loss_g

        for epoch in range(start_epoch, self.epochs):
            c_loss_ep, g_loss_ep = 0.0, 0.0

            for _ in range(steps_per_epoch):
                # --- critic ---
                for _ in range(self.n_critic):
                    idx = np.random.randint(0, n, self.batch_size)
                    x_r = tf.constant(X_data[idx])
                    y_r = tf.constant(y_data[idx])
                    c_r = tf.constant(cond_data[idx])

                    # Condicao fake: e ~ GPD para forca extremidade real
                    c_fake = cond_data[idx].copy()
                    e_gpd = self._sample_y_gpd(self.batch_size)
                    c_fake[:, -1] = self._e_to_normalized(e_gpd).flatten()
                    z = tf.random.normal([self.batch_size, self.latent_dim])

                    loss_c = critic_step(x_r, y_r, c_r, tf.constant(c_fake), z)
                    c_loss_ep += float(loss_c)

                # --- generator ---
                idx = np.random.randint(0, n, self.batch_size)
                c_g = cond_data[idx].copy()
                e_target = self._sample_y_gpd(self.batch_size)
                e_target_norm = self._e_to_normalized(e_target).flatten()
                c_g[:, -1] = e_target_norm
                z = tf.random.normal([self.batch_size, self.latent_dim])
                # e_target em unidades originais (para L_ext relativa)
                e_target_abs = tf.constant(
                    e_target.reshape(-1, 1), dtype=tf.float32
                )

                loss_g = generator_step(
                    tf.constant(c_g), z, e_target_abs, tf.constant(cond_data[idx]),
                )
                g_loss_ep += float(loss_g)

            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                print(
                    f"  [ExGAN] epoch {epoch+1}/{self.epochs} "
                    f"C={c_loss_ep/steps_per_epoch:.3f} "
                    f"G={g_loss_ep/steps_per_epoch:.3f}"
                )

            if checkpoint_dir and (
                (epoch + 1) % self.checkpoint_every == 0 or epoch + 1 == self.epochs
            ):
                ckpt.save_weights(checkpoint_dir, "final_generator", generator)
                ckpt.save_weights(checkpoint_dir, "final_critic", critic)
                ckpt.save_state(
                    checkpoint_dir,
                    {"round_idx": self.k_shift, "final_epoch": epoch + 1, "done": False},
                )
                # Print proposital aqui (não só a cada epochs//5): o wrapper
                # Modal faz commit do volume reagindo a linhas de stdout —
                # sem isso, o checkpoint fica só no disco local do
                # container até a próxima linha de log "por acaso".
                print(f"  [ExGAN] Checkpoint salvo (época {epoch + 1}/{self.epochs}).")

        self._generator = generator
        if checkpoint_dir:
            ckpt.save_state(
                checkpoint_dir,
                {"round_idx": self.k_shift, "final_epoch": self.epochs, "done": True},
            )

    # ------------------------------------------------------------------
    # Geracao (Passo 5)

    def _generate(
        self,
        n_gen: int,
        all_cluster_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        import tensorflow as tf

        cluster_choices = np.random.choice(all_cluster_ids, n_gen)
        season_choices = np.random.choice(list(SEASONS), n_gen)
        season_list = list(SEASONS)

        # e_target: GPD inverse se tau definido, ou sampling ~ GPD
        if self.tau is not None:
            e_targets = self._sample_e_for_tau(self.tau, n_gen).flatten()
        else:
            e_targets = self._sample_y_gpd(n_gen).flatten()

        # Vectorized condition building
        n_cl = len(all_cluster_ids)
        cond = np.zeros((n_gen, self._cond_dim), dtype="float32")

        # cluster one-hot
        cl_idx = np.array(
            [all_cluster_ids.index(c) for c in cluster_choices], dtype=int
        )
        cond[np.arange(n_gen), cl_idx] = 1.0

        # season one-hot
        s_idx = np.array(
            [season_list.index(s) for s in season_choices], dtype=int
        )
        cond[np.arange(n_gen), n_cl + s_idx] = 1.0

        # e_normalized (vectorized, clipped at 0)
        e_norm = (e_targets - self._threshold) / max(self._gpd_scale, 1e-6)
        cond[:, -1] = np.maximum(0.0, e_norm).astype("float32")

        z = tf.random.normal([n_gen, self.latent_dim])
        x_syn, y_syn = self._generator(
            [z, tf.constant(cond)], training=False
        )
        x_syn = x_syn.numpy().astype("float32")
        y_syn = y_syn.numpy().astype("float32").reshape(n_gen, 1)

        # Forcas colunas cluster para binario exato
        cluster_oh = cond[:, : len(all_cluster_ids)]
        x_syn[:, :, self._n_base: self._n_base + len(all_cluster_ids)] = (
            cluster_oh[:, np.newaxis, :]
        )

        # Correção de viés nas features de vento sintéticas
        if self._bias_corrector is not None:
            x_syn = self._bias_corrector.transform(
                x_syn, self._feature_names_cache
            )

        return x_syn, y_syn
