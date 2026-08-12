from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.pipeline.data.cluster_preprocessor import ClusterDataBatch, SEASONS


@dataclass
class ClusterTRTrainResult:
    # models[cluster_id]["global"] → Keras dual-head model (2 inputs) treinado
    # com as 4 estações climáticas agrupadas (sazonalidade vem de
    # month_sin/cos, já presentes nas features de entrada).
    models: dict[int, dict[str, Any]] = field(default_factory=dict)
    histories: dict[int, dict[str, dict]] = field(default_factory=dict)
    # test_losses[cluster_id][season] → loss no conjunto de teste daquela
    # estação, avaliado contra o único modelo "global" do cluster.
    test_losses: dict[int, dict[str, float]] = field(default_factory=dict)


class ClusterTRTrainer:
    """
    Trainer para o ClusterTRLSTMBuilder (arquitetura TRWindBC-style).

    Diferenças vs. ClusterTrainer:
      - Alimenta (x_dynamic, x_static) ao modelo em vez de x_dynamic sozinho.
      - Lê x_static_train/val/test do ClusterDataBatch.
      - Usa ClusterTRLSTMBuilder com dual-branch estático+dinâmico.

    O treino agrupa as 4 estações climáticas num único modelo por cluster
    (sazonalidade capturada via month_sin/cos/day_sin/cos, já presentes nas
    features) — igual ao design de cluster_mlp.py/cluster_lazy.py. Antes,
    cada (cluster, estação) treinava um modelo independente com ~300-700
    amostras; agrupar dá ~4x mais dados por modelo. A avaliação de teste
    continua reportada por estação (ver `fit()`), mesmo com treino pooled.
    """

    def __init__(
        self,
        units: int = 64,
        static_hidden: int = 32,
        dropout: float = 0.4,
        dropout_static: float = 0.05,
        recurrent_dropout: float = 0.0,
        l2_reg: float = 0.01,
        learning_rate: float = 0.001,
        huber_delta: float = 1.5,
        weight_normal: float = 0.7,
        weight_extreme: float = 0.3,
        extreme_weight: float = 20.0,
        extreme_threshold: float = 1.5,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 5,
        min_samples: int = 10,
    ) -> None:
        self.units = units
        self.static_hidden = static_hidden
        self.dropout = dropout
        self.dropout_static = dropout_static
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.huber_delta = huber_delta
        self.weight_normal = weight_normal
        self.weight_extreme = weight_extreme
        self.extreme_weight = extreme_weight
        self.extreme_threshold = extreme_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_samples = min_samples

    # ------------------------------------------------------------------

    @staticmethod
    def _pool_seasons(season_dict: dict[str, np.ndarray]) -> np.ndarray | None:
        """Concatena as 4 estações climáticas num único array, na ordem de
        SEASONS. Retorna None se nenhuma estação tiver dados."""
        parts = [season_dict[s] for s in SEASONS if s in season_dict and len(season_dict[s])]
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def _mask_split_pooled(
        self,
        data: ClusterDataBatch,
        col_idx: int,
    ) -> tuple | None:
        """Agrupa as 4 estações climáticas (treino e val) e mascara por
        cluster (one-hot) — substitui _mask_split (que ficava restrito a
        1 estação por vez)."""
        x_tr_all = self._pool_seasons(data.x_train)
        if x_tr_all is None:
            return None
        x_tr_all = x_tr_all.astype("float32")
        xs_tr_all = self._pool_seasons(data.x_static_train).astype("float32")
        y_tr_all = self._pool_seasons(data.y_train).astype("float32")

        x_vl_all = self._pool_seasons(data.x_val)
        if x_vl_all is not None:
            x_vl_all = x_vl_all.astype("float32")
            xs_vl_all = self._pool_seasons(data.x_static_val).astype("float32")
            y_vl_all = self._pool_seasons(data.y_val).astype("float32")
        else:
            x_vl_all = np.empty((0, *x_tr_all.shape[1:]), dtype="float32")
            xs_vl_all = np.empty((0, xs_tr_all.shape[1]), dtype="float32")
            y_vl_all = np.empty((0, 1), dtype="float32")

        mask_tr = x_tr_all[:, -1, col_idx] == 1
        mask_vl = x_vl_all[:, -1, col_idx] == 1 if len(x_vl_all) else np.array([], dtype=bool)
        return (
            x_tr_all[mask_tr], xs_tr_all[mask_tr], y_tr_all[mask_tr],
            x_vl_all[mask_vl], xs_vl_all[mask_vl], y_vl_all[mask_vl],
        )

    def _eval_test(
        self,
        model,
        data: ClusterDataBatch,
        season: str,
        col_idx: int,
    ) -> float | None:
        """Avalia o loss no conjunto de teste para um cluster/estação."""
        x_te_season = data.x_test.get(season)
        if x_te_season is None:
            return None
        x_te_all = x_te_season.astype("float32")
        y_te_all = data.y_test[season].astype("float32")
        xs_te_all = data.x_static_test[season].astype("float32")
        mask_te = x_te_all[:, -1, col_idx] == 1
        x_te, xs_te = x_te_all[mask_te], xs_te_all[mask_te]
        y_te = y_te_all[mask_te]
        if len(x_te) < self.min_samples:
            return None
        te_eval = model.evaluate(
            [x_te, xs_te],
            {"head_normal": y_te, "head_extreme": y_te},
            verbose=0,
        )
        # TF 2.x retorna lista [total, head1, head2]; versões mais recentes
        # podem retornar float direto quando há só um valor agregado.
        # TF 2.x retorna lista; versões mais recentes podem retornar float.
        return float(te_eval[0] if hasattr(te_eval, "__len__") else te_eval)

    def _fit_pooled(
        self, builder, data, cluster_id, col_idx, n_dynamic, n_static, lookback,
    ):
        """Treina um único modelo por cluster, com as 4 estações climáticas
        agrupadas. Retorna (model, history) ou None."""
        from tensorflow.keras.callbacks import EarlyStopping

        split = self._mask_split_pooled(data, col_idx)
        if split is None:
            return None
        x_tr, xs_tr, y_tr, x_vl, xs_vl, y_vl = split

        if len(x_tr) < self.min_samples:
            print(
                f"  [cluster {cluster_id}] pooled (4 estações): "
                f"dados insuficientes ({len(x_tr)}) — pulando."
            )
            return None

        print(
            f"  [cluster {cluster_id}] pooled (4 estações): {len(x_tr)} amostras... ",
            end="", flush=True,
        )
        model = builder.build(
            n_dynamic=n_dynamic, n_static=n_static, lookback=lookback
        )

        has_val = len(x_vl) >= self.min_samples
        val_data = (
            ([x_vl, xs_vl], {"head_normal": y_vl, "head_extreme": y_vl})
            if has_val else None
        )
        cb = [EarlyStopping(patience=self.patience, restore_best_weights=True,
                            monitor="val_loss" if has_val else "loss")]
        history = model.fit(
            [x_tr, xs_tr],
            {"head_normal": y_tr, "head_extreme": y_tr},
            validation_data=val_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cb,
            verbose=0,
        )
        print("OK")
        return model, history.history

    def fit(self, data: ClusterDataBatch) -> ClusterTRTrainResult:
        from src.models.cluster_tr_lstm_builder import ClusterTRLSTMBuilder

        builder = ClusterTRLSTMBuilder(
            units=self.units,
            static_hidden=self.static_hidden,
            dropout=self.dropout,
            dropout_static=self.dropout_static,
            recurrent_dropout=self.recurrent_dropout,
            l2_reg=self.l2_reg,
            learning_rate=self.learning_rate,
            huber_delta=self.huber_delta,
            weight_normal=self.weight_normal,
            weight_extreme=self.weight_extreme,
            extreme_weight=self.extreme_weight,
            extreme_threshold=self.extreme_threshold,
        )

        result = ClusterTRTrainResult()
        n_dynamic = len(data.feature_names)
        n_static = len(data.static_feature_names)
        lookback = data.x_train[next(iter(data.x_train))].shape[1]

        for cluster_id in data.cluster_ids:
            cluster_col = f"cluster_{cluster_id}"
            if cluster_col not in data.feature_names:
                continue
            col_idx = data.feature_names.index(cluster_col)
            result.models[cluster_id] = {}
            result.histories[cluster_id] = {}
            result.test_losses[cluster_id] = {}

            out = self._fit_pooled(
                builder, data, cluster_id, col_idx, n_dynamic, n_static, lookback,
            )
            if out is None:
                continue
            model, hist = out
            result.models[cluster_id]["global"] = model
            result.histories[cluster_id]["global"] = hist

            # Teste avaliado por estação climática contra o modelo global —
            # mantém o relatório de R²/RMSE por trimestre no dashboard mesmo
            # com o treino agrupado.
            for season in SEASONS:
                test_loss = self._eval_test(model, data, season, col_idx)
                if test_loss is not None:
                    result.test_losses[cluster_id][season] = test_loss

        return result

    # ------------------------------------------------------------------

    def predict(
        self,
        x_dict: dict[str, np.ndarray],
        xs_dict: dict[str, np.ndarray],
        models: dict[int, dict[str, Any]],
        era5_dict: dict[str, np.ndarray],
        scaler_y,
        feature_names: list[str],
        cluster_ids: list[int],
    ) -> dict[str, np.ndarray]:
        """
        Predição em valores absolutos (m/s) segregada por estação climática.

        Lógica de ensemble intra-modelo (idêntica ao ClusterTrainer):
          1. head_normal prediz razão INMET/ERA5 escalonada para todas as amostras.
          2. Amostras onde pred_normal > extreme_threshold usam head_extreme.
          3. Desnormaliza: abs = scaler_y.inverse(ratio) * era5.
        """
        preds: dict[str, np.ndarray] = {}

        for season, x_all in x_dict.items():
            if len(x_all) == 0:
                continue
            x_all = x_all.astype("float32")
            xs_all = xs_dict[season].astype("float32")
            era5_all = np.array(era5_dict[season], dtype="float32")
            n = min(len(x_all), len(era5_all))
            x_all = x_all[:n]
            xs_all = xs_all[:n]
            era5_all = era5_all[:n]

            final_pred_scaled = np.full((n, 1), np.nan, dtype="float32")

            for cluster_id in cluster_ids:
                col = f"cluster_{cluster_id}"
                if col not in feature_names:
                    continue
                col_idx = feature_names.index(col)
                mask = x_all[:, -1, col_idx] == 1
                if not np.any(mask):
                    continue

                model = models.get(cluster_id, {}).get(season)
                if model is None:
                    model = models.get(cluster_id, {}).get("global")
                if model is None:
                    continue

                outputs = model([x_all[mask], xs_all[mask]], training=False)
                pred_normal = outputs[0].numpy()
                pred_extreme = outputs[1].numpy()

                pred_final = pred_normal.copy()
                is_ext = pred_normal.flatten() > self.extreme_threshold
                if np.any(is_ext):
                    pred_final[is_ext] = pred_extreme[is_ext]

                final_pred_scaled[mask] = pred_final

            era5_safe = np.clip(era5_all, 0.1, None)
            abs_pred = (
                scaler_y.inverse_transform(final_pred_scaled).flatten()
                * era5_safe
            )
            preds[season] = abs_pred

        return preds
