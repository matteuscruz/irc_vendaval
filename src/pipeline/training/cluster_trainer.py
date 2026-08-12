from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.pipeline.data.cluster_preprocessor import ClusterDataBatch, SEASONS


@dataclass
class ClusterTrainResult:
    # models[cluster_id]["global"] → Keras dual-head model, treinado com as
    # 4 estações climáticas agrupadas (sazonalidade vem de month_sin/cos,
    # já presentes nas features de entrada).
    models: dict[int, dict[str, Any]] = field(default_factory=dict)
    # histories[cluster_id]["global"] → dict com train/val loss por cabeca
    histories: dict[int, dict[str, dict]] = field(default_factory=dict)
    # test_losses[cluster_id][season] → loss no teste daquela estação,
    # avaliado contra o único modelo "global" do cluster.
    test_losses: dict[int, dict[str, float]] = field(default_factory=dict)


class ClusterTrainer:
    """
    Treina um LSTM dual-head por cluster (4 estacoes climaticas agrupadas).

    Cada modelo tem backbone compartilhado e duas cabecas:
      - head_normal:  Huber loss (weight_normal)
      - head_extreme: robust_extreme_loss (weight_extreme)

    Na predicao, a cabeca normal decide; amostras onde
    pred_normal > extreme_threshold sao reencaminhadas a cabeca extrema.

    O treino agrupa as 4 estacoes climaticas num unico modelo por cluster
    (sazonalidade capturada via month_sin/cos/day_sin/cos, ja presentes nas
    features) — antes, cada (cluster, estacao) treinava um modelo
    independente com poucas centenas de amostras; agrupar da ~4x mais dados
    por modelo. A avaliacao de teste continua reportada por estacao.

    Total de modelos: n_clusters (ex: 14).
    """

    def __init__(
        self,
        units: int = 64,
        dropout: float = 0.4,
        l2_reg: float = 0.01,
        learning_rate: float = 0.001,
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
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
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
        """Concatena as 4 estacoes climaticas num unico array, na ordem de
        SEASONS. Retorna None se nenhuma estacao tiver dados."""
        parts = [season_dict[s] for s in SEASONS if s in season_dict and len(season_dict[s])]
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def fit(self, data: ClusterDataBatch) -> ClusterTrainResult:
        from tensorflow.keras.callbacks import EarlyStopping

        from src.models.cluster_lstm_builder import ClusterDualHeadLSTMBuilder

        builder = ClusterDualHeadLSTMBuilder(
            units=self.units,
            dropout=self.dropout,
            l2_reg=self.l2_reg,
            learning_rate=self.learning_rate,
            weight_normal=self.weight_normal,
            weight_extreme=self.weight_extreme,
            extreme_weight=self.extreme_weight,
            extreme_threshold=self.extreme_threshold,
        )

        result = ClusterTrainResult()
        n_features = len(data.feature_names)
        lookback = data.x_train[next(iter(data.x_train))].shape[1]

        for cluster_id in data.cluster_ids:
            cluster_col = f"cluster_{cluster_id}"
            if cluster_col not in data.feature_names:
                continue
            col_idx = data.feature_names.index(cluster_col)

            result.models[cluster_id] = {}
            result.histories[cluster_id] = {}
            result.test_losses[cluster_id] = {}

            X_tr_all = self._pool_seasons(data.x_train)
            if X_tr_all is None:
                continue
            X_tr_all = X_tr_all.astype("float32")
            y_tr_all = self._pool_seasons(data.y_train).astype("float32")

            X_vl_all = self._pool_seasons(data.x_val)
            if X_vl_all is not None:
                X_vl_all = X_vl_all.astype("float32")
                y_vl_all = self._pool_seasons(data.y_val).astype("float32")
            else:
                X_vl_all = np.empty((0, *X_tr_all.shape[1:]), dtype="float32")
                y_vl_all = np.empty((0, 1), dtype="float32")

            mask_tr = X_tr_all[:, -1, col_idx] == 1
            mask_vl = X_vl_all[:, -1, col_idx] == 1 if len(X_vl_all) else np.array([], dtype=bool)

            X_tr = X_tr_all[mask_tr]
            y_tr = y_tr_all[mask_tr]
            X_vl = X_vl_all[mask_vl]
            y_vl = y_vl_all[mask_vl]

            if len(X_tr) < self.min_samples:
                print(
                    f"  [cluster {cluster_id}] pooled (4 estações): "
                    f"dados insuficientes ({len(X_tr)}) — pulando."
                )
                continue

            print(
                f"  [cluster {cluster_id}] pooled (4 estações): "
                f"{len(X_tr)} amostras... ",
                end="",
                flush=True,
            )
            model = builder.build(n_features=n_features, lookback=lookback)

            # Mesmo alvo para ambas as cabecas
            y_dict_tr = {"head_normal": y_tr, "head_extreme": y_tr}

            has_val = len(X_vl) >= self.min_samples
            val_data = (
                (X_vl, {"head_normal": y_vl, "head_extreme": y_vl})
                if has_val
                else None
            )

            cb = [
                EarlyStopping(
                    patience=self.patience,
                    restore_best_weights=True,
                    monitor="val_loss" if has_val else "loss",
                )
            ]
            history = model.fit(
                X_tr,
                y_dict_tr,
                validation_data=val_data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=cb,
                verbose=0,
            )
            print("OK")

            result.models[cluster_id]["global"] = model
            result.histories[cluster_id]["global"] = history.history

            # Teste avaliado por estacao climatica contra o modelo global —
            # mantem o relatorio de R²/RMSE por trimestre mesmo com o treino
            # agrupado.
            for season in SEASONS:
                x_te_season = data.x_test.get(season)
                if x_te_season is None:
                    continue
                x_te_all = x_te_season.astype("float32")
                y_te_all = data.y_test[season].astype("float32")
                mask_te = x_te_all[:, -1, col_idx] == 1
                x_te = x_te_all[mask_te]
                y_te = y_te_all[mask_te]
                if len(x_te) < self.min_samples:
                    continue
                te_eval = model.evaluate(
                    x_te,
                    {"head_normal": y_te, "head_extreme": y_te},
                    verbose=0,
                )
                result.test_losses[cluster_id][season] = (
                    float(te_eval)
                    if isinstance(te_eval, (int, float))
                    else float(te_eval[0])
                )

        return result

    # ------------------------------------------------------------------

    def predict(
        self,
        x_dict: dict[str, np.ndarray],
        models: dict[int, dict[str, Any]],
        era5_dict: dict[str, np.ndarray],
        scaler_y,
        feature_names: list[str],
        cluster_ids: list[int],
    ) -> dict[str, np.ndarray]:
        """
        Predicao em valores absolutos (m/s) segregada por estacao climatica.

        Logica de ensemble intra-modelo:
          1. head_normal prediz a razao INMET/ERA5 escalonada para todas as amostras.
          2. Amostras onde pred_normal > extreme_threshold sao substituidas
             pelo output de head_extreme.
          3. Desnormaliza: abs = scaler_y.inverse(ratio) * era5.
        """
        preds: dict[str, np.ndarray] = {}

        for season, x_all in x_dict.items():
            if len(x_all) == 0:
                continue
            x_all = x_all.astype("float32")
            era5_all = np.array(era5_dict[season], dtype="float32")
            n = min(len(x_all), len(era5_all))
            x_all = x_all[:n]
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

                outputs = model(x_all[mask], training=False)
                pred_normal = outputs[0].numpy()   # (N_cluster, 1)
                pred_extreme = outputs[1].numpy()  # (N_cluster, 1)

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
