from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.pipeline.data.cluster_preprocessor import ClusterDataBatch, SEASONS
from src.pipeline.data.target import inverse_target


@dataclass
class ClusterTrainResult:
    # models[cluster_id]["global"] → Keras LSTM de saída única, treinado com
    # as 4 estações do ano agrupadas (sazonalidade vem de day/month sin/cos).
    models: dict[int, dict[str, Any]] = field(default_factory=dict)
    # histories[cluster_id]["global"] → {"loss": [...], "val_loss": [...]}
    histories: dict[int, dict[str, dict]] = field(default_factory=dict)
    # test_losses[cluster_id][season] → Huber no teste daquela estação do ano
    test_losses: dict[int, dict[str, float]] = field(default_factory=dict)


def cluster_mask(x: np.ndarray, feature_names: list[str], cluster_id) -> np.ndarray | None:
    """Janelas do cluster: one-hot `cluster_{id}` ligado no último passo.
    None se a coluna não existe ou `x` está vazio."""
    col = f"cluster_{cluster_id}"
    if col not in feature_names or x is None or len(x) == 0:
        return None
    return np.asarray(x)[:, -1, feature_names.index(col)] == 1


def cluster_seed(seed: int, cluster_id) -> int:
    """Semente por cluster — cada modelo reproduzível sozinho, sem depender
    da ordem em que os clusters anteriores consumiram o RNG."""
    try:
        return int(seed) + int(cluster_id)
    except (TypeError, ValueError):
        return int(seed) + zlib.crc32(str(cluster_id).encode()) % 100_000


def _pool_seasons(season_dict: dict[str, np.ndarray]) -> np.ndarray | None:
    parts = [season_dict[s] for s in SEASONS if s in season_dict and len(season_dict[s])]
    if not parts:
        return None
    return np.asarray(np.concatenate(parts, axis=0), dtype="float32")


class ClusterTrainer:
    """Treina uma LSTM de saída única por cluster (4 estações do ano
    agrupadas); o teste continua reportado por estação do ano."""

    def __init__(
        self,
        units: int = 96,
        dropout: float = 0.3,
        huber_delta: float = 1.0,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 5,
        min_samples: int = 10,
        seed: int = 42,
        predict_batch_size: int = 4096,
    ) -> None:
        self.units = units
        self.dropout = dropout
        self.huber_delta = huber_delta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_samples = min_samples
        self.seed = seed
        self.predict_batch_size = predict_batch_size

    def fit(self, data: ClusterDataBatch) -> ClusterTrainResult:
        from tensorflow.keras.callbacks import EarlyStopping

        from src.models.cluster_lstm_builder import ClusterLSTMRegressorBuilder
        from src.utils.seeding import set_global_seed

        builder = ClusterLSTMRegressorBuilder(
            units=self.units,
            dropout=self.dropout,
            huber_delta=self.huber_delta,
            learning_rate=self.learning_rate,
        )
        result = ClusterTrainResult()

        X_tr_all = _pool_seasons(data.x_train)
        if X_tr_all is None:
            print("  [training] sem janelas de treino — nada a treinar.")
            return result
        y_tr_all = _pool_seasons(data.y_train)
        X_vl_all = _pool_seasons(data.x_val)
        y_vl_all = _pool_seasons(data.y_val) if X_vl_all is not None else None
        n_steps, n_features = X_tr_all.shape[1], X_tr_all.shape[2]

        for cluster_id in data.cluster_ids:
            mask_tr = cluster_mask(X_tr_all, data.feature_names, cluster_id)
            if mask_tr is None:
                continue
            result.models[cluster_id] = {}
            result.histories[cluster_id] = {}
            result.test_losses[cluster_id] = {}

            X_tr, y_tr = X_tr_all[mask_tr], y_tr_all[mask_tr]
            if len(X_tr) < self.min_samples:
                print(f"  [cluster {cluster_id}] dados insuficientes ({len(X_tr)}) — pulando.")
                continue
            mask_vl = cluster_mask(X_vl_all, data.feature_names, cluster_id)
            has_val = mask_vl is not None and int(mask_vl.sum()) >= self.min_samples
            val_data = (X_vl_all[mask_vl], y_vl_all[mask_vl]) if has_val else None

            print(
                f"  [cluster {cluster_id}] {len(X_tr)} treino / "
                f"{len(val_data[0]) if has_val else 0} val... ",
                end="", flush=True,
            )
            set_global_seed(cluster_seed(self.seed, cluster_id))
            model = builder.build(n_features=n_features, lookback=n_steps)
            history = model.fit(
                X_tr, y_tr,
                validation_data=val_data,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[EarlyStopping(
                    patience=self.patience,
                    restore_best_weights=True,
                    monitor="val_loss" if has_val else "loss",
                )],
                verbose=0,
            )
            print(f"OK ({len(history.history['loss'])} épocas)")
            result.models[cluster_id]["global"] = model
            result.histories[cluster_id]["global"] = history.history

            for season in SEASONS:
                x_te = data.x_test.get(season)
                mask_te = cluster_mask(x_te, data.feature_names, cluster_id)
                if mask_te is None or int(mask_te.sum()) < self.min_samples:
                    continue
                te = model.evaluate(
                    np.asarray(x_te[mask_te], dtype="float32"),
                    np.asarray(data.y_test[season][mask_te], dtype="float32"),
                    batch_size=self.predict_batch_size, verbose=0,
                )
                result.test_losses[cluster_id][season] = float(np.ravel(te)[0])

        return result

    def predict(
        self,
        x_dict: dict[str, np.ndarray],
        models: dict[int, dict[str, Any]],
        scaler_y,
        feature_names: list[str],
        cluster_ids: list[int],
    ) -> dict[str, np.ndarray]:
        """Rajada prevista em m/s (clip [0, 80]) por estação do ano; NaN nas
        janelas de clusters sem modelo."""
        preds: dict[str, np.ndarray] = {}
        for season, x_all in x_dict.items():
            if x_all is None or len(x_all) == 0:
                continue
            x_all = np.asarray(x_all, dtype="float32")
            pred_scaled = np.full(len(x_all), np.nan)
            for cluster_id in cluster_ids:
                mask = cluster_mask(x_all, feature_names, cluster_id)
                if mask is None or not mask.any():
                    continue
                season_models = models.get(cluster_id, {})
                model = season_models.get(season)
                if model is None:
                    model = season_models.get("global")
                if model is None:
                    continue
                pred_scaled[mask] = np.ravel(
                    model.predict(x_all[mask], batch_size=self.predict_batch_size, verbose=0)
                )
            preds[season] = inverse_target(scaler_y, pred_scaled, clip=True)
        return preds
