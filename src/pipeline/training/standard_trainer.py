from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from src.config.schema import TrainingConfig
from src.models.base import BaseModelBuilder
from src.pipeline.training.base import BaseTrainer, TrainResult


class StandardTrainer(BaseTrainer):
    def __init__(
        self,
        model_builder: BaseModelBuilder,
        loss_fn: Any,
        cfg: TrainingConfig,
        output_dir: str,
        logger: logging.Logger | None = None,
    ):
        self.model_builder = model_builder
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)

    def fit(self, data) -> TrainResult:
        import tensorflow as tf

        model = self.model_builder.build(
            n_features=data.n_features,
            lookback=data.X_train.shape[1],
            loss_fn=self.loss_fn,
        )

        callbacks = self._build_callbacks()

        # Multi-output models (e.g. dual-head) need y replicated per output
        n_outputs = len(model.outputs)
        if n_outputs > 1:
            y_tr = [data.y_train] * n_outputs
            y_vl = [data.y_val]   * n_outputs
        else:
            y_tr, y_vl = data.y_train, data.y_val

        # Oversampling: repeat extreme training samples before fitting
        X_train, y_train_arr = data.X_train, data.y_train
        os_cfg = self.cfg.oversampling
        if os_cfg.enabled:
            threshold = np.percentile(y_train_arr, os_cfg.percentile)
            extreme_mask = y_train_arr >= threshold
            X_extreme = np.tile(X_train[extreme_mask], (os_cfg.repeat - 1, 1, 1))
            y_extreme = np.tile(y_train_arr[extreme_mask], os_cfg.repeat - 1)
            X_train = np.concatenate([X_train, X_extreme], axis=0)
            y_train_arr = np.concatenate([y_train_arr, y_extreme], axis=0)
            shuffle_idx = np.random.permutation(len(y_train_arr))
            X_train = X_train[shuffle_idx]
            y_train_arr = y_train_arr[shuffle_idx]
            self.logger.info(
                "Oversampling: repeated P%.0f+ samples %dx → %d total train samples",
                os_cfg.percentile,
                os_cfg.repeat,
                len(y_train_arr),
            )
            if n_outputs > 1:
                y_tr = [y_train_arr] * n_outputs
            else:
                y_tr = y_train_arr

        # Sample weights: upweight extreme training samples
        sw_cfg = self.cfg.sample_weight
        if sw_cfg.enabled:
            threshold = np.percentile(y_train_arr, sw_cfg.percentile)
            sample_weights = np.where(
                y_train_arr >= threshold, sw_cfg.extreme_weight, 1.0
            )
            self.logger.info(
                "Sample weights: %.0f%% of train samples weighted %.1fx  (threshold=%.4f)",
                100 * (y_train_arr >= threshold).mean(),
                sw_cfg.extreme_weight,
                threshold,
            )
        else:
            sample_weights = None

        history = model.fit(
            X_train,
            y_tr,
            sample_weight=sample_weights,
            validation_data=(data.X_val, y_vl),
            epochs=self.cfg.max_epochs,
            batch_size=self.cfg.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        epochs_trained = len(history.history["loss"])
        self.logger.info("Training complete after %d epochs.", epochs_trained)

        raw = model.predict(data.X_test, verbose=0)
        if isinstance(raw, (list, tuple)):
            # Dual-head blending:
            #   head_mean    [0] → MSE,      predicts E[Y|X]      (good R², misses extremes)
            #   head_extreme [1] → Pinball,  predicts Q_tau[Y|X]  (biased high, captures tail)
            # Blending at 60/40 gives a slightly conservative point estimate that
            # preserves overall accuracy while reaching further into the tail.
            mean_pred    = np.array(raw[0]).ravel()
            extreme_pred = np.array(raw[1]).ravel()
            y_pred = 0.6 * mean_pred + 0.4 * extreme_pred
        elif isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[1] > 1:
            y_pred = raw[:, 0]
        else:
            y_pred = raw.ravel()

        return TrainResult(history=history.history, y_pred_scaled=y_pred)

    def _build_callbacks(self):
        import tensorflow as tf

        es_cfg = self.cfg.early_stopping
        lr_cfg = self.cfg.reduce_lr
        ckpt_cfg = self.cfg.checkpoint

        cbs = [
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg.monitor,
                patience=es_cfg.patience,
                restore_best_weights=es_cfg.restore_best_weights,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=lr_cfg.monitor,
                factor=lr_cfg.factor,
                patience=lr_cfg.patience,
                min_lr=lr_cfg.min_lr,
            ),
        ]

        if ckpt_cfg.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, "best_model.keras")
            cbs.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=path,
                    monitor=ckpt_cfg.monitor,
                    save_best_only=True,
                )
            )

        return cbs
