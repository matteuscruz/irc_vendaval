from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from src.config.schema import LossConfig, TrainingConfig
from src.models.base import BaseModelBuilder
from src.pipeline.loss.registry import LossRegistry
from src.pipeline.training.base import BaseTrainer, TrainResult


class FinetuneTrainer(BaseTrainer):
    """Two-phase training strategy for extreme-event regression.

    Phase 1 — full network, MSE loss:
        Trains the entire LSTM until validation loss converges.
        The backbone learns general temporal representations.

    Phase 2 — frozen backbone, Pinball loss:
        Freezes every layer except the final Dense head.
        Recompiles with a quantile loss (default Pinball τ=0.90) and a
        much smaller learning rate so the head shifts toward conservative
        (upper-quantile) predictions without destroying the backbone.

    Combined history (Phase 1 + Phase 2 concatenated) is returned so the
    convergence plot shows both phases in a single curve.
    """

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

    # ── Public ────────────────────────────────────────────────────────────

    def fit(self, data) -> TrainResult:
        import tensorflow as tf

        ft_cfg = self.cfg.finetune

        # ── Phase 1: full network with MSE ────────────────────────────────
        self.logger.info("[finetune] Phase 1 — full network, loss=%s", self.loss_fn)
        model = self.model_builder.build(
            n_features=data.n_features,
            lookback=data.X_train.shape[1],
            loss_fn=self.loss_fn,
        )

        hist1 = model.fit(
            data.X_train,
            data.y_train,
            validation_data=(data.X_val, data.y_val),
            epochs=self.cfg.max_epochs,
            batch_size=self.cfg.batch_size,
            callbacks=self._phase1_callbacks(),
            verbose=0,
        )
        self.logger.info(
            "[finetune] Phase 1 done — %d epochs", len(hist1.history["loss"])
        )

        # ── Phase 2: freeze backbone, fine-tune Dense head ────────────────
        self.logger.info(
            "[finetune] Phase 2 — frozen backbone, loss=%s(%s)",
            ft_cfg.loss_name,
            ft_cfg.loss_params,
        )
        self._freeze_backbone(model)

        finetune_loss = LossRegistry.build(
            LossConfig(name=ft_cfg.loss_name, params=ft_cfg.loss_params)
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=ft_cfg.learning_rate,
                clipnorm=1.0,
            ),
            loss=finetune_loss,
        )

        hist2 = model.fit(
            data.X_train,
            data.y_train,
            validation_data=(data.X_val, data.y_val),
            epochs=ft_cfg.max_epochs,
            batch_size=self.cfg.batch_size,
            callbacks=self._phase2_callbacks(ft_cfg.patience),
            verbose=0,
        )
        self.logger.info(
            "[finetune] Phase 2 done — %d epochs", len(hist2.history["loss"])
        )

        # ── Save best checkpoint ──────────────────────────────────────────
        if self.cfg.checkpoint.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            model.save(os.path.join(self.output_dir, "best_model.keras"))

        # ── Combine histories for the convergence plot ────────────────────
        combined = {}
        for key in set(hist1.history) | set(hist2.history):
            combined[key] = (
                hist1.history.get(key, []) + hist2.history.get(key, [])
            )
        # Mark the phase boundary so the notebook can optionally draw a line
        combined["phase2_start"] = [len(hist1.history["loss"])]

        y_pred = model.predict(data.X_test, verbose=0).ravel()
        return TrainResult(history=combined, y_pred_scaled=y_pred)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _freeze_backbone(model) -> None:
        """Freeze every layer except the final Dense output layer."""
        import tensorflow as tf

        for layer in model.layers[:-1]:   # keep last layer (Dense) trainable
            if not isinstance(layer, tf.keras.layers.Dropout):
                layer.trainable = False

        trainable = [l.name for l in model.layers if l.trainable]
        frozen    = [l.name for l in model.layers if not l.trainable]
        logging.getLogger(__name__).info(
            "[finetune] Frozen: %s | Trainable: %s", frozen, trainable
        )

    def _phase1_callbacks(self):
        import tensorflow as tf

        es_cfg = self.cfg.early_stopping
        lr_cfg = self.cfg.reduce_lr
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg.monitor,
                patience=es_cfg.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=lr_cfg.monitor,
                factor=lr_cfg.factor,
                patience=lr_cfg.patience,
                min_lr=lr_cfg.min_lr,
            ),
        ]

    def _phase2_callbacks(self, patience: int):
        import tensorflow as tf

        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            ),
        ]
