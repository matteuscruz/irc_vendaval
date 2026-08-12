from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.config.schema import TrainingConfig
from src.models.base import BaseModelBuilder
from src.pipeline.training.base import BaseTrainer, TrainResult
from src.pipeline.training.standard_trainer import StandardTrainer


class BaggingTrainer(BaseTrainer):
    """Trains N independent models on bootstrap samples; averages predictions."""

    def __init__(
        self,
        model_builder: BaseModelBuilder,
        loss_fn: Any,
        cfg: TrainingConfig,
        output_dir: str,
        logger: logging.Logger | None = None,
        n_estimators: int = 5,
    ):
        self.model_builder = model_builder
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        self.n_estimators = n_estimators

    def fit(self, data) -> TrainResult:
        import copy

        all_preds: list[np.ndarray] = []
        merged_history: dict[str, list] = {}

        n = len(data.X_train)
        rng = np.random.default_rng()

        for i in range(self.n_estimators):
            self.logger.info("Bagging estimator %d/%d", i + 1, self.n_estimators)

            idx = rng.integers(0, n, size=n)
            boot_data = copy.copy(data)
            boot_data.X_train = data.X_train[idx]
            boot_data.y_train = data.y_train[idx]

            trainer = StandardTrainer(
                model_builder=self.model_builder,
                loss_fn=self.loss_fn,
                cfg=self.cfg,
                output_dir=f"{self.output_dir}/estimator_{i}",
                logger=self.logger,
            )
            result = trainer.fit(boot_data)
            all_preds.append(result.y_pred_scaled)

            for k, v in result.history.items():
                merged_history.setdefault(k, []).extend(v)

        y_pred_avg = np.mean(all_preds, axis=0)
        return TrainResult(history=merged_history, y_pred_scaled=y_pred_avg)
