from __future__ import annotations

import logging
from typing import Any

from src.config.schema import TrainingConfig
from src.models.base import BaseModelBuilder
from src.pipeline.training.base import BaseTrainer


def trainer_factory(
    cfg: TrainingConfig,
    model_builder: BaseModelBuilder,
    loss_fn: Any,
    output_dir: str,
    logger: logging.Logger | None = None,
) -> BaseTrainer:
    if cfg.strategy == "standard":
        from src.pipeline.training.standard_trainer import StandardTrainer

        return StandardTrainer(
            model_builder=model_builder,
            loss_fn=loss_fn,
            cfg=cfg,
            output_dir=output_dir,
            logger=logger,
        )

    if cfg.strategy == "bagging":
        from src.pipeline.training.bagging_trainer import BaggingTrainer

        return BaggingTrainer(
            model_builder=model_builder,
            loss_fn=loss_fn,
            cfg=cfg,
            output_dir=output_dir,
            logger=logger,
        )

    if cfg.strategy == "finetune":
        from src.pipeline.training.finetune_trainer import FinetuneTrainer

        return FinetuneTrainer(
            model_builder=model_builder,
            loss_fn=loss_fn,
            cfg=cfg,
            output_dir=output_dir,
            logger=logger,
        )

    raise ValueError(
        f"Unknown training strategy '{cfg.strategy}'. "
        f"Choose 'standard', 'bagging', or 'finetune'."
    )
