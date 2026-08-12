from __future__ import annotations

from src.config.schema import ModelConfig
from src.models.advanced_lstm_builder import AdvancedLSTMBuilder
from src.models.base import BaseModelBuilder
from src.models.cluster_lstm_builder import (
    ClusterDualHeadLSTMBuilder,
    ClusterLSTMBuilder,
)
from src.models.dual_head_lstm_builder import DualHeadLSTMBuilder
from src.models.lstm_attention_builder import LSTMAttentionBuilder
from src.models.lstm_builder import LSTMBuilder

_MODEL_REGISTRY: dict[str, type[BaseModelBuilder]] = {
    "lstm": LSTMBuilder,
    "dual_head_lstm": DualHeadLSTMBuilder,
    "advanced_lstm": AdvancedLSTMBuilder,
    "lstm_attention": LSTMAttentionBuilder,
    "cluster_lstm": ClusterLSTMBuilder,
    "cluster_dual_head_lstm": ClusterDualHeadLSTMBuilder,
}


class ModelRegistry:
    @staticmethod
    def build(cfg: ModelConfig) -> BaseModelBuilder:
        builder_cls = _MODEL_REGISTRY.get(cfg.name)
        if builder_cls is None:
            available = list(_MODEL_REGISTRY)
            raise ValueError(
                f"Model '{cfg.name}' not registered. Available: {available}"
            )
        return builder_cls(**cfg.params)

    @staticmethod
    def register(name: str, builder_cls: type[BaseModelBuilder]) -> None:
        _MODEL_REGISTRY[name] = builder_cls

    @staticmethod
    def available() -> list[str]:
        return list(_MODEL_REGISTRY)
