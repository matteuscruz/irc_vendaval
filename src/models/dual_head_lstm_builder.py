from __future__ import annotations

from typing import Any

from src.config.schema import LossConfig
from src.models.base import BaseModelBuilder


class DualHeadLSTMBuilder(BaseModelBuilder):
    """LSTM com dois heads de saída treinados simultaneamente.

    head_mean    (índice 0) → MSE   — usado na inferência
    head_extreme (índice 1) → Pinball(tau) — tarefa auxiliar que força o backbone
                               a aprender representações sensíveis a eventos extremos

    Na inferência usa-se head_mean. O benefício do head_extreme é que os gradientes
    do Pinball durante o treino melhoram as representações internas do LSTM.
    """

    def __init__(
        self,
        lstm_units: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        clipnorm: float = 1.0,
        weight_mean: float = 0.7,
        weight_extreme: float = 0.3,
        tau: float = 0.80,
    ):
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.weight_mean = weight_mean
        self.weight_extreme = weight_extreme
        self.tau = tau

    def build(self, n_features: int, lookback: int, loss_fn: Any = None):
        import tensorflow as tf

        from src.pipeline.loss.registry import LossRegistry

        inputs = tf.keras.Input(shape=(lookback, n_features))
        x = tf.keras.layers.LSTM(self.lstm_units)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        head_mean    = tf.keras.layers.Dense(1, name="mean")(x)
        head_extreme = tf.keras.layers.Dense(1, name="extreme")(x)

        model = tf.keras.Model(inputs, [head_mean, head_extreme])

        pinball_fn = LossRegistry.build(
            LossConfig(name="pinball", params={"tau": self.tau})
        )

        # Usar listas (não dicts) evita incompatibilidade entre nomes de tensores
        # e chaves do compile() em diferentes versões do Keras.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.clipnorm,
            ),
            loss=["mse", pinball_fn],
            loss_weights=[self.weight_mean, self.weight_extreme],
        )
        return model
