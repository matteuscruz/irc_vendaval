from __future__ import annotations

from typing import Any

from src.models.base import BaseModelBuilder


class LSTMBuilder(BaseModelBuilder):
    def __init__(
        self,
        lstm_units: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        clipnorm: float = 1.0,
    ):
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm

    def build(self, n_features: int, lookback: int, loss_fn: Any):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=(lookback, n_features))
        x = tf.keras.layers.LSTM(self.lstm_units)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.clipnorm,
            ),
            loss=loss_fn,
        )
        return model
