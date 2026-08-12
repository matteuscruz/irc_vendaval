from __future__ import annotations

from typing import Any

from src.models.base import BaseModelBuilder


class LSTMAttentionBuilder(BaseModelBuilder):
    """LSTM + Multi-Head Self-Attention para séries temporais climáticas.

    Arquitetura:
        LSTM (return_sequences=True)  → captura dependências temporais
        MultiHeadAttention + residual → foca nos timesteps mais relevantes
        LayerNormalization            → estabiliza gradientes
        GlobalAveragePooling1D        → agrega a sequência atendida
        BatchNormalization
        Dense(dense_units, relu)      → transformação não-linear final
        Dropout
        Dense(1)                      → predição escalar
    """

    def __init__(
        self,
        lstm_units: int = 256,
        num_heads: int = 4,
        key_dim: int = 32,
        dense_units: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        clipnorm: float = 1.0,
    ):
        self.lstm_units = lstm_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm

    def build(self, n_features: int, lookback: int, loss_fn: Any):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=(lookback, n_features))

        # ── LSTM com sequências para alimentar atenção ─────────────────────
        x = tf.keras.layers.LSTM(self.lstm_units, return_sequences=True)(inputs)

        # ── Multi-Head Self-Attention + conexão residual ───────────────────
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout,
        )(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn_out)

        # ── Agrega sequência e projeta ─────────────────────────────────────
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(self.dense_units, activation="relu")(x)
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
