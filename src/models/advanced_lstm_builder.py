from __future__ import annotations

from typing import Any

from src.config.schema import LossConfig
from src.models.base import BaseModelBuilder


class AdvancedLSTMBuilder(BaseModelBuilder):
    """CNN + Stacked LSTM + Multi-Head Attention + Dual Head.

    Backbone:
        Conv1D  → captures local N-day patterns (ramps, frontal passages)
        LSTM-1  → short-term temporal dynamics  (return_sequences=True)
        LSTM-2  → medium-term dynamics           (return_sequences=True)
        MHA     → learns which past timesteps matter most (with residual)
        GAP     → aggregates attended sequence into a fixed-size vector

    Heads (both receive the same backbone vector):
        head_mean    [index 0] — MSE loss, used for inference
        head_extreme [index 1] — Pinball(tau) auxiliary task; improves backbone
                                  sensitivity to extreme-event patterns
    """

    def __init__(
        self,
        lstm_units_1: int = 256,
        lstm_units_2: int = 128,
        conv_filters: int = 64,
        conv_kernel: int = 3,
        attn_heads: int = 4,
        attn_key_dim: int = 32,
        dropout: float = 0.2,
        learning_rate: float = 1e-4,
        clipnorm: float = 1.0,
        weight_mean: float = 0.7,
        weight_extreme: float = 0.3,
        tau: float = 0.80,
    ):
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.attn_heads = attn_heads
        self.attn_key_dim = attn_key_dim
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

        # ── 1. Local pattern extraction ────────────────────────────────────
        x = tf.keras.layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=self.conv_kernel,
            padding="causal",
            activation="relu",
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)

        # ── 2. Stacked LSTM ────────────────────────────────────────────────
        x = tf.keras.layers.LSTM(self.lstm_units_1, return_sequences=True)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        x = tf.keras.layers.LSTM(self.lstm_units_2, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        # ── 3. Multi-head self-attention with residual ─────────────────────
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=self.attn_heads,
            key_dim=self.attn_key_dim,
            dropout=self.dropout,
        )(x, x)
        x = tf.keras.layers.LayerNormalization()(attn_out + x)

        # ── 4. Aggregate sequence ──────────────────────────────────────────
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # ── 5. Dual heads ──────────────────────────────────────────────────
        h_mean = tf.keras.layers.Dense(128, activation="relu")(x)
        h_mean = tf.keras.layers.Dropout(self.dropout)(h_mean)
        head_mean = tf.keras.layers.Dense(1, name="mean")(h_mean)

        h_extreme = tf.keras.layers.Dense(128, activation="relu")(x)
        h_extreme = tf.keras.layers.Dropout(self.dropout)(h_extreme)
        head_extreme = tf.keras.layers.Dense(1, name="extreme")(h_extreme)

        model = tf.keras.Model(inputs, [head_mean, head_extreme])

        pinball_fn = LossRegistry.build(
            LossConfig(name="pinball", params={"tau": self.tau})
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=self.clipnorm,
            ),
            loss=["mse", pinball_fn],
            loss_weights=[self.weight_mean, self.weight_extreme],
        )
        return model
