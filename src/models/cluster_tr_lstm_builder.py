from __future__ import annotations

from typing import Any

from src.models.base import BaseModelBuilder


class ClusterTRLSTMBuilder(BaseModelBuilder):
    """
    LSTM dual-branch com condicionamento estático — adaptação do TRWindBC
    (Ouarda & Houndekindo, Energy 2025) para TF/Keras e dados diários.

    Duas entradas:
        seq_input:    (N, lookback, n_dynamic)  — sequência ERA5
        static_input: (N, n_static)             — lat, lon, percentis por estação

    Fluxo (inspirado no TR-LSTM original):
        1. Static branch: Dense(static_hidden, gelu) → Dropout
        2. Condicionamento: replica static em cada timestep e concatena com ERA5
        3. LSTM sobre [ERA5 || static_tiled]
        4. Merge: concat([lstm_out, static_emb]) → Dense(16, elu) → Dropout
        5. Duas cabeças lineares: head_normal (Huber) e head_extreme (robust_extreme_loss)

    Diferença vs. TRWindBC: eles usam softplus (predizem vento absoluto ≥ 0).
    Aqui predizemos anomalia (pode ser negativa) — ativação linear nas cabeças.
    """

    def __init__(
        self,
        units: int = 64,
        static_hidden: int = 32,
        dropout: float = 0.4,
        dropout_static: float = 0.05,
        recurrent_dropout: float = 0.0,
        l2_reg: float = 0.01,
        learning_rate: float = 0.001,
        huber_delta: float = 1.5,
        weight_normal: float = 0.7,
        weight_extreme: float = 0.3,
        extreme_weight: float = 20.0,
        extreme_threshold: float = 1.5,
    ) -> None:
        self.units = units
        self.static_hidden = static_hidden
        self.dropout = dropout
        self.dropout_static = dropout_static
        self.recurrent_dropout = recurrent_dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.huber_delta = huber_delta
        self.weight_normal = weight_normal
        self.weight_extreme = weight_extreme
        self.extreme_weight = extreme_weight
        self.extreme_threshold = extreme_threshold

    def build(
        self,
        n_dynamic: int,
        n_static: int,
        lookback: int,
        loss_fn: Any = None,  # ignorado — usa losses internas
    ):
        import tensorflow as tf
        from tensorflow.keras.layers import (
            BatchNormalization,
            Concatenate,
            Dense,
            Dropout,
            Input,
            LSTM,
            RepeatVector,
            Reshape,
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2

        from src.config.schema import LossConfig
        from src.pipeline.loss.registry import LossRegistry

        # --- Entradas ---
        seq_input = Input(shape=(lookback, n_dynamic), name="seq_input")
        static_input = Input(shape=(n_static,), name="static_input")

        # --- Static branch ---
        Xs = Dense(self.static_hidden, activation="gelu")(static_input)
        Xs = Dropout(self.dropout_static)(Xs)

        # Replica static em cada timestep e concatena com ERA5
        Xs_tiled = RepeatVector(lookback)(Xs)                        # (N, lookback, static_hidden)
        Xd = Concatenate(axis=-1)([seq_input, Xs_tiled])             # (N, lookback, n_dynamic + static_hidden)

        # --- Dynamic branch ---
        x = LSTM(
            self.units,
            kernel_regularizer=l2(self.l2_reg),
            recurrent_dropout=self.recurrent_dropout,
        )(Xd)
        x = BatchNormalization()(x)

        # --- Merge com static e cabeças de saída ---
        x = Concatenate()([x, Xs])
        x = Dense(16, activation="elu", kernel_regularizer=l2(self.l2_reg))(x)
        x = Dropout(self.dropout)(x)

        head_normal = Dense(1, name="head_normal")(x)
        head_extreme = Dense(1, name="head_extreme")(x)

        model = Model(
            inputs=[seq_input, static_input],
            outputs=[head_normal, head_extreme],
        )

        extreme_loss = LossRegistry.build(
            LossConfig(
                name="robust_extreme_loss",
                params={
                    "extreme_weight": self.extreme_weight,
                    "extreme_threshold": self.extreme_threshold,
                },
            )
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                "head_normal": tf.keras.losses.Huber(delta=self.huber_delta),
                "head_extreme": extreme_loss,
            },
            loss_weights={
                "head_normal": self.weight_normal,
                "head_extreme": self.weight_extreme,
            },
        )
        return model
