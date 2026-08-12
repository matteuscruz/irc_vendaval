from __future__ import annotations

from typing import Any, Literal

from src.models.base import BaseModelBuilder


class ClusterDualHeadLSTMBuilder(BaseModelBuilder):
    """
    LSTM com backbone compartilhado e duas cabecas de saida.

    Cabeca 'head_normal': Huber loss — predicao estavel media.
    Cabeca 'head_extreme': robust_extreme_loss — sensivel a extremos.

    Treinamento multi-task: ambas as cabecas recebem o mesmo alvo (anomalia)
    mas com funcoes de perda diferentes, forcando o backbone a aprender
    representacoes robustas para eventos normais e extremos.

    Na inferencia, a cabeca normal prediz primeiro; amostras onde
    pred_normal > extreme_threshold sao substituidas pela cabeca extrema.
    """

    def __init__(
        self,
        units: int = 64,
        dropout: float = 0.4,
        l2_reg: float = 0.01,
        learning_rate: float = 0.001,
        weight_normal: float = 0.7,
        weight_extreme: float = 0.3,
        extreme_weight: float = 20.0,
        extreme_threshold: float = 1.5,
    ) -> None:
        self.units = units
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.weight_normal = weight_normal
        self.weight_extreme = weight_extreme
        self.extreme_weight = extreme_weight
        self.extreme_threshold = extreme_threshold

    def build(
        self,
        n_features: int,
        lookback: int,
        loss_fn: Any = None,  # ignorado — usa losses internas
    ):
        import tensorflow as tf
        from tensorflow.keras.layers import (
            BatchNormalization,
            Dense,
            Dropout,
            Input,
            LSTM,
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2

        from src.config.schema import LossConfig
        from src.pipeline.loss.registry import LossRegistry

        inp = Input(shape=(lookback, n_features))
        x = LSTM(self.units, kernel_regularizer=l2(self.l2_reg))(inp)
        x = BatchNormalization()(x)
        x = Dense(16, activation="elu")(x)
        x = Dropout(self.dropout)(x)

        head_normal = Dense(1, name="head_normal")(x)
        head_extreme = Dense(1, name="head_extreme")(x)

        model = Model(inputs=inp, outputs=[head_normal, head_extreme])

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
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss={
                "head_normal": tf.keras.losses.Huber(),
                "head_extreme": extreme_loss,
            },
            loss_weights={
                "head_normal": self.weight_normal,
                "head_extreme": self.weight_extreme,
            },
        )
        return model


class ClusterLSTMBuilder(BaseModelBuilder):
    """
    LSTM dual-especialidade para o pipeline de clusters.

    Arquitetura: LSTM(units) → BatchNorm → Dense(16, elu) → Dropout → Dense(1)

    A especialidade 'Extreme' usa o dobro de unidades LSTM para maior capacidade
    de modelar eventos de cauda.
    """

    def __init__(
        self,
        normal_units: int = 32,
        extreme_units: int = 64,
        dropout: float = 0.4,
        l2_reg: float = 0.01,
        learning_rate: float = 0.001,
    ) -> None:
        self.normal_units = normal_units
        self.extreme_units = extreme_units
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate

    def build(
        self,
        n_features: int,
        lookback: int,
        loss_fn: Any,
        specialty: Literal["Normal", "Extreme"] = "Normal",
    ):
        import tensorflow as tf
        from tensorflow.keras.layers import (
            BatchNormalization,
            Dense,
            Dropout,
            Input,
            LSTM,
        )
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.regularizers import l2

        units = self.extreme_units if specialty == "Extreme" else self.normal_units

        model = Sequential(
            [
                Input(shape=(lookback, n_features)),
                LSTM(units, kernel_regularizer=l2(self.l2_reg)),
                BatchNormalization(),
                Dense(16, activation="elu"),
                Dropout(self.dropout),
                Dense(1),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss_fn,
        )
        return model
