from __future__ import annotations

from src.models.base import BaseModelBuilder


class ClusterLSTMRegressorBuilder(BaseModelBuilder):
    """
    LSTM de saída única por cluster (substitui a dual-head).

    Arquitetura: Input(T, F) → LSTM(units) → último estado oculto →
    Dropout(dropout) → Dense(1, linear, name="gust").
    Perda Huber(delta) + Adam. O alvo é a rajada em m/s escalonada por
    `scaler_y` — nada de razão × ERA5 nem troca de cabeça na inferência.
    """

    def __init__(
        self,
        units: int = 96,
        dropout: float = 0.3,
        huber_delta: float = 1.0,
        learning_rate: float = 1e-3,
    ) -> None:
        self.units = units
        self.dropout = dropout
        self.huber_delta = huber_delta
        self.learning_rate = learning_rate

    def build(self, n_features: int, lookback: int, loss_fn=None):
        """`lookback` = passos da janela (T). `loss_fn` só existe pela
        interface de BaseModelBuilder — a perda é sempre Huber(huber_delta)."""
        import tensorflow as tf
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.models import Model

        inp = Input(shape=(lookback, n_features), name="window")
        x = LSTM(self.units, name="lstm")(inp)
        x = Dropout(self.dropout, name="dropout")(x)
        out = Dense(1, name="gust")(x)

        model = Model(inputs=inp, outputs=out, name="cluster_lstm")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(delta=self.huber_delta),
        )
        return model
