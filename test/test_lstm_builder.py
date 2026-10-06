"""ClusterLSTMRegressorBuilder (src/models/cluster_lstm_builder.py): saída
única Dense(1), Huber(delta), sem cabeças normal/extrema."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models.cluster_lstm_builder import ClusterLSTMRegressorBuilder  # noqa: E402


def test_architecture_and_loss():
    model = ClusterLSTMRegressorBuilder(units=8, dropout=0.3, huber_delta=2.5).build(
        n_features=5, lookback=7,
    )
    assert model.input_shape == (None, 7, 5)
    assert model.output_shape == (None, 1)
    assert [layer.name for layer in model.layers] == ["window", "lstm", "dropout", "gust"]
    assert model.get_layer("lstm").units == 8
    assert not model.get_layer("lstm").return_sequences
    assert model.get_layer("dropout").rate == pytest.approx(0.3)
    assert isinstance(model.loss, tf.keras.losses.Huber)
    # delta checado pelo valor: |erro| = 3 > δ = 2.5 → δ·(|e| − δ/2) = 4.375
    loss = float(model.loss(np.zeros((1, 1)), np.full((1, 1), 3.0)))
    assert loss == pytest.approx(4.375)


def test_one_epoch_fits():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 24, 3)).astype("float32")
    y = x[:, -1, :1].copy()
    model = ClusterLSTMRegressorBuilder(units=8).build(n_features=3, lookback=24)
    hist = model.fit(x, y, epochs=1, batch_size=16, verbose=0)
    assert np.isfinite(hist.history["loss"][0])
    assert model.predict(x, verbose=0).shape == (32, 1)
