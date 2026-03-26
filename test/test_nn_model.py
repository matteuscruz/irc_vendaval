"""Tests for src/models/nn_model.py — Keras/TensorFlow models.

src.cnn_emos.nn_distributions is not yet in the repo; the stub is injected
via test/conftest.py. TensorFlow is skipped gracefully if not installed.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from src.models.nn_model import NNModel, NNConvModel  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(seed=0)


def _mock_distribution(num_outputs: int = 2):
    """Return a minimal mock of NNDistribution."""
    dist = MagicMock()
    dist.build_output_layers.return_value = [
        tf.keras.layers.Dense(1) for _ in range(num_outputs)
    ]
    dist.add_forecast.side_effect = lambda outputs, inputs: outputs
    dist.has_gev.return_value = False
    return dist


def _nn_model(**overrides):
    defaults = {
        "forecast_distribution": _mock_distribution(),
        "hidden_units_list": [32],
        "dense_l1_regularization": 0.0,
        "dense_l2_regularization": 0.0,
        "add_nwp_forecast": False,
    }
    defaults.update(overrides)
    return NNModel(**defaults)


def _conv_model(**overrides):
    defaults = {
        "forecast_distribution": _mock_distribution(),
        "hidden_units_list": [16],
        "dense_l1_regularization": 0.0,
        "dense_l2_regularization": 0.0,
        "conv_7x7_units": 1,
        "conv_5x5_units": 1,
        "conv_3x3_units": 1,
    }
    defaults.update(overrides)
    return NNConvModel(**defaults)


# ---------------------------------------------------------------------------
# NNBaseModel (tested via NNModel instances)
# ---------------------------------------------------------------------------


class TestNNBaseModel:
    def test_get_forecast_distribution(self):
        dist = _mock_distribution()
        model = _nn_model(forecast_distribution=dist)
        assert model.get_forecast_distribution() is dist


# ---------------------------------------------------------------------------
# NNModel
# ---------------------------------------------------------------------------


class TestNNModel:
    def test_instantiation_no_kwargs(self):
        dist = MagicMock()
        model = NNModel(forecast_distribution=dist)
        assert model._forecast_distribution is dist

    def test_hidden_layers_count(self):
        model = _nn_model(hidden_units_list=[128, 64, 32])
        assert len(model.hidden_layers) == 3

    def test_no_regularization(self):
        model = _nn_model(
            dense_l1_regularization=0.0,
            dense_l2_regularization=0.0,
        )
        assert model.hidden_layers[0].kernel_regularizer is None

    def test_l1_regularization_applied(self):
        model = _nn_model(
            dense_l1_regularization=0.01,
            dense_l2_regularization=0.0,
        )
        assert model.hidden_layers[0].kernel_regularizer is not None

    def test_l2_regularization_applied(self):
        model = _nn_model(
            dense_l1_regularization=0.0,
            dense_l2_regularization=0.01,
        )
        assert model.hidden_layers[0].kernel_regularizer is not None

    def test_l1_l2_regularization_applied(self):
        model = _nn_model(
            dense_l1_regularization=0.01,
            dense_l2_regularization=0.01,
        )
        assert model.hidden_layers[0].kernel_regularizer is not None

    def test_setup_dict_keys(self):
        model = _nn_model(
            hidden_units_list=[64, 32],
            add_nwp_forecast=True,
        )
        assert model.setup["hidden_units_list"] == [64, 32]
        assert model.setup["add_nwp_forecast"] is True

    def test_forward_without_nwp(self):
        model = _nn_model(hidden_units_list=[16], add_nwp_forecast=False)
        batch = 4
        inputs = {
            "features_1d": tf.constant(
                _rng.standard_normal((batch, 10)).astype("float32")
            ),
        }
        out = model(inputs)
        assert out.shape[0] == batch

    def test_forward_with_nwp(self):
        dist = _mock_distribution()
        model = _nn_model(
            forecast_distribution=dist,
            hidden_units_list=[16],
            add_nwp_forecast=True,
        )
        inputs = {
            "features_1d": tf.constant(
                _rng.standard_normal((4, 10)).astype("float32")
            ),
        }
        model(inputs)
        # TF may trace the graph more than once on first call
        dist.add_forecast.assert_called()


# ---------------------------------------------------------------------------
# NNConvModel
# ---------------------------------------------------------------------------


class TestNNConvModel:
    def test_instantiation_no_kwargs(self):
        dist = MagicMock()
        model = NNConvModel(forecast_distribution=dist)
        assert model._forecast_distribution is dist

    def test_conv_layer_counts(self):
        model = _conv_model(
            conv_7x7_units=2,
            conv_5x5_units=3,
            conv_3x3_units=4,
        )
        assert len(model.conv_7x7_layers) == 2
        assert len(model.conv_5x5_layers) == 3
        assert len(model.conv_3x3_layers) == 4

    def test_batch_norm_layers_match_conv_layers(self):
        model = _conv_model(
            conv_7x7_units=2, conv_5x5_units=2, conv_3x3_units=2
        )
        assert len(model.batch_norm_7x7_layers) == len(model.conv_7x7_layers)
        assert len(model.batch_norm_5x5_layers) == len(model.conv_5x5_layers)
        assert len(model.batch_norm_3x3_layers) == len(model.conv_3x3_layers)

    def test_hidden_layers_count(self):
        model = _conv_model(hidden_units_list=[64, 32])
        assert len(model.hidden_layers) == 2

    def test_has_gev_delegates_to_distribution(self):
        dist = _mock_distribution()
        dist.has_gev.return_value = True
        model = NNConvModel(
            forecast_distribution=dist,
            hidden_units_list=[16],
            dense_l1_regularization=0.0,
            dense_l2_regularization=0.0,
            conv_7x7_units=1,
            conv_5x5_units=1,
            conv_3x3_units=1,
        )
        assert model.has_gev() is True

    def test_setup_dict(self):
        model = _conv_model(
            conv_7x7_units=4,
            conv_5x5_units=4,
            conv_3x3_units=4,
            dense_l2_regularization=0.031658,
        )
        assert model.setup["conv_7x7_units"] == 4
        assert model.setup["dense_l2_regularization"] == pytest.approx(
            0.031658
        )

    def test_forward_output_batch_size(self):
        batch = 2
        model = _conv_model()
        inputs = {
            "wind_speed_grid": tf.constant(
                _rng.standard_normal((batch, 32, 32, 1)).astype("float32")
            ),
            "features_1d": tf.constant(
                _rng.standard_normal((batch, 10)).astype("float32")
            ),
            "wind_speed_forecast": tf.constant(
                _rng.standard_normal((batch, 1)).astype("float32")
            ),
        }
        out = model(inputs)
        assert out.shape[0] == batch

    def test_get_forecast_distribution(self):
        dist = _mock_distribution()
        model = _conv_model(forecast_distribution=dist)
        assert model.get_forecast_distribution() is dist
