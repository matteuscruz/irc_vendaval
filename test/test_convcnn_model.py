"""Tests for src/models/convcnn_model.py — PyTorch 3D CNN autoencoder."""
import pytest
import torch
import torch.nn as nn

#from src.models.convcnn_model import CNNModel, make_layers

class TestCNNModel:
    def test_instantiation(self):
        model = CNNModel()
        assert isinstance(model, nn.Module)

    def test_has_encoder_layers(self):
        model = CNNModel()
        for i in range(1, 5):
            assert hasattr(model, f"conv_layer{i}")

    def test_has_decoder_layers(self):
        model = CNNModel()
        for i in range(1, 5):
            assert hasattr(model, f"deconv_layer{i}")

    def test_dropout_is_disabled(self):
        model = CNNModel()
        assert model.dropout.p == 0.0

    def test_encode_output_shape(self):
        # (batch, T, C=1, H=32, W=32) -> encode -> (batch, 128, T, 2, 2)
        model = CNNModel()
        model.eval()
        x = torch.randn(2, 10, 1, 32, 32)
        with torch.no_grad():
            out = model.encode(x)
        assert out.shape == (2, 128, 10, 2, 2)

    def test_decode_output_shape(self):
        # Encoded tensor -> decode -> (batch, 1, T, H, W)
        model = CNNModel()
        model.eval()
        encoded = torch.randn(2, 128, 10, 2, 2)
        with torch.no_grad():
            out = model.decode(encoded)
        assert out.shape == (2, 1, 10, 32, 32)

    def test_forward_output_shape(self):
        # Full autoencoder: (batch, T, 1, H, W) -> (batch, T, H, W)
        model = CNNModel()
        model.eval()
        x = torch.randn(2, 10, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10, 32, 32)

    def test_forward_batch_size_1(self):
        # squeeze() removes ALL size-1 dims — documents potentially
        # surprising behaviour when batch=1 loses the batch dimension.
        model = CNNModel()
        model.eval()
        x = torch.randn(1, 10, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.ndim >= 3

    def test_conv_channel_progression(self):
        # Encoder: 1 -> 16 -> 32 -> 64 -> 128
        model = CNNModel()
        expected = [(1, 16), (16, 32), (32, 64), (64, 128)]
        for i, (in_c, out_c) in enumerate(expected, start=1):
            layer = getattr(model, f"conv_layer{i}")[0]
            assert isinstance(layer, nn.Conv3d)
            assert layer.in_channels == in_c
            assert layer.out_channels == out_c

    def test_deconv_channel_progression(self):
        # Decoder: 128 -> 64 -> 32 -> 16 -> 1
        model = CNNModel()
        expected = [(128, 64), (64, 32), (32, 16), (16, 1)]
        for i, (in_c, out_c) in enumerate(expected, start=1):
            layer = getattr(model, f"deconv_layer{i}")[0]
            assert isinstance(layer, nn.ConvTranspose3d)
            assert layer.in_channels == in_c
            assert layer.out_channels == out_c


# make_layers uses OrderedDict without importing it — NameError is the
# expected failure. Remove the xfail marks once the import is added.
_xfail_ordered_dict = pytest.mark.xfail(
    raises=NameError,
    reason="convcnn_model.py uses OrderedDict without importing it",
    strict=True,
)


class TestMakeLayers:
    @_xfail_ordered_dict
    def test_pool_layer(self):
        block = {"pool_1": [2, 2, 0]}
        seq = make_layers(block)
        assert len(seq) == 1
        assert isinstance(seq[0], nn.MaxPool2d)

    @_xfail_ordered_dict
    def test_conv_no_activation(self):
        block = {"conv_1": [1, 8, 3, 1, 1]}
        seq = make_layers(block)
        assert len(seq) == 1
        assert isinstance(seq[0], nn.Conv2d)

    @_xfail_ordered_dict
    def test_conv_relu(self):
        block = {"conv_relu_1": [1, 8, 3, 1, 1]}
        seq = make_layers(block)
        assert len(seq) == 2
        assert isinstance(seq[0], nn.Conv2d)
        assert isinstance(seq[1], nn.ReLU)

    @_xfail_ordered_dict
    def test_conv_leaky(self):
        block = {"conv_leaky_1": [1, 8, 3, 1, 1]}
        seq = make_layers(block)
        assert len(seq) == 2
        assert isinstance(seq[0], nn.Conv2d)
        assert isinstance(seq[1], nn.LeakyReLU)

    @_xfail_ordered_dict
    def test_deconv_relu(self):
        block = {"deconv_relu_1": [16, 1, 3, 1, 1]}
        seq = make_layers(block)
        assert len(seq) == 2
        assert isinstance(seq[0], nn.ConvTranspose2d)
        assert isinstance(seq[1], nn.ReLU)

    @_xfail_ordered_dict
    def test_deconv_leaky(self):
        block = {"deconv_leaky_1": [16, 1, 3, 1, 1]}
        seq = make_layers(block)
        assert len(seq) == 2
        assert isinstance(seq[0], nn.ConvTranspose2d)
        assert isinstance(seq[1], nn.LeakyReLU)

    def test_unknown_layer_raises(self):
        block = {"unknown_1": [1, 2, 3, 4, 5]}
        with pytest.raises(NotImplementedError):
            make_layers(block)
