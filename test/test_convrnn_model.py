"""Tests for src/models/convrnn_model.py — ConvGRU and ConvLSTM cells."""
import pytest
import torch
import torch.nn as nn

from src.models.convrnn_model import CGRU_cell, CLSTM_cell

CUDA = torch.cuda.is_available()


class TestCGRUCell:
    def test_instantiation(self):
        cell = CGRU_cell(shape=(16, 16), input_channels=1, filter_size=3, num_features=32, seq_len=5)
        assert isinstance(cell, nn.Module)

    def test_attributes_stored(self):
        cell = CGRU_cell(shape=(8, 8), input_channels=4, filter_size=5, num_features=64, seq_len=10)
        assert cell.shape == (8, 8)
        assert cell.input_channels == 4
        assert cell.filter_size == 5
        assert cell.num_features == 64
        assert cell.seq_len == 10

    @pytest.mark.parametrize("filter_size", [3, 5, 7])
    def test_padding_is_same(self, filter_size):
        cell = CGRU_cell(shape=(16, 16), input_channels=1, filter_size=filter_size, num_features=32, seq_len=5)
        assert cell.padding == (filter_size - 1) // 2

    def test_conv1_outputs_two_gate_channels(self):
        """conv1 produces reset and update gates → 2 * num_features channels."""
        num_features = 32
        cell = CGRU_cell(shape=(16, 16), input_channels=1, filter_size=3, num_features=num_features, seq_len=5)
        conv = cell.conv1[0]
        assert isinstance(conv, nn.Conv2d)
        assert conv.out_channels == 2 * num_features

    def test_conv2_outputs_hidden_channels(self):
        """conv2 produces the candidate hidden state → num_features channels."""
        num_features = 32
        cell = CGRU_cell(shape=(16, 16), input_channels=1, filter_size=3, num_features=num_features, seq_len=5)
        conv = cell.conv2[0]
        assert isinstance(conv, nn.Conv2d)
        assert conv.out_channels == num_features

    def test_conv1_input_channels(self):
        """conv1 takes input_channels + num_features (x concatenated with h)."""
        input_channels, num_features = 3, 32
        cell = CGRU_cell(shape=(16, 16), input_channels=input_channels, filter_size=3, num_features=num_features, seq_len=5)
        assert cell.conv1[0].in_channels == input_channels + num_features

    @pytest.mark.skipif(not CUDA, reason="CUDA not available")
    def test_forward_output_shapes(self):
        seq_len, batch, H, W = 5, 2, 16, 16
        input_channels, num_features = 1, 32
        cell = CGRU_cell(shape=(H, W), input_channels=input_channels, filter_size=3,
                         num_features=num_features, seq_len=seq_len).cuda()
        inputs = torch.randn(seq_len, batch, input_channels, H, W).cuda()
        outputs, last_hidden = cell(inputs)
        assert outputs.shape == (seq_len, batch, num_features, H, W)
        assert last_hidden.shape == (batch, num_features, H, W)

    @pytest.mark.skipif(not CUDA, reason="CUDA not available")
    def test_forward_with_explicit_hidden_state(self):
        seq_len, batch, H, W = 3, 2, 16, 16
        num_features = 32
        cell = CGRU_cell(shape=(H, W), input_channels=1, filter_size=3,
                         num_features=num_features, seq_len=seq_len).cuda()
        inputs = torch.randn(seq_len, batch, 1, H, W).cuda()
        h0 = torch.zeros(batch, num_features, H, W).cuda()
        outputs, _ = cell(inputs, hidden_state=h0)
        assert outputs.shape == (seq_len, batch, num_features, H, W)


class TestCLSTMCell:
    def test_instantiation(self):
        cell = CLSTM_cell(shape=(16, 16), input_channels=1, filter_size=3, num_features=32, seq_len=5)
        assert isinstance(cell, nn.Module)

    def test_attributes_stored(self):
        cell = CLSTM_cell(shape=(8, 8), input_channels=4, filter_size=5, num_features=64, seq_len=10)
        assert cell.shape == (8, 8)
        assert cell.input_channels == 4
        assert cell.filter_size == 5
        assert cell.num_features == 64
        assert cell.seq_len == 10

    @pytest.mark.parametrize("filter_size", [3, 5, 7])
    def test_padding_is_same(self, filter_size):
        cell = CLSTM_cell(shape=(16, 16), input_channels=1, filter_size=filter_size, num_features=32, seq_len=5)
        assert cell.padding == (filter_size - 1) // 2

    def test_conv_outputs_four_gate_channels(self):
        """Single conv produces i, f, g, o gates → 4 * num_features channels."""
        num_features = 32
        cell = CLSTM_cell(shape=(16, 16), input_channels=1, filter_size=3, num_features=num_features, seq_len=5)
        conv = cell.conv[0]
        assert isinstance(conv, nn.Conv2d)
        assert conv.out_channels == 4 * num_features

    def test_conv_input_channels(self):
        """conv takes input_channels + num_features (x concatenated with h)."""
        input_channels, num_features = 3, 32
        cell = CLSTM_cell(shape=(16, 16), input_channels=input_channels, filter_size=3,
                          num_features=num_features, seq_len=5)
        assert cell.conv[0].in_channels == input_channels + num_features

    @pytest.mark.skipif(not CUDA, reason="CUDA not available")
    def test_forward_output_shapes(self):
        seq_len, batch, H, W = 5, 2, 16, 16
        input_channels, num_features = 1, 32
        cell = CLSTM_cell(shape=(H, W), input_channels=input_channels, filter_size=3,
                          num_features=num_features, seq_len=seq_len).cuda()
        inputs = torch.randn(seq_len, batch, input_channels, H, W).cuda()
        outputs, (hy, cy) = cell(inputs)
        assert outputs.shape == (seq_len, batch, num_features, H, W)
        assert hy.shape == (batch, num_features, H, W)
        assert cy.shape == (batch, num_features, H, W)

    @pytest.mark.skipif(not CUDA, reason="CUDA not available")
    def test_forward_with_explicit_hidden_state(self):
        seq_len, batch, H, W = 3, 2, 16, 16
        num_features = 32
        cell = CLSTM_cell(shape=(H, W), input_channels=1, filter_size=3,
                          num_features=num_features, seq_len=seq_len).cuda()
        inputs = torch.randn(seq_len, batch, 1, H, W).cuda()
        h0 = torch.zeros(batch, num_features, H, W).cuda()
        c0 = torch.zeros(batch, num_features, H, W).cuda()
        outputs, (hy, cy) = cell(inputs, hidden_state=(h0, c0))
        assert outputs.shape == (seq_len, batch, num_features, H, W)
