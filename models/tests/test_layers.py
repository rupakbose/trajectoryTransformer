import pytest
import torch
from models.layers import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding

@pytest.fixture
def sample_input():
    return torch.randn(2, 10, 64)  # batch_size=2, seq_length=10, d_model=64

def test_multihead_attention(sample_input):
    mha = MultiHeadAttention(d_model=64, num_heads=4)
    output = mha(sample_input, sample_input, sample_input)
    assert output.shape == sample_input.shape

def test_feedforward(sample_input):
    ff = PositionWiseFeedForward(d_model=64, d_ff=128)
    output = ff(sample_input)
    assert output.shape == sample_input.shape

def test_positional_encoding(sample_input):
    pe = PositionalEncoding(d_model=64, max_seq_length=20)
    output = pe(sample_input)
    assert output.shape == sample_input.shape
    assert not torch.allclose(output, sample_input)  # Should have changed