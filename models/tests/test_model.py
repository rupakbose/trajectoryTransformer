import pytest
import torch
from models.model import Transformer

def test_transformer_forward():
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=25,
        d_model=128,
        num_heads=4,
        num_layers=6,
        d_ff=64,
        max_seq_length=128,
        dropout=0.2
    )
    
    # Test with batch_size=2, seq_length=64
    input_tensor = torch.randint(0, 500, (2, 64))
    output = model(input_tensor)
    
    # Should output (batch_size, seq_length + 1, tgt_vocab_size)
    assert output.shape == (2, 65, 25)