import torch
import torch.nn as nn
from typing import Optional
from .layers import MultiHeadAttention, PositionWiseFeedForward, PositionalEncoding

class EncoderLayer(nn.Module):
    """Single encoder layer in the transformer model.
    
    Args:
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        d_ff: Dimension of feed-forward inner layer
        dropout: Dropout rate
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for encoder layer.
        
        Args:
            x: Input tensor
            mask: Optional mask tensor
            
        Returns:
            Output tensor after self-attention and feed-forward
        """
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    """Transformer model implementation for sequence classification.
    
    Args:
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary (number of classes)
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of encoder layers
        d_ff: Dimension of feed-forward inner layer
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
    """
    def __init__(
        self, 
        src_vocab_size: int, 
        tgt_vocab_size: int, 
        d_model: int = 128, 
        num_heads: int = 4, 
        num_layers: int = 6, 
        d_ff: int = 64, 
        max_seq_length: int = 128, 
        dropout: float = 0.2
    ):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Embedding(1, d_model)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Forward pass for transformer model.
        
        Args:
            src: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Output tensor of shape (batch_size, seq_length + 1, tgt_vocab_size)
        """
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        cls_inp = torch.zeros((src.shape[0], 1), dtype=torch.long).to(src.device)
        cls_embedded = self.cls_token(cls_inp)
        concat_output = torch.cat((cls_embedded, src_embedded), dim=1)
        
        enc_output = concat_output
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output)

        output = self.fc(enc_output)
        return output