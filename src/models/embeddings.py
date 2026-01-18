"""
Embedding layers for transformer models.
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Converts token indices to dense vectors.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Embedding dimension
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Token indices of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) as in the original transformer
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds positional information to token embeddings using sine and cosine
    functions of different frequencies.

    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved and loaded)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (seq_len, batch_size, d_model)

        Returns:
            Output with positional encoding added
        """
        if x.dim() == 3 and x.size(0) != x.size(1):
            # Handle (seq_len, batch_size, d_model) format
            if x.size(0) < x.size(1):
                # Likely (seq_len, batch_size, d_model)
                seq_len = x.size(0)
                x = x + self.pe[:, :seq_len, :].transpose(0, 1)
            else:
                # Likely (batch_size, seq_len, d_model)
                seq_len = x.size(1)
                x = x + self.pe[:, :seq_len, :]
        else:
            seq_len = x.size(1) if x.dim() == 3 else x.size(0)
            x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class GPTEmbedding(nn.Module):
    """
    Combined embedding layer for GPT-style models.

    Combines token embeddings with learned positional embeddings.
    GPT uses learned positional embeddings rather than sinusoidal.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model/embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding (GPT-style)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Token indices of shape (batch_size, seq_len)

        Returns:
            Embeddings of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = x.size()

        # Get token embeddings
        token_emb = self.token_embedding(x)

        # Get position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # Get position embeddings
        pos_emb = self.position_embedding(positions)

        # Combine and apply dropout
        embeddings = self.dropout(token_emb + pos_emb)

        return embeddings
