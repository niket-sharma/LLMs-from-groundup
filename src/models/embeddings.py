"""
Embedding layers for transformer models.
"""

import torch
import torch.nn as nn
import math

from .rope import sinusoidal_positional_encoding


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

    Supports three positional-encoding schemes (M1.2), selected by
    ``pos_encoding``:

    - ``"learned"`` (GPT-2 default): a trainable ``nn.Embedding`` over positions,
      added to the token embedding. Simple, but has no representation for
      positions beyond ``max_seq_len`` (no extrapolation).
    - ``"sinusoidal"`` (original Transformer): a fixed sin/cos table added to the
      token embedding. Parameter-free and defined for any position.
    - ``"rope"``: **no** positional signal is added here — RoPE is applied to
      Q/K inside attention. This layer then only returns token embeddings.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Model/embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        pos_encoding: "learned" | "sinusoidal" | "rope"
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
    ):
        super().__init__()

        self.pos_encoding = pos_encoding

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional component depends on the scheme.
        self.position_embedding = None
        if pos_encoding == "learned":
            # Learned positional embedding (GPT-style)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        elif pos_encoding == "sinusoidal":
            # Fixed sin/cos table stored as a (non-persistent) buffer.
            pe = sinusoidal_positional_encoding(max_seq_len, d_model)
            self.register_buffer("sinusoidal_pe", pe, persistent=False)
        elif pos_encoding == "rope":
            # Position is injected in attention; nothing to add here.
            pass
        else:
            raise ValueError(f"unknown pos_encoding: {pos_encoding!r}")

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

        if self.pos_encoding == "learned":
            positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)
            token_emb = token_emb + self.position_embedding(positions)
        elif self.pos_encoding == "sinusoidal":
            token_emb = token_emb + self.sinusoidal_pe[:seq_len].to(token_emb.dtype)
        # "rope": token embeddings pass through unchanged.

        return self.dropout(token_emb)
