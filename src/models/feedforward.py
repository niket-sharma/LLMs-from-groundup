"""
Feed-forward and transformer block components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    This is a two-layer MLP with a GELU activation in between.
    Applied independently to each position.

    Args:
        d_model: Input and output dimension
        d_ff: Hidden layer dimension (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, seq_len, d_model)

        Returns:
            Output of shape (batch_size, seq_len, d_model)
        """
        # GELU activation (used in GPT-2, better than ReLU for language models)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block.

    Consists of:
    1. Layer normalization
    2. Causal self-attention with residual connection
    3. Layer normalization
    4. Feed-forward network with residual connection

    This uses Pre-LN (layer norm before attention/FFN) which is more stable
    than Post-LN (original transformer architecture).

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Layer normalizations (Pre-LN architecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Self-attention
        self.attention = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)

        # Feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, seq_len, d_model)

        Returns:
            output: Output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights from self-attention
        """
        # Self-attention with residual connection (Pre-LN)
        attn_out, attn_weights = self.attention(self.ln1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward with residual connection (Pre-LN)
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x, attn_weights
