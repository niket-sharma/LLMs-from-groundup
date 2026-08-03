"""
Attention mechanisms for transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .rope import apply_rotary
from .norms import RMSNorm


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    This implements the scaled dot-product attention across multiple heads,
    allowing the model to jointly attend to information from different
    representation subspaces.

    Args:
        d_model: The dimension of the model (embedding dimension)
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        rope: Optional RotaryEmbedding. When provided, RoPE is applied to Q and K
            per head (M1.2). When None, positional info comes from the embedding
            layer (learned/sinusoidal) instead.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, rope=None, qk_norm=False):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension of each head
        self.rope = rope

        # QK-norm (M1.3): RMSNorm over each head's Q and K before scoring. Used
        # in recent models (e.g. Gemma-2, Chameleon) to bound attention-logit
        # magnitude and stabilize training at scale. Applied before RoPE.
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.d_k)
            self.k_norm = RMSNorm(self.d_k)

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        past_key_value=None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # QK-norm: normalize each head's Q and K over the head dimension before
        # rotation/scoring (order: project → qk_norm → rope → scores).
        if self.qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        # RoPE: rotate Q and K by their absolute position so attention scores
        # depend only on relative offset. Applied to Q/K but NOT V (V carries
        # content, not position). Shapes: (batch, n_heads, seq_len, d_k).
        if self.rope is not None:
            seq_len = Q.size(2)
            total_len = position_offset + seq_len
            cos, sin = self.rope(total_len, device=Q.device, dtype=Q.dtype)
            Q = apply_rotary(Q, cos[position_offset:], sin[position_offset:])
            K = apply_rotary(K, cos[position_offset:], sin[position_offset:])

        # Cached K/V are already position-rotated and projected. Concatenating
        # them means decoding one new token only performs O(current_length) work
        # in attention instead of recomputing all earlier projections/blocks.
        if past_key_value is not None:
            past_k, past_v = past_key_value
            K = torch.cat((past_k, K), dim=2)
            V = torch.cat((past_v, V), dim=2)
        present_key_value = (K, V) if use_cache else None

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)

        # Reshape back
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Output projection
        output = self.W_o(context)

        if use_cache:
            return output, attention_weights, present_key_value
        return output, attention_weights


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive language modeling.

    This ensures that each position can only attend to previous positions,
    preventing information leakage from future tokens.

    Args:
        d_model: The dimension of the model
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length (for causal mask)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        rope=None,
        qk_norm=False,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout, rope=rope, qk_norm=qk_norm)
        self.max_seq_len = max_seq_len

        # Register causal mask as a buffer (not a parameter)
        # Lower triangular matrix: position i can attend to positions <= i
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('causal_mask', mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor, past_key_value=None, use_cache: bool = False):
        """
        Forward pass of causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights
        """
        seq_len = x.size(1)
        past_len = 0 if past_key_value is None else past_key_value[0].size(2)
        total_len = past_len + seq_len

        # Query row i is at absolute position past_len + i, so it may attend to
        # every cached key plus new keys through i. Constructing this small mask
        # also avoids a fragile square-mask slice during incremental decoding.
        query_positions = torch.arange(past_len, total_len, device=x.device).view(-1, 1)
        key_positions = torch.arange(total_len, device=x.device).view(1, -1)
        mask = (key_positions <= query_positions).view(1, 1, seq_len, total_len)

        return self.attention(
            x, x, x, mask, past_key_value=past_key_value, use_cache=use_cache,
            position_offset=past_len,
        )
