"""
Feed-forward and transformer block components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .norms import make_norm


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (GPT-2 style).

    A two-layer MLP with a GELU activation in between, applied independently to
    each position. Hidden dim is typically 4 * d_model (two matrices → 8·d² params).

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


def swiglu_hidden_dim(d_ff: int) -> int:
    """
    The '2/3 · 4d' convention (M1.3).

    A GELU FFN has 2 weight matrices of size d×d_ff (params ≈ 2·d·d_ff). SwiGLU
    has *three* (gate, up, down), so to keep the parameter count comparable when
    swapping GELU→SwiGLU we shrink the hidden dim to 2/3 of d_ff. Rounded to a
    multiple of 8 for hardware (tensor-core) friendliness, as LLaMA does.
    """
    hidden = int(2 * d_ff / 3)
    return max(8, 8 * round(hidden / 8))


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU feed-forward network (LLaMA/PaLM/Qwen).

        SwiGLU(x) = W_down( SiLU(W_gate x) ⊙ (W_up x) )

    A *gated* activation: one linear branch (`up`) is modulated elementwise by a
    SiLU-activated gate branch (`gate`). The gate lets the network learn which
    features to pass — empirically better than a plain GELU MLP at equal params.
    SiLU (a.k.a. swish) is x·sigmoid(x). Biases are dropped (LLaMA convention).

    `d_ff` is the *equivalent GELU* hidden dim (e.g. 4·d_model); the actual
    hidden used is ``swiglu_hidden_dim(d_ff)`` so params ≈ the GELU FFN.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        hidden = swiglu_hidden_dim(d_ff)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))     # SiLU(W_gate x)
        up = self.w_up(x)                 # W_up x
        x = self.w_down(self.dropout(gate * up))
        return self.dropout(x)


def make_ffn(activation: str, d_model: int, d_ff: int, dropout: float) -> nn.Module:
    """Build the FFN selected by config: 'gelu' -> FeedForward, 'swiglu' -> SwiGLU."""
    if activation == "gelu":
        return FeedForward(d_model, d_ff, dropout)
    if activation == "swiglu":
        return SwiGLUFeedForward(d_model, d_ff, dropout)
    raise ValueError(f"unknown activation: {activation!r}")


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
        rope=None,
        norm: str = "layernorm",
        activation: str = "gelu",
        qk_norm: bool = False,
    ):
        super().__init__()

        # Pre-LN architecture; norm kind is config-driven (LayerNorm | RMSNorm).
        self.ln1 = make_norm(norm, d_model)
        self.ln2 = make_norm(norm, d_model)

        # Self-attention (rope threaded to Q/K; optional QK-norm for stability)
        self.attention = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout, rope=rope, qk_norm=qk_norm
        )

        # Feed-forward network (GELU MLP | SwiGLU gated MLP)
        self.ffn = make_ffn(activation, d_model, d_ff, dropout)

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
