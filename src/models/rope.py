"""
Rotary Positional Embeddings (RoPE) — from scratch.

RoPE (Su et al., 2021; used by LLaMA, GPT-NeoX, Qwen, Mistral, ...) is the
default positional encoding for essentially every modern LLM, and the default
for all later modules in this repo. This file is the *production* core that the
model factory wires in; `modules/m1_fundamentals/positional_encodings.py` has
the slower, heavily-annotated educational derivations and the extrapolation
experiment.

The idea in one paragraph
-------------------------
Instead of *adding* a position vector to the token embedding (learned/sinusoidal
absolute PE), RoPE *rotates* the query and key vectors by an angle proportional
to their absolute position, done per 2-D sub-space of the head dimension. The
magic: the dot product between a query at position m and a key at position n then
depends only on their **relative** offset (m - n), because rotating both by their
absolute angle and taking an inner product leaves a function of the angle
difference. So you get relative-position awareness with zero extra parameters,
applied at attention time (not added to the residual stream).

Complex-number view
--------------------
Pair up the head dims: (x0,x1),(x2,x3),... Treat each pair as a complex number
z = x_even + i·x_odd. Rotating by angle θ = position · freq is multiplication by
e^{iθ}. Since <R_m q, R_n k> for rotations depends on (m-n)·freq, the attention
score is inherently relative. We implement the equivalent real-valued
"rotate_half" form (the LLaMA/GPT-NeoX convention) because it's faster in PyTorch.

Frequencies
-----------
freq_i = base^(-2i/d) for i in [0, d/2). Low i → fast rotation (fine position),
high i → slow rotation (coarse position). base=10000 is the GPT-NeoX default.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    The 'rotate_half' trick: split the last dim in two halves [x1 | x2] and
    return [-x2 | x1]. Combined with cos/sin below this realizes a rotation of
    every (x1_i, x2_i) pair by the same per-dim angle.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embedding to x.

    Args:
        x:   (batch, n_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)   — cos(position · freq), each freq duplicated
        sin: (seq_len, head_dim)   — sin(position · freq), each freq duplicated

    Returns:
        Rotated x, same shape. Equivalent to multiplying each 2-D sub-vector by
        its rotation matrix [[cosθ, -sinθ], [sinθ, cosθ]].
    """
    # Broadcast cos/sin over batch and head dims: (1, 1, seq_len, head_dim).
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    Precomputes and caches the cos/sin tables for RoPE.

    Context-length extension (used later for long-context discussion):
      - scaling="linear" (Position Interpolation, Chen et al. 2023): divide every
        position by `scale_factor`, squeezing a longer sequence into the trained
        range. Cheap, needs a little fine-tuning to recover quality.
      - scaling="ntk" (NTK-aware / "dynamic" base scaling): increase the RoPE base
        so high-frequency dims are interpolated less and low-frequency dims more,
        extending context *without* fine-tuning by preserving fine-grained
        resolution. Implemented as base' = base · scale_factor^(d/(d-2)).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 1024,
        base: float = 10000.0,
        scaling: str = "none",       # none | linear | ntk
        scale_factor: float = 1.0,   # >1 extends context by this factor
    ):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"
        self.head_dim = head_dim
        self.base = base
        self.scaling = scaling
        self.scale_factor = scale_factor

        eff_base = base
        if scaling == "ntk" and scale_factor != 1.0:
            # NTK-aware base scaling: stretch the base so the wavelength grows.
            eff_base = base * (scale_factor ** (head_dim / (head_dim - 2)))

        # inv_freq: (head_dim/2,)  — one frequency per rotation plane.
        inv_freq = 1.0 / (eff_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=torch.float)
        if self.scaling == "linear" and self.scale_factor != 1.0:
            # Position Interpolation: compress positions into the trained range.
            positions = positions / self.scale_factor
        # Outer product: (seq_len, head_dim/2) angles.
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate to full head_dim so cos/sin align with rotate_half's layout.
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int, device=None, dtype=None):
        """Return (cos, sin) tables of shape (seq_len, head_dim)."""
        if seq_len > self._cached_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin


def sinusoidal_positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """
    The original "Attention Is All You Need" absolute sinusoidal PE, as a plain
    tensor of shape (seq_len, d_model). Added to token embeddings (not applied in
    attention like RoPE). Even dims use sin, odd dims use cos of geometrically
    spaced frequencies — deterministic, parameter-free, and extrapolates to
    unseen positions (unlike learned PE, which has no embedding past max_seq_len).
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(base)) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
