"""
Normalization layers — RMSNorm from scratch (M1.3).

LayerNorm (Ba et al., 2016) normalizes each token's activation vector to zero
mean and unit variance, then applies a learned scale (gamma) and shift (beta):

    LN(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta

RMSNorm (Zhang & Sennrich, 2019; the norm in LLaMA, Qwen, Mistral, Gemma) drops
the mean-centering and the bias entirely, normalizing only by the root-mean-
square of the activations:

    RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)

Why RMSNorm replaced LayerNorm in modern LLMs
---------------------------------------------
1. **Cheaper.** No mean subtraction and no bias — fewer ops and one fewer
   reduction. At scale (thousands of norm calls per forward) this matters.
2. **Empirically just as good.** The re-centering LayerNorm does turns out to
   contribute little; the important part is the re-scaling. Removing the mean
   costs no measurable quality on LMs.
3. **Fewer params / simpler.** No beta term.

Both are computed in fp32 internally for stability even when the model runs in
bf16 (a standard trick — the variance/RMS reduction is precision-sensitive).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root-Mean-Square LayerNorm.

    Args:
        dim: feature dimension to normalize over (the last dim).
        eps: numerical-stability epsilon inside the sqrt.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learned per-feature scale (gamma). Initialized to 1 so the layer starts
        # as a pure normalization (identity scale).
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt(mean(x^2)) — note: mean over the last dim, keepdim for broadcast.
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the norm in fp32 for stability, then cast back to x's dtype.
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


def make_norm(kind: str, dim: int, eps: float = 1e-5) -> nn.Module:
    """
    Factory used by the model stack to pick a normalization by config string.

    "layernorm" -> nn.LayerNorm (the GPT-2 default, kept for backward compat)
    "rmsnorm"   -> RMSNorm      (the modern default from M1.3 onward)
    """
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    raise ValueError(f"unknown norm kind: {kind!r}")
