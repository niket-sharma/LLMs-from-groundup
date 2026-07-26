"""
Parity / correctness tests for M1.4 attention anatomy.

We verify the from-scratch scaled-dot-product attention against PyTorch's
reference kernel, prove the mask variants are equivalent / leak-free, and pin
down the entropy and attention-sink metrics at their known extremes.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GPTConfig, create_model  # noqa: E402
from m1_fundamentals.attention_deep_dive import (  # noqa: E402
    sdpa_annotated,
    multihead_from_single,
    causal_mask_boolean,
    causal_mask_additive,
    sliding_window_mask,
    attention_entropy,
    attention_sink_score,
    collect_attention,
)


# ------------------------------------------------------------------- SDPA parity
def test_sdpa_matches_torch_reference():
    """Our annotated SDPA must equal F.scaled_dot_product_attention (causal)."""
    torch.manual_seed(0)
    B, H, T, Dh = 2, 3, 7, 16
    Q, K, V = torch.randn(B, H, T, Dh), torch.randn(B, H, T, Dh), torch.randn(B, H, T, Dh)
    ours, _ = sdpa_annotated(Q, K, V, causal_mask_additive(T))
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_sdpa_rows_sum_to_one():
    torch.manual_seed(0)
    Q, K, V = (torch.randn(1, 2, 5, 8) for _ in range(3))
    _, attn = sdpa_annotated(Q, K, V, causal_mask_additive(5))
    assert torch.allclose(attn.sum(-1), torch.ones(1, 2, 5), atol=1e-6)


# -------------------------------------------------------------------- mask variants
def test_additive_equals_boolean():
    torch.manual_seed(0)
    T = 6
    Q, K, V = (torch.randn(1, 2, T, 8) for _ in range(3))
    _, attn_add = sdpa_annotated(Q, K, V, causal_mask_additive(T))
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(8)
    attn_bool = F.softmax(scores.masked_fill(~causal_mask_boolean(T), float("-inf")), dim=-1)
    assert torch.allclose(attn_add, attn_bool, atol=1e-6)


def test_causal_no_future_leakage():
    """Attention above the diagonal must be exactly zero."""
    torch.manual_seed(0)
    T = 8
    Q, K, V = (torch.randn(1, 1, T, 8) for _ in range(3))
    _, attn = sdpa_annotated(Q, K, V, causal_mask_additive(T))
    upper = torch.triu(attn[0, 0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7)


def test_sliding_window_mask_allowed_keys():
    T, w = 10, 4
    m = sliding_window_mask(T, w)
    # Query 8 may attend to keys 5,6,7,8 (last `w` including itself).
    allowed = (m[8] == 0).nonzero().flatten().tolist()
    assert allowed == [5, 6, 7, 8]
    # Query 1 (fewer than w prior tokens) attends to 0,1 only.
    assert (m[1] == 0).nonzero().flatten().tolist() == [0, 1]
    # Never attends to the future.
    assert torch.isinf(m[3, 4])


def test_sliding_window_full_equals_causal():
    T = 6
    full = sliding_window_mask(T, window=T)
    causal = causal_mask_additive(T)
    # Same allowed/disallowed pattern (compare finiteness).
    assert torch.equal(torch.isinf(full), torch.isinf(causal))


# --------------------------------------------------------- multi-head reshape parity
def test_multihead_single_head_equals_sdpa():
    """With n_heads=1 the multi-head path reduces to plain SDPA (up to Wo)."""
    torch.manual_seed(0)
    B, T, d = 1, 5, 16
    x = torch.randn(B, T, d)
    eye = torch.eye(d)
    mask = causal_mask_additive(T)
    out, attn = multihead_from_single(x, eye, eye, eye, eye, n_heads=1, mask=mask)
    ref_ctx, ref_attn = sdpa_annotated(
        x.view(B, T, 1, d).transpose(1, 2),
        x.view(B, T, 1, d).transpose(1, 2),
        x.view(B, T, 1, d).transpose(1, 2),
        mask,
    )
    assert torch.allclose(attn, ref_attn, atol=1e-6)
    assert torch.allclose(out, ref_ctx.transpose(1, 2).reshape(B, T, d), atol=1e-6)


def test_multihead_shapes():
    torch.manual_seed(0)
    B, T, d, H = 2, 9, 32, 4
    x = torch.randn(B, T, d)
    W = [torch.randn(d, d) for _ in range(4)]
    out, attn = multihead_from_single(x, *W, n_heads=H, mask=causal_mask_additive(T))
    assert out.shape == (B, T, d)
    assert attn.shape == (B, H, T, T)


# ------------------------------------------------------------------ entropy / sink
def test_entropy_uniform_is_log_t():
    T = 16
    uniform = torch.full((1, 1, 1, T), 1.0 / T)
    assert abs(attention_entropy(uniform) - math.log(T)) < 1e-4


def test_entropy_onehot_is_zero():
    onehot = torch.zeros(1, 1, 1, 8)
    onehot[..., 0] = 1.0
    assert attention_entropy(onehot) < 1e-6


def test_sink_score_range_and_meaning():
    # All mass on token 0 -> sink score 1.
    attn = torch.zeros(1, 1, 4, 4)
    attn[..., 0] = 1.0
    assert abs(attention_sink_score(attn) - 1.0) < 1e-6
    # No mass on token 0 -> sink score 0.
    attn2 = torch.zeros(1, 1, 4, 4)
    attn2[..., -1] = 1.0
    assert attention_sink_score(attn2) < 1e-6


# ----------------------------------------------------------------- model capture
def test_collect_attention_shapes():
    model = create_model(GPTConfig(preset="tiny", vocab_size=100))
    x = torch.randint(0, 100, (2, 12))
    maps = collect_attention(model, x)
    assert len(maps) == model.n_layers
    for attn in maps:
        assert attn.shape == (2, model.n_heads, 12, 12)
        # Each is a valid causal distribution.
        assert torch.allclose(attn.sum(-1), torch.ones(2, model.n_heads, 12), atol=1e-5)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
