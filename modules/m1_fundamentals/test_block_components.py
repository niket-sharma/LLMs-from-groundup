"""
Parity / shape tests for M1.3 modern block components: RMSNorm, SwiGLU, QK-norm.

Golden rule: each new component is proven against a reference formula (RMSNorm),
a param-count invariant (SwiGLU's 2/3 convention), and end-to-end model wiring
before any benchmark.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GPTConfig, create_model  # noqa: E402
from models.norms import RMSNorm, make_norm  # noqa: E402
from models.feedforward import (  # noqa: E402
    FeedForward,
    SwiGLUFeedForward,
    swiglu_hidden_dim,
    make_ffn,
)


# ------------------------------------------------------------------- RMSNorm
def test_rmsnorm_matches_reference_formula():
    torch.manual_seed(0)
    dim = 32
    x = torch.randn(4, 10, dim)
    norm = RMSNorm(dim)
    # Reference: x / sqrt(mean(x^2) + eps) * weight, all in fp32.
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
    assert torch.allclose(norm(x), ref, atol=1e-6)


def test_rmsnorm_unit_rms_when_weight_one():
    """With weight=1, RMSNorm output has unit RMS per token."""
    torch.manual_seed(0)
    x = torch.randn(8, 64) * 5.0 + 3.0  # arbitrary scale/shift
    out = RMSNorm(64)(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_does_not_mean_center():
    """Unlike LayerNorm, RMSNorm keeps the mean (no re-centering)."""
    torch.manual_seed(0)
    x = torch.randn(8, 64) + 10.0  # strong positive mean
    rms_out = RMSNorm(64)(x)
    ln_out = nn.LayerNorm(64)(x)
    # LayerNorm forces ~zero mean; RMSNorm does not.
    assert ln_out.mean(-1).abs().max() < 1e-4
    assert rms_out.mean(-1).abs().max() > 0.1


def test_rmsnorm_fewer_params_than_layernorm():
    ln = make_norm("layernorm", 128)
    rms = make_norm("rmsnorm", 128)
    ln_params = sum(p.numel() for p in ln.parameters())
    rms_params = sum(p.numel() for p in rms.parameters())
    # LayerNorm has weight+bias (256); RMSNorm has only weight (128).
    assert ln_params == 256
    assert rms_params == 128


def test_make_norm_unknown_raises():
    try:
        make_norm("batchnorm", 16)
        assert False, "expected ValueError"
    except ValueError:
        pass


# -------------------------------------------------------------------- SwiGLU
def test_swiglu_hidden_dim_two_thirds_convention():
    # 2/3 * d_ff, rounded to a multiple of 8.
    assert swiglu_hidden_dim(1536) == 1024        # 2/3*1536 = 1024
    assert swiglu_hidden_dim(3072) == 2048        # 2/3*3072 = 2048
    assert swiglu_hidden_dim(100) % 8 == 0        # always a multiple of 8


def test_swiglu_shape():
    torch.manual_seed(0)
    x = torch.randn(2, 12, 64)
    ffn = SwiGLUFeedForward(d_model=64, d_ff=256, dropout=0.0)
    assert ffn(x).shape == (2, 12, 64)


def test_swiglu_param_count_matches_gelu():
    """The 2/3 convention keeps SwiGLU params ≈ a GELU FFN at the same d_ff."""
    d_model, d_ff = 128, 512
    gelu = FeedForward(d_model, d_ff)
    swiglu = SwiGLUFeedForward(d_model, d_ff)
    g = sum(p.numel() for p in gelu.parameters())
    s = sum(p.numel() for p in swiglu.parameters())
    # Within 5% (rounding to multiple-of-8 causes small drift); SwiGLU has no bias.
    assert abs(g - s) / g < 0.05, (g, s)


def test_make_ffn_selects_type():
    assert isinstance(make_ffn("gelu", 32, 128, 0.0), FeedForward)
    assert isinstance(make_ffn("swiglu", 32, 128, 0.0), SwiGLUFeedForward)


# --------------------------------------------------------------- model wiring
def test_model_builds_with_each_component():
    torch.manual_seed(0)
    x = torch.randint(0, 200, (2, 16))
    for kwargs in (
        {"norm": "rmsnorm"},
        {"activation": "swiglu"},
        {"qk_norm": True},
        {"pos_encoding": "rope", "norm": "rmsnorm", "activation": "swiglu", "qk_norm": True},
    ):
        model = create_model(GPTConfig(preset="tiny", vocab_size=200, **kwargs)).eval()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 16, 200)


def test_qk_norm_adds_norm_modules():
    model = create_model(GPTConfig(preset="tiny", vocab_size=100, qk_norm=True))
    attn = model.blocks[0].attention.attention
    assert attn.qk_norm
    assert isinstance(attn.q_norm, RMSNorm)
    assert isinstance(attn.k_norm, RMSNorm)


def test_backward_compatible_defaults():
    """Default stack is unchanged GPT-2: LayerNorm + GELU, no qk_norm."""
    model = create_model(GPTConfig(preset="tiny"))
    assert model.norm == "layernorm"
    assert model.activation == "gelu"
    assert isinstance(model.ln_f, nn.LayerNorm)
    assert isinstance(model.blocks[0].ffn, FeedForward)
    assert not model.blocks[0].attention.attention.qk_norm


def test_modern_stack_trains_a_step():
    """A LLaMA-ish stack takes a gradient step without NaN."""
    torch.manual_seed(0)
    model = create_model(GPTConfig(
        preset="tiny", vocab_size=100,
        pos_encoding="rope", norm="rmsnorm", activation="swiglu", qk_norm=True,
    ))
    x = torch.randint(0, 100, (4, 16))
    _, loss = model(x, x)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
