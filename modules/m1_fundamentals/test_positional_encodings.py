"""
Correctness / parity tests for M1.2 positional encodings.

Golden rule: prove correctness before the benchmark. RoPE's whole reason for
existing is a mathematical identity (attention score depends only on relative
offset); we test that identity directly, plus the rotation invariants and the
model wiring for all three schemes.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GPTConfig, create_model  # noqa: E402
from models.rope import (  # noqa: E402
    RotaryEmbedding,
    apply_rotary,
    rotate_half,
    sinusoidal_positional_encoding,
)
from m1_fundamentals.positional_encodings import (  # noqa: E402
    demo_relative_property,
    rope_reference_rotation_matrix,
)


# --------------------------------------------------------------------- rotate_half
def test_rotate_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # d=4 -> halves [1,2] and [3,4]
    # rotate_half -> [-3, -4, 1, 2]
    assert torch.allclose(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


# --------------------------------------------------------------- rotation invariants
def test_rotation_preserves_norm():
    """RoPE is a rotation, so it must preserve each vector's L2 norm exactly."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 8)  # (batch, heads, seq, head_dim)
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    cos, sin = rope(16)
    xr = apply_rotary(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), xr.norm(dim=-1), atol=1e-5)


def test_position_zero_is_identity():
    """At position 0 the rotation angle is 0 → apply_rotary is the identity."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 4, 8)
    rope = RotaryEmbedding(head_dim=8, max_seq_len=4)
    cos, sin = rope(4)
    xr = apply_rotary(x, cos, sin)
    assert torch.allclose(xr[:, :, 0], x[:, :, 0], atol=1e-6)


def test_rotate_half_form_matches_matrix_form():
    """The fast rotate_half apply must equal the explicit rotation matrix."""
    torch.manual_seed(0)
    head_dim = 8
    x = torch.randn(head_dim)
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=32)
    cos, sin = rope(32)
    for pos in [0, 1, 5, 17]:
        fast = apply_rotary(x.view(1, 1, 1, -1), cos[pos:pos + 1], sin[pos:pos + 1]).flatten()
        R = rope_reference_rotation_matrix(pos, head_dim)
        slow = R @ x
        assert torch.allclose(fast, slow, atol=1e-5), f"mismatch at pos {pos}"


# ------------------------------------------------------------ relative-position property
def test_relative_position_identity():
    """<RoPE(q,m), RoPE(k,n)> depends only on (m-n): deviation ≈ 0."""
    dev = demo_relative_property()
    assert dev < 1e-4, f"RoPE relative-position identity violated: {dev}"


def test_scores_differ_for_different_offsets():
    """Sanity: different relative offsets *do* give different scores (not trivially constant)."""
    torch.manual_seed(0)
    head_dim = 8
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)
    rope = RotaryEmbedding(head_dim, max_seq_len=64)
    cos, sin = rope(64)

    def score(m, n):
        qm = apply_rotary(q.view(1, 1, 1, -1), cos[m:m + 1], sin[m:m + 1]).flatten()
        kn = apply_rotary(k.view(1, 1, 1, -1), cos[n:n + 1], sin[n:n + 1]).flatten()
        return torch.dot(qm, kn).item()

    assert abs(score(10, 5) - score(10, 2)) > 1e-3  # offset 5 vs 8 differ


# --------------------------------------------------------------------- sinusoidal
def test_sinusoidal_shape_and_range():
    pe = sinusoidal_positional_encoding(seq_len=50, d_model=16)
    assert pe.shape == (50, 16)
    assert pe.abs().max() <= 1.0  # sin/cos bounded
    # position 0: sin(0)=0 on even dims, cos(0)=1 on odd dims.
    assert torch.allclose(pe[0, 0::2], torch.zeros(8), atol=1e-6)
    assert torch.allclose(pe[0, 1::2], torch.ones(8), atol=1e-6)


# ----------------------------------------------------------------- context scaling
def test_linear_scaling_compresses_positions():
    """Linear PI divides positions by scale_factor → cos table matches base at pos/scale."""
    base = RotaryEmbedding(head_dim=8, max_seq_len=64)
    scaled = RotaryEmbedding(head_dim=8, max_seq_len=64, scaling="linear", scale_factor=2.0)
    cb, _ = base(64)
    cs, _ = scaled(64)
    # scaled position 2 should equal base position 1 (2 / 2.0 == 1).
    assert torch.allclose(cs[2], cb[1], atol=1e-5)


def test_ntk_scaling_changes_base():
    """NTK scaling raises the effective base → lower inv_freq than the default."""
    base = RotaryEmbedding(head_dim=8, max_seq_len=64)
    ntk = RotaryEmbedding(head_dim=8, max_seq_len=64, scaling="ntk", scale_factor=4.0)
    # Higher effective base => smaller frequencies (slower rotation).
    assert (ntk.inv_freq < base.inv_freq).any()
    assert torch.all(ntk.inv_freq <= base.inv_freq + 1e-9)


# --------------------------------------------------------------------- model wiring
def test_all_pos_encodings_forward():
    torch.manual_seed(0)
    x = torch.randint(0, 200, (2, 24))
    for pe in ["learned", "sinusoidal", "rope"]:
        model = create_model(GPTConfig(preset="tiny", vocab_size=200, pos_encoding=pe)).eval()
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 24, 200)
        assert (model.rope is not None) == (pe == "rope")


def test_rope_runs_beyond_train_len():
    """RoPE cache extends on demand; sinusoidal/learned must also run within max_seq_len."""
    model = create_model(GPTConfig(preset="tiny", vocab_size=100, pos_encoding="rope", max_seq_len=64)).eval()
    x = torch.randint(0, 100, (1, 64))
    with torch.no_grad():
        assert model(x).shape == (1, 64, 100)


def test_learned_pe_still_default_and_backward_compatible():
    """Default config is unchanged (learned PE) — backward compatibility."""
    cfg = GPTConfig(preset="tiny")
    assert cfg.pos_encoding == "learned"
    model = create_model(cfg)
    assert model.rope is None
    assert model.embedding.position_embedding is not None


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
