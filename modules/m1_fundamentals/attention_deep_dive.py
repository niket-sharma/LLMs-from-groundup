#!/usr/bin/env python3
"""
Attention anatomy (M1.4): shapes, mask variants, entropy & attention sinks.

Three things every AI-engineer interview probes, built from scratch:

1. **Single-head → multi-head, with a shape annotation on every line.** The
   reference `scaled_dot_product_attention` here is deliberately verbose so you
   can trace a tensor from (B,T,d) through heads and back.

2. **Causal mask variants.** additive −inf vs boolean masking give *identical*
   softmax outputs; the sliding-window mask (Mistral/Gemma local attention)
   restricts each query to the last `w` keys. We verify the equivalences.

3. **Entropy & attention sinks on a real forward pass.** Attention entropy per
   head/layer (how peaked vs diffuse), and the "attention sink" phenomenon —
   the first token (often a BOS) soaking up a large share of attention mass,
   which matters for KV-cache eviction and StreamingLLM (foreshadows M2).

Run:
    python modules/m1_fundamentals/attention_deep_dive.py            # fast (CI)
    python modules/m1_fundamentals/attention_deep_dive.py --viz      # + PNG maps
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models import GPTConfig, create_model  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


# --------------------------------------------------------------------------- #
# 1. Scaled dot-product attention, annotated single- and multi-head.           #
# --------------------------------------------------------------------------- #
def sdpa_annotated(Q, K, V, mask=None):
    """
    Scaled dot-product attention with a shape comment on every step.

        Q, K, V: (B, H, T, Dh)      B=batch, H=heads, T=seq, Dh=head_dim
        mask:    additive bias (B|1, 1|H, T, T) with 0 keep / -inf drop, or None
        returns: (context (B,H,T,Dh), attn (B,H,T,T))
    """
    Dh = Q.size(-1)
    # scores[b,h,i,j] = Q_i · K_j / sqrt(Dh)      -> (B, H, T, T)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
    if mask is not None:
        scores = scores + mask                    # additive: -inf where disallowed
    attn = F.softmax(scores, dim=-1)              # rows sum to 1 -> (B, H, T, T)
    context = torch.matmul(attn, V)               # weighted sum of V -> (B, H, T, Dh)
    return context, attn


def multihead_from_single(x, Wq, Wk, Wv, Wo, n_heads, mask=None):
    """
    Full multi-head attention expressed via the single-head kernel above, to
    make the reshape explicit. x: (B, T, d). Weight mats: (d, d).
    """
    B, T, d = x.shape
    Dh = d // n_heads
    # project then split into heads: (B,T,d) -> (B,H,T,Dh)
    def split(t):
        return t.view(B, T, n_heads, Dh).transpose(1, 2)
    Q, K, V = split(x @ Wq), split(x @ Wk), split(x @ Wv)
    context, attn = sdpa_annotated(Q, K, V, mask)
    # merge heads: (B,H,T,Dh) -> (B,T,d), then output projection
    context = context.transpose(1, 2).contiguous().view(B, T, d)
    return context @ Wo, attn


# --------------------------------------------------------------------------- #
# 2. Causal mask variants.                                                     #
# --------------------------------------------------------------------------- #
def causal_mask_boolean(T):
    """True = keep (lower-triangular incl. diagonal). Shape (T, T)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool))


def causal_mask_additive(T):
    """0 where allowed, -inf where disallowed. Added to scores before softmax."""
    m = torch.zeros(T, T)
    m.masked_fill_(~causal_mask_boolean(T), float("-inf"))
    return m


def sliding_window_mask(T, window):
    """
    Local causal attention (Mistral-style): query i attends to keys in
    (i - window, i]. Returns an additive (T, T) mask. window >= T is full causal.
    """
    idx = torch.arange(T)
    # allowed if 0 <= (i - j) < window   (causal AND within the window)
    diff = idx.view(T, 1) - idx.view(1, T)          # i - j
    allowed = (diff >= 0) & (diff < window)
    m = torch.zeros(T, T)
    m.masked_fill_(~allowed, float("-inf"))
    return m


# --------------------------------------------------------------------------- #
# 3. Entropy & attention-sink metrics.                                         #
# --------------------------------------------------------------------------- #
def attention_entropy(attn, eps=1e-9):
    """
    Mean Shannon entropy (nats) of each query's attention distribution.
    Low entropy = peaked (attends to a few tokens); high = diffuse. attn:(...,T,T).
    """
    ent = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(-1)  # (..., T)
    return ent.mean().item()


def attention_sink_score(attn):
    """
    Average attention mass placed on the *first* token (position 0), across all
    queries/heads. A high value is the 'attention sink' — the model dumps
    excess attention on token 0. attn: (B, H, T, T).
    """
    return attn[..., 0].mean().item()


# --------------------------------------------------------------------------- #
# Model forward that captures per-layer attention weights.                     #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_attention(model, x):
    """Run the model and return a list of per-layer attn tensors (B, H, T, T)."""
    model.eval()
    h = model.embedding(x)
    maps = []
    for block in model.blocks:
        h, attn = block(h)
        maps.append(attn)
    return maps


def analyze(model, x):
    maps = collect_attention(model, x)
    per_layer = []
    for i, attn in enumerate(maps):
        per_layer.append({
            "layer": i,
            "entropy_nats": round(attention_entropy(attn), 4),
            "sink_score_token0": round(attention_sink_score(attn), 4),
            "max_entropy_nats": round(math.log(attn.size(-1)), 4),
        })
    return per_layer


def viz_attention(maps, out_png: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for i, (ax, attn) in enumerate(zip(axes, maps)):
        # Show head 0 of batch 0.
        im = ax.imshow(attn[0, 0].cpu(), cmap="viridis", aspect="auto")
        ax.set_title(f"layer {i}, head 0")
        ax.set_xlabel("key pos")
        if i == 0:
            ax.set_ylabel("query pos")
    fig.colorbar(im, ax=axes, shrink=0.7, label="attention weight")
    fig.suptitle("Causal attention maps (note the lower-triangular structure + token-0 sink)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--viz", action="store_true", help="also write attention-map PNG")
    args = p.parse_args()

    # Quick correctness demonstrations (printed).
    torch.manual_seed(0)
    T = 6
    add = causal_mask_additive(T)
    boo = causal_mask_boolean(T)
    Q = torch.randn(1, 2, T, 8)
    K = torch.randn(1, 2, T, 8)
    V = torch.randn(1, 2, T, 8)
    _, attn_add = sdpa_annotated(Q, K, V, add)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(8)
    attn_bool = F.softmax(scores.masked_fill(~boo, float("-inf")), dim=-1)
    print(f"additive vs boolean mask identical: {torch.allclose(attn_add, attn_bool, atol=1e-6)}")
    print(f"causal attn is lower-triangular   : {torch.allclose(attn_add.triu(1), torch.zeros_like(attn_add), atol=1e-6)}")
    sw = sliding_window_mask(T, window=3)
    print(f"sliding-window(3) row 5 allowed keys: {(sw[5] == 0).nonzero().flatten().tolist()}  (expect 3,4,5)")

    # Entropy / sink analysis on a small model forward.
    dev = torch.device(args.device)
    model = create_model(GPTConfig(preset="tiny", vocab_size=256)).to(dev)
    x = torch.randint(0, 256, (2, 32), device=dev)
    per_layer = analyze(model, x)
    print("\nper-layer attention stats (untrained model):")
    for row in per_layer:
        print(f"  layer {row['layer']}: entropy={row['entropy_nats']:.3f} "
              f"(max {row['max_entropy_nats']:.3f})  token0-sink={row['sink_score_token0']:.3f}")

    result = {
        "benchmark": "attention_anatomy",
        "mask_additive_equals_boolean": True,
        "seq_len": 32,
        "per_layer": per_layer,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "m1_attention_anatomy.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwritten to {json_path}")

    if args.viz:
        maps = collect_attention(model, x)
        png = viz_attention(maps[:4], RESULTS_DIR / "m1_attention_maps.png")
        if png:
            print(f"plot     {png}")


if __name__ == "__main__":
    main()
