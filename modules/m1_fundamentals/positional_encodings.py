#!/usr/bin/env python3
"""
Positional encodings: learned vs sinusoidal vs RoPE — derivations + experiment.

Transformers are permutation-equivariant: without positional information,
"dog bites man" and "man bites dog" look identical to attention. This file walks
through the three ways to inject order, then runs the money experiment: train
tiny models at sequence length L_train and measure how gracefully each
extrapolates to a *longer* length L_eval.

The production RoPE/sinusoidal cores live in `src/models/rope.py`; this file
keeps the slow, annotated reference versions plus the experiment/plot. Run:

    python modules/m1_fundamentals/positional_encodings.py            # fast (CI)
    python modules/m1_fundamentals/positional_encodings.py --full     # real training

--------------------------------------------------------------------------------
1. Learned absolute PE (GPT-2)
   A trainable vector per position, added to the token embedding. Flexible, but
   position ``max_seq_len`` and beyond have *no* embedding — zero extrapolation.

2. Sinusoidal absolute PE (original Transformer)
   Fixed sin/cos of geometric frequencies. Parameter-free and defined for any
   position, so it can *run* on longer sequences (quality still degrades because
   the model never trained on those absolute values).

3. RoPE — Rotary (LLaMA / GPT-NeoX / Qwen / Mistral default)
   Rotate Q and K by an angle ∝ absolute position, per 2-D sub-space. The
   attention dot product then depends only on the *relative* offset (m − n),
   which is exactly what generalizes across lengths. No parameters, applied in
   attention, and the basis for context-extension tricks (linear PI, NTK).
--------------------------------------------------------------------------------
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models import GPTConfig, create_model  # noqa: E402
from models.rope import RotaryEmbedding, apply_rotary  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "results"


# --------------------------------------------------------------------------- #
# Reference (educational) RoPE to make the relative-position property concrete. #
# --------------------------------------------------------------------------- #
def rope_reference_rotation_matrix(pos: int, head_dim: int, base: float = 10000.0):
    """
    Build the explicit (head_dim x head_dim) block-diagonal rotation matrix for a
    single absolute position, the way the RoPE paper defines it. Each 2-D block
    i rotates by angle pos * base^(-2i/d). This is the O(d^2) textbook form; the
    production code uses the O(d) rotate_half trick that's mathematically equal.
    """
    R = torch.zeros(head_dim, head_dim)
    for i in range(head_dim // 2):
        theta = pos * (base ** (-2.0 * i / head_dim))
        c, s = math.cos(theta), math.sin(theta)
        # rotate_half pairs dim i with dim i + d/2 (not adjacent dims), so the
        # matrix we build must use that same pairing to match apply_rotary.
        j = i + head_dim // 2
        R[i, i] = c
        R[i, j] = -s
        R[j, i] = s
        R[j, j] = c
    return R


def demo_relative_property(head_dim: int = 8, base: float = 10000.0) -> float:
    """
    Empirically confirm RoPE's defining identity:
        <RoPE(q, m), RoPE(k, n)>  depends only on (m - n).
    Returns the max deviation across offset-preserving position pairs (≈ 0).
    """
    torch.manual_seed(0)
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)
    rope = RotaryEmbedding(head_dim, max_seq_len=256, base=base)
    cos, sin = rope(256)

    def score(m, n):
        qm = apply_rotary(q.view(1, 1, 1, -1), cos[m:m + 1], sin[m:m + 1]).flatten()
        kn = apply_rotary(k.view(1, 1, 1, -1), cos[n:n + 1], sin[n:n + 1]).flatten()
        return torch.dot(qm, kn).item()

    # All pairs with the same offset must give (nearly) the same score.
    offset = 5
    scores = [score(m, m - offset) for m in range(offset, offset + 20)]
    return max(scores) - min(scores)


# --------------------------------------------------------------------------- #
# Extrapolation experiment.                                                     #
# --------------------------------------------------------------------------- #
DELAY = 4  # lookback distance for the delayed-copy task


def make_copy_batch(batch, seq_len, vocab, device, seed_offset=0):
    """
    Synthetic 'delayed-copy' task: predict the token that appeared DELAY steps
    ago, i.e. target[i] = x[i - DELAY]. Tokens are **random**, so the answer
    can't be guessed from content — the model *must* attend a fixed relative
    offset back. That makes this a clean probe of positional generalization:

    - RoPE encodes relative offset directly in the attention score, so a head
      that learned "attend DELAY back" works at any absolute position → it
      extrapolates to unseen lengths.
    - Learned absolute PE relies on position embeddings that were never trained
      past L_train, so attention breaks down on longer sequences.

    Positions < DELAY have no valid source token → target = -1 (ignored by the
    model's cross-entropy ``ignore_index``). Deterministic given the offset.
    """
    g = torch.Generator().manual_seed(1234 + seed_offset)
    x = torch.randint(0, vocab, (batch, seq_len), generator=g)
    y = torch.full((batch, seq_len), -1, dtype=torch.long)
    y[:, DELAY:] = x[:, :-DELAY]
    return x.contiguous().to(device), y.contiguous().to(device)


@torch.no_grad()
def eval_loss(model, seq_len, vocab, device, batches=4):
    model.eval()
    total = 0.0
    for b in range(batches):
        x, y = make_copy_batch(16, seq_len, vocab, device, seed_offset=1000 + b)
        _, loss = model(x, y)
        total += loss.item()
    return total / batches


def train_briefly(model, steps, seq_len, vocab, device, lr=3e-3):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        x, y = make_copy_batch(16, seq_len, vocab, device, seed_offset=step)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return loss.item()


def run_experiment(full: bool, device: str):
    dev = torch.device(device)
    vocab = 128
    l_train = 128 if full else 64
    l_eval = 256 if full else 128
    steps = 400 if full else 40
    # Models must be allocated enough max_seq_len to *run* at L_eval; learned PE
    # then has untrained position rows in [L_train, L_eval) — the whole point.
    max_seq_len = l_eval

    results = {}
    for pe in ["learned", "sinusoidal", "rope"]:
        torch.manual_seed(0)
        cfg = GPTConfig(
            vocab_size=vocab, d_model=64, n_heads=4, n_layers=3, d_ff=256,
            max_seq_len=max_seq_len, dropout=0.0, pos_encoding=pe,
        )
        model = create_model(cfg).to(dev)
        train_briefly(model, steps, l_train, vocab, dev)
        in_dist = eval_loss(model, l_train, vocab, dev)
        extrap = eval_loss(model, l_eval, vocab, dev)
        results[pe] = {
            "loss_at_train_len": round(in_dist, 4),
            "loss_at_eval_len": round(extrap, 4),
            "extrapolation_gap": round(extrap - in_dist, 4),
        }
        print(
            f"{pe:11}  L={l_train} loss={in_dist:.3f}   "
            f"L={l_eval} loss={extrap:.3f}   gap={extrap - in_dist:+.3f}"
        )
    return {
        "benchmark": "positional_encoding_extrapolation",
        "train_len": l_train,
        "eval_len": l_eval,
        "train_steps": steps,
        "results": results,
    }


def plot_extrapolation(result, out_png: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    pes = list(result["results"].keys())
    train_l = [result["results"][p]["loss_at_train_len"] for p in pes]
    eval_l = [result["results"][p]["loss_at_eval_len"] for p in pes]
    x = range(len(pes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - w / 2 for i in x], train_l, w, label=f"train len {result['train_len']}", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], eval_l, w, label=f"eval len {result['eval_len']} (extrapolation)", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pes)
    ax.set_ylabel("cross-entropy loss (lower better)")
    ax.set_title("Positional encoding: length extrapolation\n(shifted-copy task)")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--full", action="store_true", help="longer training + longer eval length")
    args = p.parse_args()

    # Quick correctness demonstrations (no training).
    dev = demo_relative_property()
    print(f"RoPE relative-position identity: max score deviation = {dev:.2e} (≈0 ✔)\n")

    result = run_experiment(args.full, args.device)
    result["rope_relative_property_deviation"] = dev

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "m1_positional_encoding.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    png_path = plot_extrapolation(result, RESULTS_DIR / "m1_positional_encoding.png")
    print(f"\nwritten to {json_path}")
    if png_path:
        print(f"plot     {png_path}")


if __name__ == "__main__":
    main()
